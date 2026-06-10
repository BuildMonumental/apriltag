#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

#include "apriltag.h"
#include "tagStandard52h13.h"
#include "common/image_u8.h"
#include "common/timeprofile.h"
#include "common/zarray.h"

static double nowMs(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static double processCpuMs(void) {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_utime.tv_sec * 1000.0 + usage.ru_utime.tv_usec / 1000.0 +
           usage.ru_stime.tv_sec * 1000.0 + usage.ru_stime.tv_usec / 1000.0;
}

static int compareDetections(const void *a, const void *b) {
    const apriltag_detection_t *da = *(apriltag_detection_t *const *)a;
    const apriltag_detection_t *db = *(apriltag_detection_t *const *)b;
    if (da->id != db->id)
        return (da->id > db->id) - (da->id < db->id);
    return (da->c[0] > db->c[0]) - (da->c[0] < db->c[0]);
}

static zarray_t *benchDetect(const char *label, apriltag_detector_t *td, image_u8_t *im, int iters) {
    zarray_t *kept = NULL;
    double walls[32];
    double cpuStart = processCpuMs();
    for (int i = 0; i < iters; i++) {
        if (kept != NULL)
            apriltag_detections_destroy(kept);
        double start = nowMs();
        kept = apriltag_detector_detect(td, im);
        walls[i] = nowMs() - start;
    }
    double cpuPer = (processCpuMs() - cpuStart) / iters;
    double median;
    {
        double sorted[32];
        memcpy(sorted, walls, sizeof(double) * iters);
        for (int i = 0; i < iters; i++)
            for (int j = i + 1; j < iters; j++)
                if (sorted[j] < sorted[i]) {
                    double t = sorted[i];
                    sorted[i] = sorted[j];
                    sorted[j] = t;
                }
        median = sorted[iters / 2];
    }
    printf("%-10s wall %6.2f ms   cpu %6.2f core-ms   detections %d\n",
           label, median, cpuPer, zarray_size(kept));
    printf("--- %s stage profile (last iteration) ---\n", label);
    timeprofile_display(td->tp);
    return kept;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s image.pgm\n", argv[0]);
        return 1;
    }
    image_u8_t *im = image_u8_create_from_pnm(argv[1]);
    if (im == NULL)
        return 1;

    apriltag_family_t *family = tagStandard52h13_create();
    apriltag_detector_t *td = apriltag_detector_create();
    apriltag_detector_add_family(td, family);
    td->nthreads = 8;
    td->quad_decimate = 1.0f;
    td->quad_sigma = 0.0f;

    unsetenv("APRILTAG_OPENCL");
    zarray_t *warmup = apriltag_detector_detect(td, im);
    apriltag_detections_destroy(warmup);
    zarray_t *cpuDetections = benchDetect("CPU", td, im, 12);

    setenv("APRILTAG_OPENCL", "1", 1);
    warmup = apriltag_detector_detect(td, im);
    apriltag_detections_destroy(warmup);
    zarray_t *gpuDetections = benchDetect("GPU", td, im, 12);
    zarray_t *gpuDetections2 = benchDetect("GPU run2", td, im, 12);

    int cpuCount = zarray_size(cpuDetections);
    int gpuCount = zarray_size(gpuDetections);
    if (cpuCount != gpuCount) {
        printf("RESULT: DETECTION COUNT MISMATCH cpu=%d gpu=%d\n", cpuCount, gpuCount);
        return 2;
    }

    apriltag_detection_t **cpuSorted = malloc(sizeof(void *) * cpuCount);
    apriltag_detection_t **gpuSorted = malloc(sizeof(void *) * gpuCount);
    for (int i = 0; i < cpuCount; i++) {
        zarray_get(cpuDetections, i, &cpuSorted[i]);
        zarray_get(gpuDetections, i, &gpuSorted[i]);
    }
    qsort(cpuSorted, cpuCount, sizeof(void *), compareDetections);
    qsort(gpuSorted, gpuCount, sizeof(void *), compareDetections);

    int idMismatches = 0, hammingMismatches = 0;
    double maxCornerDelta = 0.0, maxCenterDelta = 0.0;
    for (int i = 0; i < cpuCount; i++) {
        if (cpuSorted[i]->id != gpuSorted[i]->id) {
            idMismatches++;
            continue;
        }
        if (cpuSorted[i]->hamming != gpuSorted[i]->hamming)
            hammingMismatches++;
        for (int corner = 0; corner < 4; corner++) {
            for (int axis = 0; axis < 2; axis++) {
                double delta = fabs(cpuSorted[i]->p[corner][axis] - gpuSorted[i]->p[corner][axis]);
                if (delta > maxCornerDelta)
                    maxCornerDelta = delta;
            }
        }
        for (int axis = 0; axis < 2; axis++) {
            double delta = fabs(cpuSorted[i]->c[axis] - gpuSorted[i]->c[axis]);
            if (delta > maxCenterDelta)
                maxCenterDelta = delta;
        }
    }

    printf("RESULT: %d detections both modes, id mismatches %d, hamming mismatches %d\n",
           cpuCount, idMismatches, hammingMismatches);
    printf("RESULT: max corner delta %.6f px, max center delta %.6f px\n",
           maxCornerDelta, maxCenterDelta);

    // GPU run-to-run determinism: same detections across two GPU runs?
    double gpuRunDelta = 0.0;
    int gpuRunIdMismatch = (zarray_size(gpuDetections2) != gpuCount);
    if (!gpuRunIdMismatch) {
        apriltag_detection_t **gpu2Sorted = malloc(sizeof(void *) * gpuCount);
        for (int i = 0; i < gpuCount; i++)
            zarray_get(gpuDetections2, i, &gpu2Sorted[i]);
        qsort(gpu2Sorted, gpuCount, sizeof(void *), compareDetections);
        for (int i = 0; i < gpuCount; i++) {
            if (gpuSorted[i]->id != gpu2Sorted[i]->id) {
                gpuRunIdMismatch = 1;
                break;
            }
            for (int corner = 0; corner < 4; corner++)
                for (int axis = 0; axis < 2; axis++) {
                    double delta = fabs(gpuSorted[i]->p[corner][axis] - gpu2Sorted[i]->p[corner][axis]);
                    if (delta > gpuRunDelta)
                        gpuRunDelta = delta;
                }
        }
    }
    printf("RESULT: gpu run-to-run max corner delta %.6f px%s\n",
           gpuRunDelta, gpuRunIdMismatch ? " (ID MISMATCH)" : "");

    // 0.05 px gate: point-order differences within clusters shift fitted
    // corners at this scale; vide reprojection errors are an order of
    // magnitude larger.
    int pass = (idMismatches == 0) && (hammingMismatches == 0) && (maxCornerDelta < 0.05) && !gpuRunIdMismatch;
    printf("RESULT: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 2;
}
