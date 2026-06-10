#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

#include "apriltag.h"
#include "tag16h5.h"
#include "tag36h11.h"
#include "tagCustom48h12.h"
#include "tagStandard52h13.h"
#include "common/image_u8.h"
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

typedef struct {
    const char *name;
    apriltag_family_t *(*create)(void);
    void (*destroy)(apriltag_family_t *);
} FamilyEntry;

static const FamilyEntry families[] = {
    { "tagStandard52h13", tagStandard52h13_create, tagStandard52h13_destroy },
    { "tag36h11", tag36h11_create, tag36h11_destroy },
    { "tagCustom48h12", tagCustom48h12_create, tagCustom48h12_destroy },
    { "tag16h5", tag16h5_create, tag16h5_destroy },
};

static int compareDetections(const void *a, const void *b) {
    const apriltag_detection_t *da = *(apriltag_detection_t *const *)a;
    const apriltag_detection_t *db = *(apriltag_detection_t *const *)b;
    if (da->id != db->id)
        return (da->id > db->id) - (da->id < db->id);
    if (da->c[0] != db->c[0])
        return (da->c[0] > db->c[0]) - (da->c[0] < db->c[0]);
    return (da->c[1] > db->c[1]) - (da->c[1] < db->c[1]);
}

static zarray_t *runDetect(apriltag_detector_t *td, image_u8_t *im, int gpu,
                           double *wallMs, double *cpuMs) {
    if (gpu)
        setenv("APRILTAG_OPENCL", "1", 1);
    else
        unsetenv("APRILTAG_OPENCL");
    zarray_t *warmup = apriltag_detector_detect(td, im);
    apriltag_detections_destroy(warmup);
    double cpuStart = processCpuMs();
    double start = nowMs();
    zarray_t *detections = apriltag_detector_detect(td, im);
    *wallMs = nowMs() - start;
    *cpuMs = processCpuMs() - cpuStart;
    return detections;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s image.pgm [image.pgm ...]\n", argv[0]);
        return 1;
    }
    const double decimates[] = { 1.0, 1.5, 2.0 };
    int totalCases = 0, passedCases = 0;
    double cpuWallSum = 0, cpuCpuSum = 0, gpuWallSum = 0, gpuCpuSum = 0;
    double worstCornerDelta = 0.0;

    for (int imgIdx = 1; imgIdx < argc; imgIdx++) {
        image_u8_t *im = image_u8_create_from_pnm(argv[imgIdx]);
        if (im == NULL) {
            fprintf(stderr, "failed to load %s\n", argv[imgIdx]);
            return 1;
        }
        const char *shortName = strrchr(argv[imgIdx], '/');
        shortName = shortName != NULL ? shortName + 1 : argv[imgIdx];

        for (size_t decIdx = 0; decIdx < 3; decIdx++) {
            for (size_t famIdx = 0; famIdx < sizeof(families) / sizeof(families[0]); famIdx++) {
                apriltag_family_t *family = families[famIdx].create();
                apriltag_detector_t *td = apriltag_detector_create();
                apriltag_detector_add_family(td, family);
                td->nthreads = 8;
                td->quad_decimate = (float)decimates[decIdx];
                td->quad_sigma = 0.0f;

                double cpuWall, cpuCpu, gpuWall, gpuCpu;
                zarray_t *cpuDet = runDetect(td, im, 0, &cpuWall, &cpuCpu);
                zarray_t *gpuDet = runDetect(td, im, 1, &gpuWall, &gpuCpu);
                cpuWallSum += cpuWall;
                cpuCpuSum += cpuCpu;
                gpuWallSum += gpuWall;
                gpuCpuSum += gpuCpu;

                int cpuCount = zarray_size(cpuDet);
                int gpuCount = zarray_size(gpuDet);
                double maxDelta = 0.0;
                int idMismatch = 0, hammingMismatch = 0;
                if (cpuCount == gpuCount && cpuCount > 0) {
                    apriltag_detection_t **cs = malloc(sizeof(void *) * cpuCount);
                    apriltag_detection_t **gs = malloc(sizeof(void *) * cpuCount);
                    for (int i = 0; i < cpuCount; i++) {
                        zarray_get(cpuDet, i, &cs[i]);
                        zarray_get(gpuDet, i, &gs[i]);
                    }
                    qsort(cs, cpuCount, sizeof(void *), compareDetections);
                    qsort(gs, cpuCount, sizeof(void *), compareDetections);
                    for (int i = 0; i < cpuCount; i++) {
                        if (cs[i]->id != gs[i]->id) {
                            idMismatch++;
                            continue;
                        }
                        if (cs[i]->hamming != gs[i]->hamming)
                            hammingMismatch++;
                        for (int corner = 0; corner < 4; corner++)
                            for (int axis = 0; axis < 2; axis++) {
                                double d = fabs(cs[i]->p[corner][axis] - gs[i]->p[corner][axis]);
                                if (d > maxDelta)
                                    maxDelta = d;
                            }
                    }
                    free(cs);
                    free(gs);
                }
                int pass = (cpuCount == gpuCount) && (idMismatch == 0) &&
                           (hammingMismatch == 0) && (maxDelta < 0.05);
                totalCases++;
                if (pass)
                    passedCases++;
                if (maxDelta > worstCornerDelta)
                    worstCornerDelta = maxDelta;
                printf("%-22s dec %.1f %-18s cpu %3d gpu %3d maxd %.4f  %s\n",
                       shortName, decimates[decIdx], families[famIdx].name,
                       cpuCount, gpuCount, maxDelta, pass ? "PASS" : "FAIL");
                if (!pass)
                    printf("    DETAIL: idMismatch=%d hammingMismatch=%d\n", idMismatch, hammingMismatch);

                apriltag_detections_destroy(cpuDet);
                apriltag_detections_destroy(gpuDet);
                apriltag_detector_destroy(td);
                families[famIdx].destroy(family);
            }
        }
        image_u8_destroy(im);
    }

    printf("\nSUMMARY: %d/%d cases passed, worst corner delta %.4f px\n",
           passedCases, totalCases, worstCornerDelta);
    printf("SUMMARY: corpus totals  cpu %.0f ms wall / %.0f core-ms   gpu %.0f ms wall / %.0f core-ms\n",
           cpuWallSum, cpuCpuSum, gpuWallSum, gpuCpuSum);
    return passedCases == totalCases ? 0 : 2;
}
