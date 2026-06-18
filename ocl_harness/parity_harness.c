#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

#include "apriltag.h"
#include "tagStandard52h13.h"
#include "common/image_u8.h"
#include "common/zarray.h"

image_u8_t *threshold(apriltag_detector_t *td, image_u8_t *im);

static double nowMs(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static double processCpuMs(void) {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    double user = usage.ru_utime.tv_sec * 1000.0 + usage.ru_utime.tv_usec / 1000.0;
    double sys = usage.ru_stime.tv_sec * 1000.0 + usage.ru_stime.tv_usec / 1000.0;
    return user + sys;
}

static int compareDoubles(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

static void benchThreshold(const char *label, apriltag_detector_t *td, image_u8_t *im, int iters) {
    double times[64];
    double cpuStart = processCpuMs();
    for (int i = 0; i < iters; i++) {
        double start = nowMs();
        image_u8_t *result = threshold(td, im);
        times[i] = nowMs() - start;
        image_u8_destroy(result);
    }
    double cpuPerCall = (processCpuMs() - cpuStart) / iters;
    qsort(times, iters, sizeof(double), compareDoubles);
    printf("%-14s wall %6.2f ms   cpu %6.2f core-ms per call\n", label, times[iters / 2], cpuPerCall);
}

static void benchDetect(const char *label, apriltag_detector_t *td, image_u8_t *im, int iters) {
    double times[64];
    int tagCount = 0;
    double cpuStart = processCpuMs();
    for (int i = 0; i < iters; i++) {
        double start = nowMs();
        zarray_t *detections = apriltag_detector_detect(td, im);
        times[i] = nowMs() - start;
        tagCount = zarray_size(detections);
        apriltag_detections_destroy(detections);
    }
    double cpuPerCall = (processCpuMs() - cpuStart) / iters;
    qsort(times, iters, sizeof(double), compareDoubles);
    printf("%-14s wall %6.2f ms   cpu %6.2f core-ms per call   tags %d\n",
           label, times[iters / 2], cpuPerCall, tagCount);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s image.pgm\n", argv[0]);
        return 1;
    }
    image_u8_t *im = image_u8_create_from_pnm(argv[1]);
    if (im == NULL) {
        fprintf(stderr, "failed to load %s\n", argv[1]);
        return 1;
    }

    apriltag_family_t *family = tagStandard52h13_create();
    apriltag_detector_t *td = apriltag_detector_create();
    apriltag_detector_add_family(td, family);
    td->nthreads = 8;
    td->quad_decimate = 1.0f;
    td->quad_sigma = 0.0f;

    unsetenv("APRILTAG_OPENCL");
    zarray_t *warmupDetections = apriltag_detector_detect(td, im);
    apriltag_detections_destroy(warmupDetections);

    image_u8_t *cpuResult = threshold(td, im);
    setenv("APRILTAG_OPENCL", "1", 1);
    image_u8_t *gpuResult = threshold(td, im);

    size_t bytes = (size_t)cpuResult->stride * cpuResult->height;
    int identical = memcmp(cpuResult->buf, gpuResult->buf, bytes) == 0;
    if (identical) {
        printf("parity: IDENTICAL (%zu bytes)\n", bytes);
    } else {
        size_t diffCount = 0, firstDiff = 0;
        for (size_t i = 0; i < bytes; i++) {
            if (cpuResult->buf[i] != gpuResult->buf[i]) {
                if (diffCount == 0)
                    firstDiff = i;
                diffCount++;
            }
        }
        printf("parity: MISMATCH (%zu of %zu bytes differ, first at %zu: cpu=%d gpu=%d)\n",
               diffCount, bytes, firstDiff, cpuResult->buf[firstDiff], gpuResult->buf[firstDiff]);
    }
    image_u8_destroy(cpuResult);
    image_u8_destroy(gpuResult);

    setenv("APRILTAG_OPENCL", "1", 1);
    benchThreshold("thresh GPU", td, im, 30);
    unsetenv("APRILTAG_OPENCL");
    benchThreshold("thresh CPU", td, im, 30);

    setenv("APRILTAG_OPENCL", "1", 1);
    benchDetect("detect GPUthr", td, im, 15);
    unsetenv("APRILTAG_OPENCL");
    benchDetect("detect CPU", td, im, 15);

    apriltag_detector_destroy(td);
    tagStandard52h13_destroy(family);
    image_u8_destroy(im);
    return identical ? 0 : 2;
}
