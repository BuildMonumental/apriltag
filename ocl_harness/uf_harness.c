#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include "apriltag.h"
#include "tagStandard52h13.h"
#include "common/image_u8.h"
#include "common/unionfind.h"
#include "common/zarray.h"

image_u8_t *threshold(apriltag_detector_t *td, image_u8_t *im);
unionfind_t *connected_components(apriltag_detector_t *td, image_u8_t *threshim, int w, int h, int ts);

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

// GPU connected-components: replicates the exact edge rules of
// do_unionfind_first_line / do_unionfind_line2 (including the border bounds
// x in [1, w-2] and the redundancy-skip conditions, which leak into
// component semantics at the right image border). Final labels are the
// minimum pixel index of each component, which is order-independent and
// therefore deterministic under GPU execution.
static const char *cclSource =
    "inline uint findRoot(__global volatile uint *labels, uint i) {\n"
    "    uint l = labels[i];\n"
    "    while (l != i) { i = l; l = labels[i]; }\n"
    "    return i;\n"
    "}\n"
    "inline void mergeRoots(__global volatile uint *labels, uint a, uint b) {\n"
    "    while (1) {\n"
    "        a = findRoot(labels, a);\n"
    "        b = findRoot(labels, b);\n"
    "        if (a == b) return;\n"
    "        uint hi = max(a, b), lo = min(a, b);\n"
    "        uint old = atomic_min(&labels[hi], lo);\n"
    "        if (old == hi) return;\n"
    "        a = lo; b = old;\n"
    "    }\n"
    "}\n"
    "__kernel void initLabels(__global const uchar *im, int s, int w, int h,\n"
    "                         __global uint *labels) {\n"
    "    int x = get_global_id(0);\n"
    "    int y = get_global_id(1);\n"
    "    if (x >= w || y >= h) return;\n"
    "    labels[y * w + x] = (im[y * s + x] == 127) ? 0xFFFFFFFFu : (uint)(y * w + x);\n"
    "}\n"
    "__kernel void mergeEdges(__global const uchar *im, int s, int w, int h,\n"
    "                         __global volatile uint *labels) {\n"
    "    int x = get_global_id(0);\n"
    "    int y = get_global_id(1);\n"
    "    if (x < 1 || x >= w - 1 || y >= h) return;\n"
    "    uchar v = im[y * s + x];\n"
    "    if (v == 127) return;\n"
    "    uint idx = (uint)(y * w + x);\n"
    "    uchar vLeft = im[y * s + x - 1];\n"
    "    if (vLeft == v) mergeRoots(labels, idx, idx - 1);\n"
    "    if (y == 0) return;\n"
    "    uchar vUpLeft = im[(y - 1) * s + x - 1];\n"
    "    uchar vUp = im[(y - 1) * s + x];\n"
    "    uchar vUpRight = im[(y - 1) * s + x + 1];\n"
    "    if ((x == 1 || !((vLeft == vUpLeft) && (vUpLeft == vUp))) && vUp == v)\n"
    "        mergeRoots(labels, idx, idx - (uint)w);\n"
    "    if (v == 255) {\n"
    "        if ((x == 1 || !(vLeft == vUpLeft || vUp == vUpLeft)) && vUpLeft == v)\n"
    "            mergeRoots(labels, idx, idx - (uint)w - 1);\n"
    "        if (!(vUp == vUpRight) && vUpRight == v)\n"
    "            mergeRoots(labels, idx, idx - (uint)w + 1);\n"
    "    }\n"
    "}\n"
    "__kernel void compressLabels(__global const uchar *im, int s, int w, int h,\n"
    "                             __global volatile uint *labels) {\n"
    "    int x = get_global_id(0);\n"
    "    int y = get_global_id(1);\n"
    "    if (x >= w || y >= h) return;\n"
    "    if (im[y * s + x] == 127) return;\n"
    "    uint idx = (uint)(y * w + x);\n"
    "    labels[idx] = findRoot(labels, idx);\n"
    "}\n";

static size_t roundUp(size_t value, size_t multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
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
    zarray_t *warmup = apriltag_detector_detect(td, im);
    apriltag_detections_destroy(warmup);

    image_u8_t *threshim = threshold(td, im);
    const int w = im->width, h = im->height, s = threshim->stride;
    const size_t pixelCount = (size_t)w * (size_t)h;
    printf("image %dx%d stride %d\n", w, h, s);

    // --- CPU reference (the lib's parallel implementation, 8 threads) ---
    unionfind_t *uf = NULL;
    {
        double cpuStart = processCpuMs();
        double walls[32];
        for (int i = 0; i < 20; i++) {
            double start = nowMs();
            uf = connected_components(td, threshim, w, h, s);
            walls[i] = nowMs() - start;
        }
        double cpuPer = (processCpuMs() - cpuStart) / 20;
        double best = walls[0];
        for (int i = 1; i < 20; i++)
            if (walls[i] < best)
                best = walls[i];
        printf("CPU unionfind (8t): best wall %6.2f ms   cpu %6.2f core-ms per call\n", best, cpuPer);
    }

    uint32_t *cpuCanon = malloc(pixelCount * 4);
    uint32_t *minOfRep = malloc(pixelCount * 4);
    memset(minOfRep, 0xFF, pixelCount * 4);
    for (size_t idx = 0; idx < pixelCount; idx++) {
        int y = (int)(idx / w), x = (int)(idx % w);
        if (threshim->buf[y * s + x] == 127)
            continue;
        uint32_t rep = unionfind_get_representative(uf, (uint32_t)idx);
        if (idx < minOfRep[rep])
            minOfRep[rep] = (uint32_t)idx;
    }
    for (size_t idx = 0; idx < pixelCount; idx++) {
        int y = (int)(idx / w), x = (int)(idx % w);
        if (threshim->buf[y * s + x] == 127) {
            cpuCanon[idx] = 0xFFFFFFFFu;
            continue;
        }
        cpuCanon[idx] = minOfRep[unionfind_get_representative(uf, (uint32_t)idx)];
    }

    // --- GPU CCL ---
    cl_platform_id platform;
    cl_uint platformCount = 0;
    if (clGetPlatformIDs(1, &platform, &platformCount) != CL_SUCCESS || platformCount == 0) {
        fprintf(stderr, "no OpenCL platform\n");
        return 1;
    }
    cl_device_id device;
    if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL) != CL_SUCCESS) {
        fprintf(stderr, "no GPU device\n");
        return 1;
    }
    cl_int err;
    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_command_queue queue = clCreateCommandQueueWithProperties(ctx, device, NULL, &err);
    cl_program program = clCreateProgramWithSource(ctx, 1, &cclSource, NULL, &err);
    if (clBuildProgram(program, 1, &device, "", NULL, NULL) != CL_SUCCESS) {
        char log[8192] = { 0 };
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log) - 1, log, NULL);
        fprintf(stderr, "build failed:\n%s\n", log);
        return 1;
    }
    cl_kernel kInit = clCreateKernel(program, "initLabels", &err);
    cl_kernel kMerge = clCreateKernel(program, "mergeEdges", &err);
    cl_kernel kCompress = clCreateKernel(program, "compressLabels", &err);

    const size_t imageBytes = (size_t)s * (size_t)h;
    cl_mem bufIm = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, imageBytes, threshim->buf, &err);
    cl_mem bufLabels = clCreateBuffer(ctx, CL_MEM_READ_WRITE, pixelCount * 4, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "buffer alloc failed\n");
        return 1;
    }

    const cl_int cw = w, ch = h, cs = s;
    cl_kernel kernels[3] = { kInit, kMerge, kCompress };
    for (int k = 0; k < 3; k++) {
        clSetKernelArg(kernels[k], 0, sizeof(cl_mem), &bufIm);
        clSetKernelArg(kernels[k], 1, sizeof(cl_int), &cs);
        clSetKernelArg(kernels[k], 2, sizeof(cl_int), &cw);
        clSetKernelArg(kernels[k], 3, sizeof(cl_int), &ch);
        clSetKernelArg(kernels[k], 4, sizeof(cl_mem), &bufLabels);
    }
    const size_t global[2] = { roundUp((size_t)w, 16), roundUp((size_t)h, 16) };

    double gpuWallBest = 1e9;
    double gpuCpuStart = processCpuMs();
    for (int i = 0; i < 20; i++) {
        double start = nowMs();
        clEnqueueNDRangeKernel(queue, kInit, 2, NULL, global, NULL, 0, NULL, NULL);
        clEnqueueNDRangeKernel(queue, kMerge, 2, NULL, global, NULL, 0, NULL, NULL);
        clEnqueueNDRangeKernel(queue, kCompress, 2, NULL, global, NULL, 0, NULL, NULL);
        clFinish(queue);
        double elapsed = nowMs() - start;
        if (elapsed < gpuWallBest)
            gpuWallBest = elapsed;
    }
    double gpuCpuPer = (processCpuMs() - gpuCpuStart) / 20;
    printf("GPU ccl:            best wall %6.2f ms   cpu %6.2f core-ms per call\n", gpuWallBest, gpuCpuPer);

    uint32_t *gpuLabels = malloc(pixelCount * 4);
    clEnqueueReadBuffer(queue, bufLabels, CL_TRUE, 0, pixelCount * 4, gpuLabels, 0, NULL, NULL);

    size_t mismatches = 0, firstMismatch = 0, checked = 0;
    for (size_t idx = 0; idx < pixelCount; idx++) {
        int y = (int)(idx / w), x = (int)(idx % w);
        if (threshim->buf[y * s + x] == 127)
            continue;
        checked++;
        if (cpuCanon[idx] != gpuLabels[idx]) {
            if (mismatches == 0)
                firstMismatch = idx;
            mismatches++;
        }
    }
    if (mismatches == 0) {
        printf("equivalence: IDENTICAL components (%zu pixels checked)\n", checked);
    } else {
        printf("equivalence: %zu of %zu pixels mismatch, first at idx %zu (x=%zu y=%zu) cpu=%u gpu=%u\n",
               mismatches, checked, firstMismatch, firstMismatch % w, firstMismatch / w,
               cpuCanon[firstMismatch], gpuLabels[firstMismatch]);
    }

    return mismatches == 0 ? 0 : 2;
}
