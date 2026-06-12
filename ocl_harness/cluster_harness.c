#include <stdint.h>
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
zarray_t *gradient_clusters(apriltag_detector_t *td, image_u8_t *threshim, int w, int h, int ts, unionfind_t *uf);

struct pt {
    uint16_t x, y;
    int16_t gx, gy;
    float slope;
};

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

// CCL kernels are identical to uf_harness.c; extractPairs replicates
// do_gradient_clusters' emit rules, including the connected_last dedup
// (computable per-pixel from the left neighbour's would-emit state).
static const char *clSource =
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
    "    int x = get_global_id(0), y = get_global_id(1);\n"
    "    if (x >= w || y >= h) return;\n"
    "    labels[y * w + x] = (im[y * s + x] == 127) ? 0xFFFFFFFFu : (uint)(y * w + x);\n"
    "}\n"
    "__kernel void mergeEdges(__global const uchar *im, int s, int w, int h,\n"
    "                         __global volatile uint *labels) {\n"
    "    int x = get_global_id(0), y = get_global_id(1);\n"
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
    "    int x = get_global_id(0), y = get_global_id(1);\n"
    "    if (x >= w || y >= h) return;\n"
    "    if (im[y * s + x] == 127) return;\n"
    "    uint idx = (uint)(y * w + x);\n"
    "    labels[idx] = findRoot(labels, idx);\n"
    "}\n"
    "__kernel void countSizes(__global const uchar *im, int s, int w, int h,\n"
    "                         __global const uint *labels, __global volatile uint *sizes) {\n"
    "    int x = get_global_id(0), y = get_global_id(1);\n"
    "    if (x >= w || y >= h) return;\n"
    "    if (im[y * s + x] == 127) return;\n"
    "    atomic_inc(&sizes[labels[y * w + x]]);\n"
    "}\n"
    "inline int wouldEmit(__global const uchar *im, __global const uint *labels,\n"
    "                     __global const uint *sizes, int s, int w, uint minCluster,\n"
    "                     int x, int y, int dx, int dy) {\n"
    "    uchar v0 = im[y * s + x];\n"
    "    if (v0 == 127) return 0;\n"
    "    if (sizes[labels[y * w + x]] < minCluster) return 0;\n"
    "    uchar v1 = im[(y + dy) * s + x + dx];\n"
    "    if ((int)v0 + (int)v1 != 255) return 0;\n"
    "    if (sizes[labels[(y + dy) * w + x + dx]] < minCluster) return 0;\n"
    "    return 1;\n"
    "}\n"
    "inline void emitPair(__global const uchar *im, __global const uint *labels,\n"
    "                     int s, int w, int x, int y, int dx, int dy,\n"
    "                     __global volatile uint *counter, __global ulong2 *records, uint capacity) {\n"
    "    uchar v0 = im[y * s + x];\n"
    "    uchar v1 = im[(y + dy) * s + x + dx];\n"
    "    uint rep0 = labels[y * w + x];\n"
    "    uint rep1 = labels[(y + dy) * w + x + dx];\n"
    "    ulong key = (rep0 < rep1) ? (((ulong)rep1 << 32) | rep0) : (((ulong)rep0 << 32) | rep1);\n"
    "    int grad = (int)v1 - (int)v0;\n"
    "    ushort px = (ushort)(2 * x + dx), py = (ushort)(2 * y + dy);\n"
    "    ushort pgx = (ushort)(short)(dx * grad), pgy = (ushort)(short)(dy * grad);\n"
    "    ulong packed = ((ulong)px << 48) | ((ulong)py << 32) | ((ulong)pgx << 16) | (ulong)pgy;\n"
    "    uint slot = atomic_inc(counter);\n"
    "    if (slot < capacity) records[slot] = (ulong2)(key, packed);\n"
    "}\n"
    "__kernel void extractPairs(__global const uchar *im, int s, int w, int h,\n"
    "                           __global const uint *labels, __global const uint *sizes,\n"
    "                           uint minCluster, __global volatile uint *counter,\n"
    "                           __global ulong2 *records, uint capacity) {\n"
    "    int x = get_global_id(0), y = get_global_id(1);\n"
    "    if (x < 1 || x >= w - 1 || y < 1 || y >= h - 1) return;\n"
    "    uchar v0 = im[y * s + x];\n"
    "    if (v0 == 127) return;\n"
    "    if (sizes[labels[y * w + x]] < minCluster) return;\n"
    "    if (wouldEmit(im, labels, sizes, s, w, minCluster, x, y, 1, 0))\n"
    "        emitPair(im, labels, s, w, x, y, 1, 0, counter, records, capacity);\n"
    "    if (wouldEmit(im, labels, sizes, s, w, minCluster, x, y, 0, 1))\n"
    "        emitPair(im, labels, s, w, x, y, 0, 1, counter, records, capacity);\n"
    "    int prevEmitted = (x > 1) && wouldEmit(im, labels, sizes, s, w, minCluster, x - 1, y, 1, 1);\n"
    "    if (!prevEmitted && wouldEmit(im, labels, sizes, s, w, minCluster, x, y, -1, 1))\n"
    "        emitPair(im, labels, s, w, x, y, -1, 1, counter, records, capacity);\n"
    "    if (wouldEmit(im, labels, sizes, s, w, minCluster, x, y, 1, 1))\n"
    "        emitPair(im, labels, s, w, x, y, 1, 1, counter, records, capacity);\n"
    "}\n";

static size_t roundUp(size_t value, size_t multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

static int compareU64(const void *a, const void *b) {
    uint64_t da = *(const uint64_t *)a, db = *(const uint64_t *)b;
    return (da > db) - (da < db);
}

static int compareRecords(const void *a, const void *b) {
    const uint64_t *ra = (const uint64_t *)a, *rb = (const uint64_t *)b;
    if (ra[0] != rb[0])
        return (ra[0] > rb[0]) - (ra[0] < rb[0]);
    return (ra[1] > rb[1]) - (ra[1] < rb[1]);
}

typedef struct {
    uint64_t *pts;
    size_t count;
} Blob;

static int compareBlobs(const void *a, const void *b) {
    const Blob *ba = (const Blob *)a, *bb = (const Blob *)b;
    if (ba->count != bb->count)
        return (ba->count > bb->count) - (ba->count < bb->count);
    return memcmp(ba->pts, bb->pts, ba->count * 8);
}

static uint64_t packPt(const struct pt *p) {
    return ((uint64_t)p->x << 48) | ((uint64_t)p->y << 32) |
           ((uint64_t)(uint16_t)p->gx << 16) | (uint64_t)(uint16_t)p->gy;
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

    image_u8_t *threshim = threshold(td, im);
    const int w = im->width, h = im->height, s = threshim->stride;
    const size_t pixelCount = (size_t)w * (size_t)h;
    const uint32_t minCluster = (uint32_t)td->qtp.min_cluster_pixels;
    printf("image %dx%d stride %d, min_cluster_pixels %u\n", w, h, s, minCluster);

    unionfind_t *uf = connected_components(td, threshim, w, h, s);

    // CPU clusters timing + reference content
    zarray_t *cpuClusters = NULL;
    {
        double walls[8];
        double cpuStart = processCpuMs();
        for (int i = 0; i < 8; i++) {
            if (cpuClusters != NULL) {
                for (int c = 0; c < zarray_size(cpuClusters); c++) {
                    zarray_t *cl;
                    zarray_get(cpuClusters, c, &cl);
                    zarray_destroy(cl);
                }
                zarray_destroy(cpuClusters);
            }
            double start = nowMs();
            cpuClusters = gradient_clusters(td, threshim, w, h, s, uf);
            walls[i] = nowMs() - start;
        }
        double cpuPer = (processCpuMs() - cpuStart) / 8;
        double best = walls[0];
        for (int i = 1; i < 8; i++)
            if (walls[i] < best)
                best = walls[i];
        printf("CPU clusters (8t):   best wall %6.2f ms   cpu %6.2f core-ms per call   clusters %d\n",
               best, cpuPer, zarray_size(cpuClusters));
    }

    // GPU setup
    cl_platform_id platform;
    cl_device_id device;
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    cl_int err;
    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_command_queue queue = clCreateCommandQueueWithProperties(ctx, device, NULL, &err);
    cl_program program = clCreateProgramWithSource(ctx, 1, &clSource, NULL, &err);
    if (clBuildProgram(program, 1, &device, "", NULL, NULL) != CL_SUCCESS) {
        char log[8192] = { 0 };
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log) - 1, log, NULL);
        fprintf(stderr, "build failed:\n%s\n", log);
        return 1;
    }
    cl_kernel kInit = clCreateKernel(program, "initLabels", &err);
    cl_kernel kMerge = clCreateKernel(program, "mergeEdges", &err);
    cl_kernel kCompress = clCreateKernel(program, "compressLabels", &err);
    cl_kernel kSizes = clCreateKernel(program, "countSizes", &err);
    cl_kernel kExtract = clCreateKernel(program, "extractPairs", &err);

    const size_t imageBytes = (size_t)s * (size_t)h;
    const cl_uint capacity = 8 * 1024 * 1024;
    cl_mem bufIm = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, imageBytes, threshim->buf, &err);
    cl_mem bufLabels = clCreateBuffer(ctx, CL_MEM_READ_WRITE, pixelCount * 4, NULL, &err);
    cl_mem bufSizes = clCreateBuffer(ctx, CL_MEM_READ_WRITE, pixelCount * 4, NULL, &err);
    cl_mem bufCounter = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 4, NULL, &err);
    cl_mem bufRecords = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)capacity * 16, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "buffer alloc failed\n");
        return 1;
    }

    const cl_int cw = w, ch = h, cs = s;
    cl_kernel layoutKernels[4] = { kInit, kMerge, kCompress, kSizes };
    for (int k = 0; k < 4; k++) {
        clSetKernelArg(layoutKernels[k], 0, sizeof(cl_mem), &bufIm);
        clSetKernelArg(layoutKernels[k], 1, sizeof(cl_int), &cs);
        clSetKernelArg(layoutKernels[k], 2, sizeof(cl_int), &cw);
        clSetKernelArg(layoutKernels[k], 3, sizeof(cl_int), &ch);
        clSetKernelArg(layoutKernels[k], 4, sizeof(cl_mem), &bufLabels);
    }
    clSetKernelArg(kSizes, 5, sizeof(cl_mem), &bufSizes);
    clSetKernelArg(kExtract, 0, sizeof(cl_mem), &bufIm);
    clSetKernelArg(kExtract, 1, sizeof(cl_int), &cs);
    clSetKernelArg(kExtract, 2, sizeof(cl_int), &cw);
    clSetKernelArg(kExtract, 3, sizeof(cl_int), &ch);
    clSetKernelArg(kExtract, 4, sizeof(cl_mem), &bufLabels);
    clSetKernelArg(kExtract, 5, sizeof(cl_mem), &bufSizes);
    clSetKernelArg(kExtract, 6, sizeof(cl_uint), &minCluster);
    clSetKernelArg(kExtract, 7, sizeof(cl_mem), &bufCounter);
    clSetKernelArg(kExtract, 8, sizeof(cl_mem), &bufRecords);
    clSetKernelArg(kExtract, 9, sizeof(cl_uint), &capacity);

    const size_t global[2] = { roundUp((size_t)w, 16), roundUp((size_t)h, 16) };
    const cl_uint zero = 0;

    double gpuWallBest = 1e9;
    double gpuCpuStart = processCpuMs();
    for (int i = 0; i < 20; i++) {
        double start = nowMs();
        clEnqueueFillBuffer(queue, bufSizes, &zero, 4, 0, pixelCount * 4, 0, NULL, NULL);
        clEnqueueFillBuffer(queue, bufCounter, &zero, 4, 0, 4, 0, NULL, NULL);
        clEnqueueNDRangeKernel(queue, kInit, 2, NULL, global, NULL, 0, NULL, NULL);
        clEnqueueNDRangeKernel(queue, kMerge, 2, NULL, global, NULL, 0, NULL, NULL);
        clEnqueueNDRangeKernel(queue, kCompress, 2, NULL, global, NULL, 0, NULL, NULL);
        clEnqueueNDRangeKernel(queue, kSizes, 2, NULL, global, NULL, 0, NULL, NULL);
        clEnqueueNDRangeKernel(queue, kExtract, 2, NULL, global, NULL, 0, NULL, NULL);
        clFinish(queue);
        double elapsed = nowMs() - start;
        if (elapsed < gpuWallBest)
            gpuWallBest = elapsed;
    }
    double gpuCpuPer = (processCpuMs() - gpuCpuStart) / 20;

    cl_uint recordCount = 0;
    clEnqueueReadBuffer(queue, bufCounter, CL_TRUE, 0, 4, &recordCount, 0, NULL, NULL);
    printf("GPU frontend chain:  best wall %6.2f ms   cpu %6.2f core-ms per call   records %u\n",
           gpuWallBest, gpuCpuPer, recordCount);
    if (recordCount > capacity) {
        fprintf(stderr, "record overflow\n");
        return 1;
    }

    uint64_t *records = malloc((size_t)recordCount * 16);
    clEnqueueReadBuffer(queue, bufRecords, CL_TRUE, 0, (size_t)recordCount * 16, records, 0, NULL, NULL);

    double sortStart = nowMs();
    qsort(records, recordCount, 16, compareRecords);
    printf("validation-only CPU qsort of records: %.2f ms\n", nowMs() - sortStart);

    // GPU blobs: group sorted records by key
    Blob *gpuBlobs = malloc(sizeof(Blob) * (recordCount > 0 ? recordCount : 1));
    size_t gpuBlobCount = 0;
    size_t groupStart = 0;
    for (size_t i = 1; i <= recordCount; i++) {
        if (i == recordCount || records[2 * i] != records[2 * groupStart]) {
            size_t count = i - groupStart;
            uint64_t *pts = malloc(count * 8);
            for (size_t j = 0; j < count; j++)
                pts[j] = records[2 * (groupStart + j) + 1];
            gpuBlobs[gpuBlobCount].pts = pts;
            gpuBlobs[gpuBlobCount].count = count;
            gpuBlobCount++;
            groupStart = i;
        }
    }

    // CPU blobs
    size_t cpuBlobCount = (size_t)zarray_size(cpuClusters);
    Blob *cpuBlobs = malloc(sizeof(Blob) * (cpuBlobCount > 0 ? cpuBlobCount : 1));
    for (size_t c = 0; c < cpuBlobCount; c++) {
        zarray_t *cl;
        zarray_get(cpuClusters, (int)c, &cl);
        size_t count = (size_t)zarray_size(cl);
        uint64_t *pts = malloc(count * 8);
        for (size_t j = 0; j < count; j++) {
            struct pt *p;
            zarray_get_volatile(cl, (int)j, &p);
            pts[j] = packPt(p);
        }
        qsort(pts, count, 8, compareU64);
        cpuBlobs[c].pts = pts;
        cpuBlobs[c].count = count;
    }

    qsort(gpuBlobs, gpuBlobCount, sizeof(Blob), compareBlobs);
    qsort(cpuBlobs, cpuBlobCount, sizeof(Blob), compareBlobs);

    if (gpuBlobCount != cpuBlobCount) {
        printf("equivalence: CLUSTER COUNT MISMATCH cpu=%zu gpu=%zu\n", cpuBlobCount, gpuBlobCount);
        return 2;
    }
    size_t mismatches = 0;
    for (size_t i = 0; i < cpuBlobCount; i++) {
        if (cpuBlobs[i].count != gpuBlobs[i].count ||
            memcmp(cpuBlobs[i].pts, gpuBlobs[i].pts, cpuBlobs[i].count * 8) != 0)
            mismatches++;
    }
    if (mismatches == 0)
        printf("equivalence: IDENTICAL cluster content (%zu clusters)\n", cpuBlobCount);
    else
        printf("equivalence: %zu of %zu clusters differ\n", mismatches, cpuBlobCount);

    return mismatches == 0 ? 0 : 2;
}
