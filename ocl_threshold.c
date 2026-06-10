#include "ocl_threshold.h"

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/workerpool.h"

// GPU implementation of the detector frontend: adaptive tile threshold,
// connected components, component sizes, and boundary-pair extraction,
// replicating the CPU implementations' exact semantics (see
// apriltag_quad_thresh.c). Cluster grouping uses a single GPU partition pass
// over the high bits of the component-pair key; final grouping happens on
// the CPU during the cluster build it must perform anyway.
//
// All entry points return NULL when the GPU path is disabled (APRILTAG_OPENCL
// unset) or unavailable, in which case callers run the CPU implementation.
// APRILTAG_OPENCL_PROFILE=1 prints per-kernel GPU timings to stderr.

static const char *sourceThreshold =
    "__kernel void tileMinmax(__global const uchar *im, int s, int tw,\n"
    "                         __global uchar *tileMax, __global uchar *tileMin) {\n"
    "    int tx = get_global_id(0);\n"
    "    int ty = get_global_id(1);\n"
    "    if (tx >= tw) return;\n"
    "    uchar mx = 0, mn = 255;\n"
    "    int base = (ty * 4) * s + tx * 4;\n"
    "    for (int dy = 0; dy < 4; dy++) {\n"
    "        for (int dx = 0; dx < 4; dx++) {\n"
    "            uchar v = im[base + dy * s + dx];\n"
    "            mx = max(mx, v);\n"
    "            mn = min(mn, v);\n"
    "        }\n"
    "    }\n"
    "    tileMax[ty * tw + tx] = mx;\n"
    "    tileMin[ty * tw + tx] = mn;\n"
    "}\n"
    "__kernel void tileBlur(__global const uchar *tileMax, __global const uchar *tileMin,\n"
    "                       int tw, int th,\n"
    "                       __global uchar *blurMax, __global uchar *blurMin) {\n"
    "    int tx = get_global_id(0);\n"
    "    int ty = get_global_id(1);\n"
    "    if (tx >= tw || ty >= th) return;\n"
    "    uchar mx = 0, mn = 255;\n"
    "    for (int dy = -1; dy <= 1; dy++) {\n"
    "        if (ty + dy < 0 || ty + dy >= th) continue;\n"
    "        for (int dx = -1; dx <= 1; dx++) {\n"
    "            if (tx + dx < 0 || tx + dx >= tw) continue;\n"
    "            mx = max(mx, tileMax[(ty + dy) * tw + tx + dx]);\n"
    "            mn = min(mn, tileMin[(ty + dy) * tw + tx + dx]);\n"
    "        }\n"
    "    }\n"
    "    blurMax[ty * tw + tx] = mx;\n"
    "    blurMin[ty * tw + tx] = mn;\n"
    "}\n"
    // classify also seeds the CCL labels so the frontend path can skip a
    // whole initLabels pass over the image.
    "__kernel void classify(__global const uchar *im, int s, int w, int h,\n"
    "                       int tw, int th,\n"
    "                       __global const uchar *blurMax, __global const uchar *blurMin,\n"
    "                       int minWhiteBlackDiff, __global uchar *out, __global uint *labels) {\n"
    "    int x = get_global_id(0);\n"
    "    int y = get_global_id(1);\n"
    "    if (x >= w || y >= h) return;\n"
    "    int interior = (x < tw * 4) && (y < th * 4);\n"
    "    int tx = min(x >> 2, tw - 1);\n"
    "    int ty = min(y >> 2, th - 1);\n"
    "    int mn = blurMin[ty * tw + tx];\n"
    "    int mx = blurMax[ty * tw + tx];\n"
    "    uchar result;\n"
    "    if (interior && (mx - mn < minWhiteBlackDiff)) {\n"
    "        result = 127;\n"
    "    } else {\n"
    "        int thresh = mn + (mx - mn) / 2;\n"
    "        result = (im[y * s + x] > thresh) ? (uchar)255 : (uchar)0;\n"
    "    }\n"
    "    out[y * s + x] = result;\n"
    "    labels[y * w + x] = (result == 127) ? 0xFFFFFFFFu : (uint)(y * w + x);\n"
    "}\n";

static const char *sourceCcl =
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
    "}\n";

static const char *sourceCompress =
    // Path-compress every label and accumulate component sizes with one
    // atomic per horizontal same-root run instead of one per pixel: large
    // uniform regions otherwise serialize millions of atomics on one root.
    "__kernel void compressAndCount(__global const uchar *im, int s, int w, int h,\n"
    "                               __global volatile uint *labels, __global volatile uint *sizes) {\n"
    "    int x = get_global_id(0), y = get_global_id(1);\n"
    "    int lid = get_local_id(0);\n"
    "    __local uint roots[128];\n"
    "    uint root = 0xFFFFFFFFu;\n"
    "    if (x < w && y < h && im[y * s + x] != 127) {\n"
    "        root = findRoot(labels, (uint)(y * w + x));\n"
    "        labels[y * w + x] = root;\n"
    "    }\n"
    "    roots[lid] = root;\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    if (root == 0xFFFFFFFFu) return;\n"
    "    if (lid > 0 && roots[lid - 1] == root) return;\n"
    "    uint run = 1;\n"
    "    uint lsz = (uint)get_local_size(0);\n"
    "    while ((uint)lid + run < lsz && roots[lid + run] == root) run++;\n"
    "    atomic_add(&sizes[root], run);\n"
    "}\n"
    // One coalesced byte per pixel replaces the extract kernel's repeated
    // scattered label+size lookups.
    "__kernel void buildBigMap(__global const uchar *im, int s, int w, int h,\n"
    "                          __global const uint *labels, __global const uint *sizes,\n"
    "                          uint minCluster, __global uchar *bigMap) {\n"
    "    int x = get_global_id(0), y = get_global_id(1);\n"
    "    if (x >= w || y >= h) return;\n"
    "    uint idx = (uint)(y * w + x);\n"
    "    uchar ok = 0;\n"
    "    if (im[y * s + x] != 127)\n"
    "        ok = (sizes[labels[idx]] >= minCluster) ? (uchar)1 : (uchar)0;\n"
    "    bigMap[idx] = ok;\n"
    "}\n";

static const char *sourceExtract =
    "inline int wouldEmit(__global const uchar *im, __global const uchar *bigMap,\n"
    "                     int s, int w, int x, int y, int dx, int dy) {\n"
    "    if (bigMap[y * w + x] == 0) return 0;\n"
    "    uchar v0 = im[y * s + x];\n"
    "    uchar v1 = im[(y + dy) * s + x + dx];\n"
    "    if ((int)v0 + (int)v1 != 255) return 0;\n"
    "    return bigMap[(y + dy) * w + x + dx] != 0;\n"
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
    "                           __global const uint *labels, __global const uchar *bigMap,\n"
    "                           __global volatile uint *counter,\n"
    "                           __global ulong2 *records, uint capacity) {\n"
    "    int x = get_global_id(0), y = get_global_id(1);\n"
    "    if (x < 1 || x >= w - 1 || y < 1 || y >= h - 1) return;\n"
    "    if (bigMap[y * w + x] == 0) return;\n"
    "    if (wouldEmit(im, bigMap, s, w, x, y, 1, 0))\n"
    "        emitPair(im, labels, s, w, x, y, 1, 0, counter, records, capacity);\n"
    "    if (wouldEmit(im, bigMap, s, w, x, y, 0, 1))\n"
    "        emitPair(im, labels, s, w, x, y, 0, 1, counter, records, capacity);\n"
    "    int prevEmitted = (x > 1) && wouldEmit(im, bigMap, s, w, x - 1, y, 1, 1);\n"
    "    if (!prevEmitted && wouldEmit(im, bigMap, s, w, x, y, -1, 1))\n"
    "        emitPair(im, labels, s, w, x, y, -1, 1, counter, records, capacity);\n"
    "    if (wouldEmit(im, bigMap, s, w, x, y, 1, 1))\n"
    "        emitPair(im, labels, s, w, x, y, 1, 1, counter, records, capacity);\n"
    "}\n";

static const char *sourcePartition =
    // Both kernels read the live record count from the device so the host
    // never has to stall mid-chain; launched over the full capacity.
    "__kernel void histKeys(__global const ulong2 *records, __global const uint *counter,\n"
    "                       uint capacity, __global volatile uint *hist) {\n"
    "    __local uint localCount;\n"
    "    if (get_local_id(0) == 0) localCount = min(counter[0], capacity);\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    uint i = get_global_id(0);\n"
    "    if (i >= localCount) return;\n"
    "    atomic_inc(&hist[(uint)((records[i].x >> 39) & 0xFFFFul)]);\n"
    "}\n"
    "__kernel void scatterRecords(__global const ulong2 *records, __global const uint *counter,\n"
    "                             uint capacity, __global volatile uint *offsets,\n"
    "                             __global ulong2 *out) {\n"
    "    __local uint localCount;\n"
    "    if (get_local_id(0) == 0) localCount = min(counter[0], capacity);\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    uint i = get_global_id(0);\n"
    "    if (i >= localCount) return;\n"
    "    ulong2 r = records[i];\n"
    "    out[atomic_inc(&offsets[(uint)((r.x >> 39) & 0xFFFFul)])] = r;\n"
    "}\n";

static const char *sourceScan =
    "__kernel void scanLocal(__global const uint *hist, __global uint *offsets,\n"
    "                        __global uint *blockSums) {\n"
    "    int lid = get_local_id(0);\n"
    "    int gid = get_global_id(0);\n"
    "    __local uint tmp[256];\n"
    "    tmp[lid] = hist[gid];\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    for (int offset = 1; offset < 256; offset <<= 1) {\n"
    "        uint v = (lid >= offset) ? tmp[lid - offset] : 0u;\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "        tmp[lid] += v;\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    }\n"
    "    offsets[gid] = tmp[lid] - hist[gid];\n"
    "    if (lid == 255) blockSums[get_group_id(0)] = tmp[lid];\n"
    "}\n"
    "__kernel void scanBlocks(__global uint *blockSums) {\n"
    "    uint running = 0;\n"
    "    for (int i = 0; i < 256; i++) {\n"
    "        uint v = blockSums[i];\n"
    "        blockSums[i] = running;\n"
    "        running += v;\n"
    "    }\n"
    "}\n"
    "__kernel void addBlockOffsets(__global uint *offsets, __global const uint *blockSums) {\n"
    "    offsets[get_global_id(0)] += blockSums[get_group_id(0)];\n"
    "}\n";

#define OCL_RECORD_CAPACITY (8u * 1024u * 1024u)
#define OCL_BIN_COUNT 65536u

typedef struct {
    uint16_t x, y;
    int16_t gx, gy;
    float slope;
} OclPt;

static pthread_once_t oclInitOnce = PTHREAD_ONCE_INIT;
static pthread_mutex_t oclMutex = PTHREAD_MUTEX_INITIALIZER;
static int oclReady = 0;
static cl_context oclContext;
static cl_command_queue oclQueue;
static cl_kernel oclKernelTileMinmax;
static cl_kernel oclKernelTileBlur;
static cl_kernel oclKernelClassify;
static cl_kernel oclKernelInitLabels;
static cl_kernel oclKernelMergeEdges;
static cl_kernel oclKernelCompressAndCount;
static cl_kernel oclKernelBuildBigMap;
static cl_kernel oclKernelExtractPairs;
static cl_kernel oclKernelHistKeys;
static cl_kernel oclKernelScatterRecords;
static cl_kernel oclKernelScanLocal;
static cl_kernel oclKernelScanBlocks;
static cl_kernel oclKernelAddBlockOffsets;

typedef struct {
    int valid;
    cl_int w, h, s;
    const void *thresholdOutputFor;
    cl_mem bufIm;
    cl_mem bufOut;
    cl_mem bufMaxRaw;
    cl_mem bufMinRaw;
    cl_mem bufMaxBlur;
    cl_mem bufMinBlur;
    cl_mem bufLabels;
    cl_mem bufSizes;
    cl_mem bufBigMap;
    cl_mem bufCounter;
    cl_mem bufRecords;
    cl_mem bufPartitioned;
    cl_mem bufHist;
    cl_mem bufOffsets;
    cl_mem bufBlockSums;
} OclBufferCache;

static OclBufferCache cache;
static uint32_t histHost[OCL_BIN_COUNT];
static uint32_t offsetsHost[OCL_BIN_COUNT];
static uint32_t counterHost;

// APRILTAG_OPENCL_EXACT=1 sorts cluster points into the CPU emitter's exact
// order, making output bit-identical to the CPU path (validation mode). It
// currently costs more CPU than it saves; production runs without it, where
// output is content-equivalent and corners may differ at the 0.02 px level.
// TODO: emit records row-ordered on the GPU to get exactness for free.
static int oclExactOrder;

static cl_event profEventsArr[32];
static const char *profNamesArr[32];
static int profEventCount;
static int profEnabled;

static cl_event *profSlot(const char *name)
{
    if (!profEnabled || profEventCount >= 32)
        return NULL;
    profNamesArr[profEventCount] = name;
    return &profEventsArr[profEventCount++];
}

static void profReset(void)
{
    profEnabled = getenv("APRILTAG_OPENCL_PROFILE") != NULL;
    profEventCount = 0;
}

static void profPrint(void)
{
    if (!profEnabled)
        return;
    cl_ulong first = (cl_ulong)-1, last = 0;
    for (int i = 0; i < profEventCount; i++) {
        cl_ulong t0 = 0, t1 = 0;
        clGetEventProfilingInfo(profEventsArr[i], CL_PROFILING_COMMAND_START, sizeof(t0), &t0, NULL);
        clGetEventProfilingInfo(profEventsArr[i], CL_PROFILING_COMMAND_END, sizeof(t1), &t1, NULL);
        fprintf(stderr, "  %-16s %8.1f us\n", profNamesArr[i], (t1 - t0) / 1000.0);
        if (t0 < first)
            first = t0;
        if (t1 > last)
            last = t1;
        clReleaseEvent(profEventsArr[i]);
    }
    if (profEventCount > 0)
        fprintf(stderr, "  %-16s %8.1f us\n", "gpu span", (last - first) / 1000.0);
    profEventCount = 0;
}

static void oclDebugLog(const char *message)
{
    if (getenv("APRILTAG_OPENCL_DEBUG") != NULL)
        fprintf(stderr, "apriltag opencl: %s\n", message);
}

static void oclInit(void)
{
    cl_platform_id platforms[8];
    cl_uint platformCount = 0;
    if (clGetPlatformIDs(8, platforms, &platformCount) != CL_SUCCESS || platformCount == 0) {
        oclDebugLog("no OpenCL platforms");
        return;
    }

    cl_device_id device = NULL;
    for (cl_uint i = 0; i < platformCount && device == NULL; i++) {
        cl_uint deviceCount = 0;
        if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1, &device, &deviceCount) != CL_SUCCESS)
            device = NULL;
    }
    if (device == NULL) {
        oclDebugLog("no GPU device");
        return;
    }

    cl_int err = CL_SUCCESS;
    oclContext = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS)
        return;
    const cl_queue_properties queueProps[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    oclQueue = clCreateCommandQueueWithProperties(oclContext, device, queueProps, &err);
    if (err != CL_SUCCESS)
        return;

    const char *sources[6] = { sourceThreshold, sourceCcl, sourceCompress, sourceExtract, sourcePartition, sourceScan };
    cl_program program = clCreateProgramWithSource(oclContext, 6, sources, NULL, &err);
    if (err != CL_SUCCESS)
        return;
    err = clBuildProgram(program, 1, &device, "", NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[8192] = { 0 };
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log) - 1, log, NULL);
        oclDebugLog(log);
        clReleaseProgram(program);
        return;
    }

    struct { cl_kernel *handle; const char *name; } kernels[] = {
        { &oclKernelTileMinmax, "tileMinmax" },
        { &oclKernelTileBlur, "tileBlur" },
        { &oclKernelClassify, "classify" },
        { &oclKernelInitLabels, "initLabels" },
        { &oclKernelMergeEdges, "mergeEdges" },
        { &oclKernelCompressAndCount, "compressAndCount" },
        { &oclKernelBuildBigMap, "buildBigMap" },
        { &oclKernelExtractPairs, "extractPairs" },
        { &oclKernelHistKeys, "histKeys" },
        { &oclKernelScatterRecords, "scatterRecords" },
        { &oclKernelScanLocal, "scanLocal" },
        { &oclKernelScanBlocks, "scanBlocks" },
        { &oclKernelAddBlockOffsets, "addBlockOffsets" },
    };
    int failed = 0;
    for (size_t i = 0; i < sizeof(kernels) / sizeof(kernels[0]); i++) {
        *kernels[i].handle = clCreateKernel(program, kernels[i].name, &err);
        if (err != CL_SUCCESS)
            failed = 1;
    }
    clReleaseProgram(program);
    if (failed)
        return;

    oclReady = 1;
}

static void releaseBuffer(cl_mem buffer)
{
    if (buffer != NULL)
        clReleaseMemObject(buffer);
}

static void releaseCache(void)
{
    releaseBuffer(cache.bufIm);
    releaseBuffer(cache.bufOut);
    releaseBuffer(cache.bufMaxRaw);
    releaseBuffer(cache.bufMinRaw);
    releaseBuffer(cache.bufMaxBlur);
    releaseBuffer(cache.bufMinBlur);
    releaseBuffer(cache.bufLabels);
    releaseBuffer(cache.bufSizes);
    releaseBuffer(cache.bufBigMap);
    releaseBuffer(cache.bufCounter);
    releaseBuffer(cache.bufRecords);
    releaseBuffer(cache.bufPartitioned);
    releaseBuffer(cache.bufHist);
    releaseBuffer(cache.bufOffsets);
    releaseBuffer(cache.bufBlockSums);
    memset(&cache, 0, sizeof(cache));
}

static cl_mem createOrFail(cl_mem_flags flags, size_t bytes, void *host, int *failed)
{
    cl_int err = CL_SUCCESS;
    cl_mem buffer = clCreateBuffer(oclContext, flags, bytes, host, &err);
    if (err != CL_SUCCESS)
        *failed = 1;
    return buffer;
}

static int ensureCache(cl_int w, cl_int h, cl_int s, cl_int tw, cl_int th)
{
    if (cache.valid && cache.w == w && cache.h == h && cache.s == s)
        return 1;
    releaseCache();

    const size_t imageBytes = (size_t)s * (size_t)h;
    const size_t tileBytes = (size_t)tw * (size_t)th;
    const size_t pixelCount = (size_t)w * (size_t)h;
    int failed = 0;
    cache.bufIm = createOrFail(CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, imageBytes, NULL, &failed);
    cache.bufOut = createOrFail(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, imageBytes, NULL, &failed);
    cache.bufMaxRaw = createOrFail(CL_MEM_READ_WRITE, tileBytes, NULL, &failed);
    cache.bufMinRaw = createOrFail(CL_MEM_READ_WRITE, tileBytes, NULL, &failed);
    cache.bufMaxBlur = createOrFail(CL_MEM_READ_WRITE, tileBytes, NULL, &failed);
    cache.bufMinBlur = createOrFail(CL_MEM_READ_WRITE, tileBytes, NULL, &failed);
    cache.bufLabels = createOrFail(CL_MEM_READ_WRITE, pixelCount * 4, NULL, &failed);
    cache.bufSizes = createOrFail(CL_MEM_READ_WRITE, pixelCount * 4, NULL, &failed);
    cache.bufBigMap = createOrFail(CL_MEM_READ_WRITE, pixelCount, NULL, &failed);
    cache.bufCounter = createOrFail(CL_MEM_READ_WRITE, 4, NULL, &failed);
    cache.bufRecords = createOrFail(CL_MEM_READ_WRITE, (size_t)OCL_RECORD_CAPACITY * 16, NULL, &failed);
    cache.bufPartitioned = createOrFail(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, (size_t)OCL_RECORD_CAPACITY * 16, NULL, &failed);
    cache.bufHist = createOrFail(CL_MEM_READ_WRITE, OCL_BIN_COUNT * 4, NULL, &failed);
    cache.bufOffsets = createOrFail(CL_MEM_READ_WRITE, OCL_BIN_COUNT * 4, NULL, &failed);
    cache.bufBlockSums = createOrFail(CL_MEM_READ_WRITE, 256 * 4, NULL, &failed);
    if (failed) {
        releaseCache();
        return 0;
    }

    // Stride padding bytes are never written by the threshold kernels; zero
    // the output buffer once so padding matches the CPU path's calloc'd image.
    const cl_uchar zero = 0;
    if (clEnqueueFillBuffer(oclQueue, cache.bufOut, &zero, 1, 0, imageBytes, 0, NULL, NULL) != CL_SUCCESS) {
        releaseCache();
        return 0;
    }

    cache.valid = 1;
    cache.w = w;
    cache.h = h;
    cache.s = s;
    return 1;
}

static size_t roundUp(size_t value, size_t multiple)
{
    return ((value + multiple - 1) / multiple) * multiple;
}

static cl_int setThresholdArgs(cl_mem input, cl_int w, cl_int h, cl_int s, cl_int tw, cl_int th, cl_int minWhiteBlackDiff)
{
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(oclKernelTileMinmax, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(oclKernelTileMinmax, 1, sizeof(cl_int), &s);
    err |= clSetKernelArg(oclKernelTileMinmax, 2, sizeof(cl_int), &tw);
    err |= clSetKernelArg(oclKernelTileMinmax, 3, sizeof(cl_mem), &cache.bufMaxRaw);
    err |= clSetKernelArg(oclKernelTileMinmax, 4, sizeof(cl_mem), &cache.bufMinRaw);
    err |= clSetKernelArg(oclKernelTileBlur, 0, sizeof(cl_mem), &cache.bufMaxRaw);
    err |= clSetKernelArg(oclKernelTileBlur, 1, sizeof(cl_mem), &cache.bufMinRaw);
    err |= clSetKernelArg(oclKernelTileBlur, 2, sizeof(cl_int), &tw);
    err |= clSetKernelArg(oclKernelTileBlur, 3, sizeof(cl_int), &th);
    err |= clSetKernelArg(oclKernelTileBlur, 4, sizeof(cl_mem), &cache.bufMaxBlur);
    err |= clSetKernelArg(oclKernelTileBlur, 5, sizeof(cl_mem), &cache.bufMinBlur);
    err |= clSetKernelArg(oclKernelClassify, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(oclKernelClassify, 1, sizeof(cl_int), &s);
    err |= clSetKernelArg(oclKernelClassify, 2, sizeof(cl_int), &w);
    err |= clSetKernelArg(oclKernelClassify, 3, sizeof(cl_int), &h);
    err |= clSetKernelArg(oclKernelClassify, 4, sizeof(cl_int), &tw);
    err |= clSetKernelArg(oclKernelClassify, 5, sizeof(cl_int), &th);
    err |= clSetKernelArg(oclKernelClassify, 6, sizeof(cl_mem), &cache.bufMaxBlur);
    err |= clSetKernelArg(oclKernelClassify, 7, sizeof(cl_mem), &cache.bufMinBlur);
    err |= clSetKernelArg(oclKernelClassify, 8, sizeof(cl_int), &minWhiteBlackDiff);
    err |= clSetKernelArg(oclKernelClassify, 9, sizeof(cl_mem), &cache.bufOut);
    err |= clSetKernelArg(oclKernelClassify, 10, sizeof(cl_mem), &cache.bufLabels);
    return err;
}

image_u8_t *oclThreshold(apriltag_detector_t *td, image_u8_t *im)
{
    if (getenv("APRILTAG_OPENCL") == NULL)
        return NULL;
    if (td->qtp.deglitch != 0)
        return NULL;

    pthread_once(&oclInitOnce, oclInit);
    if (oclReady == 0)
        return NULL;

    const int tilesz = 4;
    const cl_int w = im->width, h = im->height, s = im->stride;
    const cl_int tw = w / tilesz, th = h / tilesz;
    if (tw < 1 || th < 1)
        return NULL;
    const size_t imageBytes = (size_t)s * (size_t)h;

    image_u8_t *threshim = NULL;
    int ok = 0;

    pthread_mutex_lock(&oclMutex);
    if (!ensureCache(w, h, s, tw, th))
        goto done;

    cl_int err = CL_SUCCESS;
    void *stagingIn = clEnqueueMapBuffer(oclQueue, cache.bufIm, CL_TRUE,
                                         CL_MAP_WRITE_INVALIDATE_REGION, 0, imageBytes, 0, NULL, NULL, &err);
    if (err != CL_SUCCESS)
        goto done;
    memcpy(stagingIn, im->buf, imageBytes);
    err = clEnqueueUnmapMemObject(oclQueue, cache.bufIm, stagingIn, 0, NULL, NULL);
    if (err != CL_SUCCESS)
        goto done;

    err = setThresholdArgs(cache.bufIm, w, h, s, tw, th, td->qtp.min_white_black_diff);
    if (err != CL_SUCCESS)
        goto done;

    const size_t tileGlobal[2] = { (size_t)tw, (size_t)th };
    const size_t pixelGlobal[2] = { roundUp((size_t)w, 16), roundUp((size_t)h, 16) };
    err |= clEnqueueNDRangeKernel(oclQueue, oclKernelTileMinmax, 2, NULL, tileGlobal, NULL, 0, NULL, NULL);
    err |= clEnqueueNDRangeKernel(oclQueue, oclKernelTileBlur, 2, NULL, tileGlobal, NULL, 0, NULL, NULL);
    err |= clEnqueueNDRangeKernel(oclQueue, oclKernelClassify, 2, NULL, pixelGlobal, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS)
        goto done;

    void *stagingOut = clEnqueueMapBuffer(oclQueue, cache.bufOut, CL_TRUE, CL_MAP_READ, 0, imageBytes, 0, NULL, NULL, &err);
    if (err != CL_SUCCESS)
        goto done;
    threshim = image_u8_create_alignment(w, h, s);
    if (threshim->stride == s) {
        memcpy(threshim->buf, stagingOut, imageBytes);
        cache.thresholdOutputFor = threshim->buf;
        ok = 1;
    }
    clEnqueueUnmapMemObject(oclQueue, cache.bufOut, stagingOut, 0, NULL, NULL);

done:
    pthread_mutex_unlock(&oclMutex);
    if (ok == 0) {
        oclDebugLog("GPU threshold failed, falling back to CPU");
        if (threshim != NULL)
            image_u8_destroy(threshim);
        return NULL;
    }
    return threshim;
}

static void appendPt(zarray_t *cluster, uint64_t packed)
{
    if (cluster->size == cluster->alloc)
        zarray_ensure_capacity(cluster, cluster->size == 0 ? 16 : cluster->size * 2);
    OclPt *dst = (OclPt *)(cluster->data + (size_t)cluster->size * cluster->el_sz);
    dst->x = (uint16_t)(packed >> 48);
    dst->y = (uint16_t)(packed >> 32);
    dst->gx = (int16_t)(uint16_t)(packed >> 16);
    dst->gy = (int16_t)(uint16_t)packed;
    dst->slope = 0.0f;
    cluster->size++;
}

typedef struct {
    uint64_t key;
    zarray_t *cluster;
} PartnerSlot;

// Canonical within-cluster ordering: reconstruct the CPU emitter's raster
// order (y, then x, then connectivity-check index) from the point fields.
// This makes GPU output deterministic run-to-run regardless of atomic emit
// order, and aligns marginal quad fits with the CPU implementation.
static uint64_t ptOrderKey(const OclPt *p)
{
    int conn;
    if (p->gy == 0)
        conn = 0;
    else if (p->gx == 0)
        conn = 1;
    else if (p->gx == -p->gy)
        conn = 2;
    else
        conn = 3;
    int dx = (conn == 0 || conn == 3) ? 1 : (conn == 2 ? -1 : 0);
    int dy = (conn == 0) ? 0 : 1;
    uint64_t y = ((uint64_t)p->y - (uint64_t)dy) / 2;
    uint64_t x = ((uint64_t)(p->x - dx)) / 2;
    return (y << 18) | (x << 2) | (uint64_t)conn;
}

static int comparePtOrder(const void *a, const void *b)
{
    uint64_t ka = ptOrderKey((const OclPt *)a);
    uint64_t kb = ptOrderKey((const OclPt *)b);
    return (ka > kb) - (ka < kb);
}

typedef struct {
    const uint64_t *records;
    uint32_t binStart, binEnd;
    zarray_t *clusters;
} BuildTask;

static void doBuildTask(void *p)
{
    BuildTask *task = (BuildTask *)p;
    const uint64_t *records = task->records;
    int partnersCap = 256;
    PartnerSlot *partners = malloc(sizeof(PartnerSlot) * partnersCap);

    for (uint32_t bin = task->binStart; bin < task->binEnd; bin++) {
        uint32_t n = histHost[bin];
        if (n == 0)
            continue;
        uint32_t base = offsetsHost[bin];
        int partnerCount = 0;
        for (uint32_t i = 0; i < n; i++) {
            uint64_t key = records[2 * (base + i)];
            uint64_t payload = records[2 * (base + i) + 1];
            int slot = -1;
            for (int j = partnerCount - 1; j >= 0; j--) {
                if (partners[j].key == key) {
                    slot = j;
                    break;
                }
            }
            if (slot < 0) {
                if (partnerCount == partnersCap) {
                    partnersCap *= 2;
                    partners = realloc(partners, sizeof(PartnerSlot) * partnersCap);
                }
                partners[partnerCount].key = key;
                partners[partnerCount].cluster = zarray_create(sizeof(OclPt));
                slot = partnerCount++;
            }
            appendPt(partners[slot].cluster, payload);
        }
        for (int j = 0; j < partnerCount; j++) {
            zarray_t *cluster = partners[j].cluster;
            if (oclExactOrder)
                qsort(cluster->data, cluster->size, cluster->el_sz, comparePtOrder);
            zarray_add(task->clusters, &cluster);
        }
    }
    free(partners);
}

static zarray_t *buildClusters(apriltag_detector_t *td, const uint64_t *records, uint32_t recordCount)
{
    int taskCount = (td->wp != NULL && td->nthreads > 1) ? td->nthreads : 1;
    if (taskCount > 16)
        taskCount = 16;
    BuildTask tasks[16];

    // Split bins into ranges balanced by record count so workers finish together.
    uint32_t targetPerTask = recordCount / (uint32_t)taskCount + 1;
    uint32_t bin = 0;
    for (int t = 0; t < taskCount; t++) {
        tasks[t].records = records;
        tasks[t].binStart = bin;
        tasks[t].clusters = zarray_create(sizeof(zarray_t *));
        uint32_t taken = 0;
        while (bin < OCL_BIN_COUNT && (taken < targetPerTask || t == taskCount - 1)) {
            taken += histHost[bin];
            bin++;
        }
        tasks[t].binEnd = bin;
    }
    tasks[taskCount - 1].binEnd = OCL_BIN_COUNT;

    if (taskCount == 1) {
        doBuildTask(&tasks[0]);
    } else {
        for (int t = 0; t < taskCount; t++)
            workerpool_add_task(td->wp, doBuildTask, &tasks[t]);
        workerpool_run(td->wp);
    }

    zarray_t *clusters = zarray_create(sizeof(zarray_t *));
    for (int t = 0; t < taskCount; t++) {
        for (int i = 0; i < zarray_size(tasks[t].clusters); i++) {
            zarray_t *cluster;
            zarray_get(tasks[t].clusters, i, &cluster);
            zarray_add(clusters, &cluster);
        }
        zarray_destroy(tasks[t].clusters);
    }
    return clusters;
}

// Runs CCL + sizes + extraction + partition over the threshold image already
// in inputBuffer, then builds the cluster arrays on the CPU. Caller holds
// oclMutex and has a valid cache. labelsReady indicates the classify kernel
// already seeded the labels buffer.
static zarray_t *runClusterChain(apriltag_detector_t *td, cl_mem inputBuffer, cl_int cw, cl_int ch, cl_int cs, int labelsReady)
{
    oclExactOrder = getenv("APRILTAG_OPENCL_EXACT") != NULL;
    const cl_uint minCluster = (cl_uint)td->qtp.min_cluster_pixels;
    const cl_uint capacity = OCL_RECORD_CAPACITY;
    zarray_t *clusters = NULL;
    cl_int err = CL_SUCCESS;
    const cl_uint zero = 0;
    const size_t pixelCount = (size_t)cw * (size_t)ch;

    err |= clEnqueueFillBuffer(oclQueue, cache.bufSizes, &zero, 4, 0, pixelCount * 4, 0, NULL, profSlot("fillSizes"));
    err |= clEnqueueFillBuffer(oclQueue, cache.bufCounter, &zero, 4, 0, 4, 0, NULL, NULL);
    err |= clEnqueueFillBuffer(oclQueue, cache.bufHist, &zero, 4, 0, OCL_BIN_COUNT * 4, 0, NULL, NULL);
    if (err != CL_SUCCESS)
        goto done;

    cl_kernel layoutKernels[4] = { oclKernelInitLabels, oclKernelMergeEdges, oclKernelCompressAndCount, oclKernelBuildBigMap };
    for (int k = 0; k < 4; k++) {
        err |= clSetKernelArg(layoutKernels[k], 0, sizeof(cl_mem), &inputBuffer);
        err |= clSetKernelArg(layoutKernels[k], 1, sizeof(cl_int), &cs);
        err |= clSetKernelArg(layoutKernels[k], 2, sizeof(cl_int), &cw);
        err |= clSetKernelArg(layoutKernels[k], 3, sizeof(cl_int), &ch);
        err |= clSetKernelArg(layoutKernels[k], 4, sizeof(cl_mem), &cache.bufLabels);
    }
    err |= clSetKernelArg(oclKernelCompressAndCount, 5, sizeof(cl_mem), &cache.bufSizes);
    err |= clSetKernelArg(oclKernelBuildBigMap, 5, sizeof(cl_mem), &cache.bufSizes);
    err |= clSetKernelArg(oclKernelBuildBigMap, 6, sizeof(cl_uint), &minCluster);
    err |= clSetKernelArg(oclKernelBuildBigMap, 7, sizeof(cl_mem), &cache.bufBigMap);
    err |= clSetKernelArg(oclKernelExtractPairs, 0, sizeof(cl_mem), &inputBuffer);
    err |= clSetKernelArg(oclKernelExtractPairs, 1, sizeof(cl_int), &cs);
    err |= clSetKernelArg(oclKernelExtractPairs, 2, sizeof(cl_int), &cw);
    err |= clSetKernelArg(oclKernelExtractPairs, 3, sizeof(cl_int), &ch);
    err |= clSetKernelArg(oclKernelExtractPairs, 4, sizeof(cl_mem), &cache.bufLabels);
    err |= clSetKernelArg(oclKernelExtractPairs, 5, sizeof(cl_mem), &cache.bufBigMap);
    err |= clSetKernelArg(oclKernelExtractPairs, 6, sizeof(cl_mem), &cache.bufCounter);
    err |= clSetKernelArg(oclKernelExtractPairs, 7, sizeof(cl_mem), &cache.bufRecords);
    err |= clSetKernelArg(oclKernelExtractPairs, 8, sizeof(cl_uint), &capacity);
    if (err != CL_SUCCESS)
        goto done;

    const size_t global[2] = { roundUp((size_t)cw, 16), roundUp((size_t)ch, 16) };
    const size_t countGlobal[2] = { roundUp((size_t)cw, 128), (size_t)ch };
    const size_t countLocal[2] = { 128, 1 };
    if (!labelsReady) {
        err |= clEnqueueNDRangeKernel(oclQueue, oclKernelInitLabels, 2, NULL, global, NULL, 0, NULL, profSlot("initLabels"));
    }
    err |= clEnqueueNDRangeKernel(oclQueue, oclKernelMergeEdges, 2, NULL, global, NULL, 0, NULL, profSlot("mergeEdges"));
    err |= clEnqueueNDRangeKernel(oclQueue, oclKernelCompressAndCount, 2, NULL, countGlobal, countLocal, 0, NULL, profSlot("compressCount"));
    err |= clEnqueueNDRangeKernel(oclQueue, oclKernelBuildBigMap, 2, NULL, global, NULL, 0, NULL, profSlot("buildBigMap"));
    err |= clEnqueueNDRangeKernel(oclQueue, oclKernelExtractPairs, 2, NULL, global, NULL, 0, NULL, profSlot("extractPairs"));
    if (err != CL_SUCCESS)
        goto done;

    err |= clSetKernelArg(oclKernelHistKeys, 0, sizeof(cl_mem), &cache.bufRecords);
    err |= clSetKernelArg(oclKernelHistKeys, 1, sizeof(cl_mem), &cache.bufCounter);
    err |= clSetKernelArg(oclKernelHistKeys, 2, sizeof(cl_uint), &capacity);
    err |= clSetKernelArg(oclKernelHistKeys, 3, sizeof(cl_mem), &cache.bufHist);
    err |= clSetKernelArg(oclKernelScanLocal, 0, sizeof(cl_mem), &cache.bufHist);
    err |= clSetKernelArg(oclKernelScanLocal, 1, sizeof(cl_mem), &cache.bufOffsets);
    err |= clSetKernelArg(oclKernelScanLocal, 2, sizeof(cl_mem), &cache.bufBlockSums);
    err |= clSetKernelArg(oclKernelScanBlocks, 0, sizeof(cl_mem), &cache.bufBlockSums);
    err |= clSetKernelArg(oclKernelAddBlockOffsets, 0, sizeof(cl_mem), &cache.bufOffsets);
    err |= clSetKernelArg(oclKernelAddBlockOffsets, 1, sizeof(cl_mem), &cache.bufBlockSums);
    err |= clSetKernelArg(oclKernelScatterRecords, 0, sizeof(cl_mem), &cache.bufRecords);
    err |= clSetKernelArg(oclKernelScatterRecords, 1, sizeof(cl_mem), &cache.bufCounter);
    err |= clSetKernelArg(oclKernelScatterRecords, 2, sizeof(cl_uint), &capacity);
    err |= clSetKernelArg(oclKernelScatterRecords, 3, sizeof(cl_mem), &cache.bufOffsets);
    err |= clSetKernelArg(oclKernelScatterRecords, 4, sizeof(cl_mem), &cache.bufPartitioned);
    if (err != CL_SUCCESS)
        goto done;

    const size_t capacityGlobal[1] = { (size_t)capacity };
    const size_t scanGlobal[1] = { OCL_BIN_COUNT };
    const size_t scanLocalSize[1] = { 256 };
    const size_t singleItem[1] = { 1 };
    err |= clEnqueueNDRangeKernel(oclQueue, oclKernelHistKeys, 1, NULL, capacityGlobal, NULL, 0, NULL, profSlot("histKeys"));
    err |= clEnqueueNDRangeKernel(oclQueue, oclKernelScanLocal, 1, NULL, scanGlobal, scanLocalSize, 0, NULL, profSlot("scanLocal"));
    err |= clEnqueueNDRangeKernel(oclQueue, oclKernelScanBlocks, 1, NULL, singleItem, NULL, 0, NULL, profSlot("scanBlocks"));
    err |= clEnqueueNDRangeKernel(oclQueue, oclKernelAddBlockOffsets, 1, NULL, scanGlobal, scanLocalSize, 0, NULL, profSlot("addBlockOffs"));
    // Read pre-scatter offsets, counts, and the record counter for the host
    // build walk; the in-order queue places these before the scatter mutates
    // the offsets, and none of them stall the host.
    cl_event counterEvent = NULL;
    err |= clEnqueueReadBuffer(oclQueue, cache.bufCounter, CL_FALSE, 0, 4, &counterHost, 0, NULL, &counterEvent);
    err |= clEnqueueReadBuffer(oclQueue, cache.bufOffsets, CL_FALSE, 0, OCL_BIN_COUNT * 4, offsetsHost, 0, NULL, profSlot("readOffsets"));
    err |= clEnqueueReadBuffer(oclQueue, cache.bufHist, CL_FALSE, 0, OCL_BIN_COUNT * 4, histHost, 0, NULL, profSlot("readHist"));
    err |= clEnqueueNDRangeKernel(oclQueue, oclKernelScatterRecords, 1, NULL, capacityGlobal, NULL, 0, NULL, profSlot("scatter"));
    if (err != CL_SUCCESS) {
        if (counterEvent != NULL)
            clReleaseEvent(counterEvent);
        goto done;
    }

    // The counter read completes mid-chain; waiting on it costs nothing
    // extra (the map below blocks on the whole chain anyway) and lets us map
    // only the live records instead of the full capacity buffer.
    clWaitForEvents(1, &counterEvent);
    clReleaseEvent(counterEvent);
    if (counterHost > capacity) {
        oclDebugLog("record capacity exceeded");
        goto done;
    }
    if (counterHost == 0) {
        clusters = zarray_create(sizeof(zarray_t *));
        goto done;
    }

    void *mapped = clEnqueueMapBuffer(oclQueue, cache.bufPartitioned, CL_TRUE, CL_MAP_READ, 0,
                                      (size_t)counterHost * 16, 0, NULL, profSlot("mapRecords"), &err);
    if (err != CL_SUCCESS)
        goto done;
    clusters = buildClusters(td, (const uint64_t *)mapped, counterHost);
    clEnqueueUnmapMemObject(oclQueue, cache.bufPartitioned, mapped, 0, NULL, NULL);

done:
    return clusters;
}

zarray_t *oclClusters(apriltag_detector_t *td, image_u8_t *threshim, int w, int h, int ts)
{
    if (getenv("APRILTAG_OPENCL") == NULL)
        return NULL;
    if (td->debug != 0)
        return NULL;

    pthread_once(&oclInitOnce, oclInit);
    if (oclReady == 0)
        return NULL;
    if ((int64_t)w * (int64_t)h >= ((int64_t)1 << 23))
        return NULL;
    if (w / 4 < 1 || h / 4 < 1)
        return NULL;

    const size_t imageBytes = (size_t)ts * (size_t)h;
    zarray_t *clusters = NULL;

    pthread_mutex_lock(&oclMutex);
    profReset();
    if (!ensureCache(w, h, ts, w / 4, h / 4))
        goto done;

    // When the GPU threshold just produced this exact image, its device copy
    // is still in bufOut — consume the tag and skip the re-upload.
    {
        cl_mem inputBuffer = cache.bufIm;
        int labelsReady = 0;
        if (cache.thresholdOutputFor == (const void *)threshim->buf) {
            inputBuffer = cache.bufOut;
            labelsReady = 1;
        } else {
            cl_int err = CL_SUCCESS;
            void *stagingIn = clEnqueueMapBuffer(oclQueue, cache.bufIm, CL_TRUE,
                                                 CL_MAP_WRITE_INVALIDATE_REGION, 0, imageBytes, 0, NULL, NULL, &err);
            if (err != CL_SUCCESS)
                goto done;
            memcpy(stagingIn, threshim->buf, imageBytes);
            if (clEnqueueUnmapMemObject(oclQueue, cache.bufIm, stagingIn, 0, NULL, NULL) != CL_SUCCESS)
                goto done;
        }
        cache.thresholdOutputFor = NULL;
        clusters = runClusterChain(td, inputBuffer, w, h, ts, labelsReady);
    }

done:
    profPrint();
    pthread_mutex_unlock(&oclMutex);
    if (clusters == NULL)
        oclDebugLog("GPU clusters failed, falling back to CPU");
    return clusters;
}

zarray_t *oclFrontend(apriltag_detector_t *td, image_u8_t *im)
{
    if (getenv("APRILTAG_OPENCL") == NULL)
        return NULL;
    if (td->debug != 0)
        return NULL;
    if (td->qtp.deglitch != 0)
        return NULL;

    pthread_once(&oclInitOnce, oclInit);
    if (oclReady == 0)
        return NULL;

    const int tilesz = 4;
    const cl_int w = im->width, h = im->height, s = im->stride;
    const cl_int tw = w / tilesz, th = h / tilesz;
    if (tw < 1 || th < 1)
        return NULL;
    if ((int64_t)w * (int64_t)h >= ((int64_t)1 << 23))
        return NULL;
    const size_t imageBytes = (size_t)s * (size_t)h;

    zarray_t *clusters = NULL;
    pthread_mutex_lock(&oclMutex);
    profReset();
    if (!ensureCache(w, h, s, tw, th))
        goto done;
    cache.thresholdOutputFor = NULL;

    cl_int err = CL_SUCCESS;
    void *stagingIn = clEnqueueMapBuffer(oclQueue, cache.bufIm, CL_TRUE,
                                         CL_MAP_WRITE_INVALIDATE_REGION, 0, imageBytes, 0, NULL, NULL, &err);
    if (err != CL_SUCCESS)
        goto done;
    memcpy(stagingIn, im->buf, imageBytes);
    err = clEnqueueUnmapMemObject(oclQueue, cache.bufIm, stagingIn, 0, NULL, NULL);
    if (err != CL_SUCCESS)
        goto done;

    err = setThresholdArgs(cache.bufIm, w, h, s, tw, th, td->qtp.min_white_black_diff);
    if (err != CL_SUCCESS)
        goto done;

    const size_t tileGlobal[2] = { (size_t)tw, (size_t)th };
    const size_t pixelGlobal[2] = { roundUp((size_t)w, 16), roundUp((size_t)h, 16) };
    err |= clEnqueueNDRangeKernel(oclQueue, oclKernelTileMinmax, 2, NULL, tileGlobal, NULL, 0, NULL, profSlot("tileMinmax"));
    err |= clEnqueueNDRangeKernel(oclQueue, oclKernelTileBlur, 2, NULL, tileGlobal, NULL, 0, NULL, profSlot("tileBlur"));
    err |= clEnqueueNDRangeKernel(oclQueue, oclKernelClassify, 2, NULL, pixelGlobal, NULL, 0, NULL, profSlot("classify"));
    if (err != CL_SUCCESS)
        goto done;

    clusters = runClusterChain(td, cache.bufOut, w, h, s, 1);

done:
    profPrint();
    pthread_mutex_unlock(&oclMutex);
    if (clusters == NULL)
        oclDebugLog("GPU frontend failed, falling back to CPU");
    return clusters;
}
