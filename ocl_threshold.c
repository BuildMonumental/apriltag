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
// apriltag_quad_thresh.c). Boundary records are emitted in the CPU
// emitter's raster order (per 256-pixel row segment: count, scan, then
// sequential emit), so the CPU-side grouping walk reproduces the CPU
// path's cluster content and within-cluster point order exactly — output
// is bit-identical to the CPU implementation.
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
    // One coalesced byte per pixel replaces the extract logic's repeated
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
    // Mask bits follow the CPU's DO_CONN emit order: (1,0), (0,1), (-1,1), (1,1).
    "inline int emitMask(__global const uchar *im, __global const uchar *bigMap,\n"
    "                    int s, int w, int x, int y) {\n"
    "    if (bigMap[y * w + x] == 0) return 0;\n"
    "    int mask = 0;\n"
    "    if (wouldEmit(im, bigMap, s, w, x, y, 1, 0)) mask |= 1;\n"
    "    if (wouldEmit(im, bigMap, s, w, x, y, 0, 1)) mask |= 2;\n"
    "    int prevEmitted = (x > 1) && wouldEmit(im, bigMap, s, w, x - 1, y, 1, 1);\n"
    "    if (!prevEmitted && wouldEmit(im, bigMap, s, w, x, y, -1, 1)) mask |= 4;\n"
    "    if (wouldEmit(im, bigMap, s, w, x, y, 1, 1)) mask |= 8;\n"
    "    return mask;\n"
    "}\n"
    "inline ulong2 makeRecord(__global const uchar *im, __global const uint *labels,\n"
    "                         int s, int w, int x, int y, int dx, int dy) {\n"
    "    uchar v0 = im[y * s + x];\n"
    "    uchar v1 = im[(y + dy) * s + x + dx];\n"
    "    uint rep0 = labels[y * w + x];\n"
    "    uint rep1 = labels[(y + dy) * w + x + dx];\n"
    // Roots are < 2^23 (enforced by the w*h guard), so the pair packs into
    // 46 contiguous bits — six 8-bit radix passes cover the whole key.
    "    ulong key = (rep0 < rep1) ? (((ulong)rep1 << 23) | rep0) : (((ulong)rep0 << 23) | rep1);\n"
    "    int grad = (int)v1 - (int)v0;\n"
    "    ushort px = (ushort)(2 * x + dx), py = (ushort)(2 * y + dy);\n"
    "    ushort pgx = (ushort)(short)(dx * grad), pgy = (ushort)(short)(dy * grad);\n"
    "    ulong packed = ((ulong)px << 48) | ((ulong)py << 32) | ((ulong)pgx << 16) | (ulong)pgy;\n"
    "    return (ulong2)(key, packed);\n"
    "}\n";

static const char *sourceEmit =
    // Stream compaction: one 256-thread workgroup per row segment, one pixel
    // per thread (coalesced), local prefix scan assigns each thread's record
    // slots. Per-pixel records stay in DO_CONN order and threads ascend in x,
    // so global record order remains the CPU emitter's raster order. The
    // count pass stores each pixel's emit mask so the emit pass reads one
    // byte instead of re-evaluating the neighbour conditions.
    "__kernel void countSegments(__global const uchar *im, __global const uchar *bigMap,\n"
    "                            int s, int w, int h, int segsPerRow, int nSegs,\n"
    "                            __global uint *segCounts, __global uchar *masks) {\n"
    "    int seg = get_group_id(0);\n"
    "    int lid = get_local_id(0);\n"
    "    int y = seg / segsPerRow;\n"
    "    int x = (seg % segsPerRow) * 256 + lid;\n"
    "    int mask = 0;\n"
    "    if (y >= 1 && y < h - 1 && x >= 1 && x < w - 1)\n"
    "        mask = emitMask(im, bigMap, s, w, x, y);\n"
    "    if (x < w && y < h)\n"
    "        masks[y * w + x] = (uchar)mask;\n"
    "    __local uint counts[256];\n"
    "    counts[lid] = (uint)popcount(mask);\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    for (int off = 128; off > 0; off >>= 1) {\n"
    "        if (lid < off) counts[lid] += counts[lid + off];\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    }\n"
    "    if (lid == 0 && seg < nSegs) segCounts[seg] = counts[0];\n"
    "}\n"
    "__kernel void emitSegments(__global const uchar *im, __global const uchar *masks,\n"
    "                           __global const uint *labels, int s, int w, int h,\n"
    "                           int segsPerRow, int nSegs, __global const uint *segOffsets,\n"
    "                           uint capacity, __global ulong2 *records) {\n"
    "    int seg = get_group_id(0);\n"
    "    int lid = get_local_id(0);\n"
    "    int y = seg / segsPerRow;\n"
    "    int x = (seg % segsPerRow) * 256 + lid;\n"
    "    int mask = 0;\n"
    "    if (y >= 1 && y < h - 1 && x >= 1 && x < w - 1)\n"
    "        mask = masks[y * w + x];\n"
    "    uint mine = (uint)popcount(mask);\n"
    "    __local uint scanBuf[256];\n"
    "    scanBuf[lid] = mine;\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    for (int off = 1; off < 256; off <<= 1) {\n"
    "        uint v = (lid >= off) ? scanBuf[lid - off] : 0u;\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "        scanBuf[lid] += v;\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    }\n"
    "    if (mask == 0 || seg >= nSegs) return;\n"
    "    uint slot = segOffsets[seg] + scanBuf[lid] - mine;\n"
    "    if (mask & 1) { if (slot < capacity) records[slot] = makeRecord(im, labels, s, w, x, y, 1, 0); slot++; }\n"
    "    if (mask & 2) { if (slot < capacity) records[slot] = makeRecord(im, labels, s, w, x, y, 0, 1); slot++; }\n"
    "    if (mask & 4) { if (slot < capacity) records[slot] = makeRecord(im, labels, s, w, x, y, -1, 1); slot++; }\n"
    "    if (mask & 8) { if (slot < capacity) records[slot] = makeRecord(im, labels, s, w, x, y, 1, 1); slot++; }\n"
    "}\n";

static const char *sourceSort =
    // Stable LSD radix sort over the compacted 46-bit cluster key, 8-bit
    // digits, 1024-record blocks (32 threads x 32 records, thread-blocked so
    // within-thread order is sequential). Stability preserves the raster
    // point order the emitter established. Per-(digit, block) offsets are
    // scanned on the host between passes.
    "__kernel void radixHist(__global const ulong2 *records, uint count, uint shift,\n"
    "                        uint numBlocks, __global uint *hist) {\n"
    "    int block = get_group_id(0), lid = get_local_id(0);\n"
    "    __local uint h[256];\n"
    "    for (int i = lid; i < 256; i += 32) h[i] = 0;\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    uint base = (uint)block * 1024u + (uint)lid * 32u;\n"
    "    for (int j = 0; j < 32; j++) {\n"
    "        uint i = base + (uint)j;\n"
    "        if (i < count) atomic_inc(&h[(uint)((records[i].x >> shift) & 0xFFul)]);\n"
    "    }\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    for (int i = lid; i < 256; i += 32)\n"
    "        hist[(uint)i * numBlocks + (uint)block] = h[i];\n"
    "}\n"
    "__kernel void radixScatter(__global const ulong2 *in, uint count, uint shift,\n"
    "                           uint numBlocks, __global const uint *offsets,\n"
    "                           __global ulong2 *out) {\n"
    "    int block = get_group_id(0), lid = get_local_id(0);\n"
    "    __local uint counts[8192];\n"
    "    for (int i = lid; i < 8192; i += 32) counts[i] = 0;\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    uint base = (uint)block * 1024u + (uint)lid * 32u;\n"
    "    uchar digs[32];\n"
    "    for (int j = 0; j < 32; j++) {\n"
    "        uint i = base + (uint)j;\n"
    "        if (i < count) {\n"
    "            digs[j] = (uchar)((in[i].x >> shift) & 0xFFul);\n"
    "            counts[(uint)digs[j] * 32u + (uint)lid]++;\n"
    "        }\n"
    "    }\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    for (int j = 0; j < 32; j++) {\n"
    "        uint i = base + (uint)j;\n"
    "        if (i >= count) continue;\n"
    "        uint d = digs[j];\n"
    "        uint intra = 0;\n"
    "        for (int t = 0; t < lid; t++) intra += counts[d * 32u + (uint)t];\n"
    "        uint own = 0;\n"
    "        for (int k = 0; k < j; k++) own += (digs[k] == (uchar)d) ? 1u : 0u;\n"
    "        out[offsets[d * numBlocks + (uint)block] + intra + own] = in[i];\n"
    "    }\n"
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
#define OCL_SEG_COUNT 65536u
#define OCL_SEG_WIDTH 256

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
static cl_kernel oclKernelCountSegments;
static cl_kernel oclKernelEmitSegments;
static cl_kernel oclKernelScanLocal;
static cl_kernel oclKernelScanBlocks;
static cl_kernel oclKernelAddBlockOffsets;
static cl_kernel oclKernelRadixHist;
static cl_kernel oclKernelRadixScatter;

typedef struct {
    int valid;
    cl_int w, h, s;
    size_t allocImageBytes;
    size_t allocTileBytes;
    size_t allocPixelCount;
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
    cl_mem bufMasks;
    cl_mem bufRecords;
    cl_mem bufRecordsAlt;
    cl_mem bufSortHist;
    cl_mem bufSortOffsets;
    cl_mem bufSegCounts;
    cl_mem bufSegOffsets;
    cl_mem bufBlockSums;
} OclBufferCache;

static OclBufferCache cache;
static uint32_t segCountsHost[OCL_SEG_COUNT];
static uint32_t segOffsetsHost[OCL_SEG_COUNT];

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

    const char *sources[7] = { sourceThreshold, sourceCcl, sourceCompress, sourceExtract, sourceEmit, sourceSort, sourceScan };
    cl_program program = clCreateProgramWithSource(oclContext, 7, sources, NULL, &err);
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
        { &oclKernelCountSegments, "countSegments" },
        { &oclKernelEmitSegments, "emitSegments" },
        { &oclKernelScanLocal, "scanLocal" },
        { &oclKernelScanBlocks, "scanBlocks" },
        { &oclKernelAddBlockOffsets, "addBlockOffsets" },
        { &oclKernelRadixHist, "radixHist" },
        { &oclKernelRadixScatter, "radixScatter" },
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
    releaseBuffer(cache.bufMasks);
    releaseBuffer(cache.bufRecords);
    releaseBuffer(cache.bufRecordsAlt);
    releaseBuffer(cache.bufSortHist);
    releaseBuffer(cache.bufSortOffsets);
    releaseBuffer(cache.bufSegCounts);
    releaseBuffer(cache.bufSegOffsets);
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

    const size_t imageBytes = (size_t)s * (size_t)h;
    const size_t tileBytes = (size_t)tw * (size_t)th;
    const size_t pixelCount = (size_t)w * (size_t)h;

    // Smaller frames reuse the existing (larger) buffers: only the threshold
    // output needs re-zeroing so stride padding from a previous larger frame
    // can't leak into the byte-exact output image.
    if (cache.valid && imageBytes <= cache.allocImageBytes &&
        tileBytes <= cache.allocTileBytes && pixelCount <= cache.allocPixelCount) {
        const cl_uchar zeroByte = 0;
        if (clEnqueueFillBuffer(oclQueue, cache.bufOut, &zeroByte, 1, 0, imageBytes, 0, NULL, NULL) != CL_SUCCESS)
            return 0;
        cache.w = w;
        cache.h = h;
        cache.s = s;
        cache.thresholdOutputFor = NULL;
        return 1;
    }

    releaseCache();
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
    cache.bufMasks = createOrFail(CL_MEM_READ_WRITE, pixelCount, NULL, &failed);
    cache.bufRecords = createOrFail(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, (size_t)OCL_RECORD_CAPACITY * 16, NULL, &failed);
    cache.bufRecordsAlt = createOrFail(CL_MEM_READ_WRITE, (size_t)OCL_RECORD_CAPACITY * 16, NULL, &failed);
    cache.bufSortHist = createOrFail(CL_MEM_READ_WRITE, (size_t)256 * (OCL_RECORD_CAPACITY / 1024) * 4, NULL, &failed);
    cache.bufSortOffsets = createOrFail(CL_MEM_READ_WRITE, (size_t)256 * (OCL_RECORD_CAPACITY / 1024) * 4, NULL, &failed);
    cache.bufSegCounts = createOrFail(CL_MEM_READ_WRITE, OCL_SEG_COUNT * 4, NULL, &failed);
    cache.bufSegOffsets = createOrFail(CL_MEM_READ_WRITE, OCL_SEG_COUNT * 4, NULL, &failed);
    cache.bufBlockSums = createOrFail(CL_MEM_READ_WRITE, 256 * 4, NULL, &failed);
    if (failed) {
        releaseCache();
        return 0;
    }
    cache.allocImageBytes = imageBytes;
    cache.allocTileBytes = tileBytes;
    cache.allocPixelCount = pixelCount;

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

// Open-addressing map from cluster key to cluster index. Keys are never 0
// (the high half is always the larger of two distinct roots), so 0 marks an
// empty slot. Records arrive in raster order, so appending in encounter
// order reproduces the CPU emitter's within-cluster point order.
#define OCL_HASH_BITS 16
#define OCL_HASH_SIZE (1u << OCL_HASH_BITS)

typedef struct {
    uint64_t key;
    uint32_t clusterIdx;
} HashEntry;

typedef struct {
    const uint64_t *records;
    uint32_t recStart, recEnd;
    zarray_t *clusters;
    uint64_t *clusterKeys;
    int clusterCap;
    int failed;
} BuildTask;

static uint32_t hashSlot(uint64_t key)
{
    return (uint32_t)((key * 0x9E3779B97F4A7C15ull) >> (64 - OCL_HASH_BITS));
}

static void doBuildTask(void *p)
{
    BuildTask *task = (BuildTask *)p;
    HashEntry *table = calloc(OCL_HASH_SIZE, sizeof(HashEntry));
    if (table == NULL) {
        task->failed = 1;
        return;
    }
    int clusterCount = 0;

    for (uint32_t i = task->recStart; i < task->recEnd; i++) {
        uint64_t key = task->records[2 * i];
        uint64_t payload = task->records[2 * i + 1];
        uint32_t slot = hashSlot(key);
        while (table[slot].key != 0 && table[slot].key != key)
            slot = (slot + 1) & (OCL_HASH_SIZE - 1);
        if (table[slot].key == 0) {
            if (clusterCount >= (int)(OCL_HASH_SIZE / 2)) {
                task->failed = 1;
                break;
            }
            if (clusterCount == task->clusterCap) {
                task->clusterCap *= 2;
                task->clusterKeys = realloc(task->clusterKeys, sizeof(uint64_t) * task->clusterCap);
            }
            table[slot].key = key;
            table[slot].clusterIdx = (uint32_t)clusterCount;
            zarray_t *cluster = zarray_create(sizeof(OclPt));
            zarray_add(task->clusters, &cluster);
            task->clusterKeys[clusterCount] = key;
            clusterCount++;
        }
        zarray_t *cluster;
        zarray_get(task->clusters, (int)table[slot].clusterIdx, &cluster);
        appendPt(cluster, payload);
    }
    free(table);
}

static void destroyTaskClusters(BuildTask *task)
{
    for (int i = 0; i < zarray_size(task->clusters); i++) {
        zarray_t *cluster;
        zarray_get(task->clusters, i, &cluster);
        zarray_destroy(cluster);
    }
    zarray_destroy(task->clusters);
    free(task->clusterKeys);
}

// Merge per-task clusters in task order: tasks cover ascending row ranges,
// so concatenation preserves raster point order within each cluster.
static zarray_t *mergeTaskClusters(BuildTask *tasks, int taskCount)
{
    zarray_t *clusters = zarray_create(sizeof(zarray_t *));
    HashEntry *table = calloc(OCL_HASH_SIZE, sizeof(HashEntry));
    if (table == NULL) {
        for (int t = 0; t < taskCount; t++)
            destroyTaskClusters(&tasks[t]);
        return clusters;
    }

    for (int t = 0; t < taskCount; t++) {
        for (int i = 0; i < zarray_size(tasks[t].clusters); i++) {
            zarray_t *cluster;
            zarray_get(tasks[t].clusters, i, &cluster);
            uint64_t key = tasks[t].clusterKeys[i];
            uint32_t slot = hashSlot(key);
            while (table[slot].key != 0 && table[slot].key != key)
                slot = (slot + 1) & (OCL_HASH_SIZE - 1);
            if (table[slot].key == 0) {
                table[slot].key = key;
                table[slot].clusterIdx = (uint32_t)zarray_size(clusters);
                zarray_add(clusters, &cluster);
            } else {
                zarray_t *dst;
                zarray_get(clusters, (int)table[slot].clusterIdx, &dst);
                zarray_ensure_capacity(dst, dst->size + cluster->size);
                memcpy(dst->data + (size_t)dst->size * dst->el_sz, cluster->data,
                       (size_t)cluster->size * cluster->el_sz);
                dst->size += cluster->size;
                zarray_destroy(cluster);
            }
        }
        zarray_destroy(tasks[t].clusters);
        free(tasks[t].clusterKeys);
    }
    free(table);
    return clusters;
}

static zarray_t *buildClusters(apriltag_detector_t *td, const uint64_t *records, uint32_t recordCount,
                               int segsPerRow, cl_int h)
{
    int taskCount = (td->wp != NULL && td->nthreads > 1) ? td->nthreads : 1;
    if (taskCount > 16)
        taskCount = 16;
    BuildTask tasks[16];

    // Split rows into contiguous ranges balanced by record count; row r's
    // records start at segOffsetsHost[r * segsPerRow].
    uint32_t targetPerTask = recordCount / (uint32_t)taskCount + 1;
    cl_int row = 0;
    for (int t = 0; t < taskCount; t++) {
        uint32_t recStart = (row < h) ? segOffsetsHost[(size_t)row * segsPerRow] : recordCount;
        tasks[t].records = records;
        tasks[t].recStart = recStart;
        tasks[t].clusters = zarray_create(sizeof(zarray_t *));
        tasks[t].clusterCap = 256;
        tasks[t].clusterKeys = malloc(sizeof(uint64_t) * tasks[t].clusterCap);
        tasks[t].failed = 0;
        while (row < h) {
            row++;
            uint32_t nextStart = (row < h) ? segOffsetsHost[(size_t)row * segsPerRow] : recordCount;
            if (t < taskCount - 1 && nextStart - recStart >= targetPerTask)
                break;
        }
        tasks[t].recEnd = (row < h) ? segOffsetsHost[(size_t)row * segsPerRow] : recordCount;
    }
    tasks[taskCount - 1].recEnd = recordCount;

    if (taskCount == 1) {
        doBuildTask(&tasks[0]);
    } else {
        for (int t = 0; t < taskCount; t++)
            workerpool_add_task(td->wp, doBuildTask, &tasks[t]);
        workerpool_run(td->wp);
    }

    for (int t = 0; t < taskCount; t++) {
        if (tasks[t].failed) {
            for (int u = 0; u < taskCount; u++)
                destroyTaskClusters(&tasks[u]);
            return NULL;
        }
    }
    return mergeTaskClusters(tasks, taskCount);
}

// Stable 6-pass LSD radix sort of the record buffer by the compacted 46-bit
// cluster key. Stability plus raster-ordered input means each cluster ends
// up contiguous with its points in the CPU emitter's exact order. The
// per-(digit, block) offset scan runs on the host between passes. Caller
// holds oclMutex. Returns 0 on failure; on success the sorted records are
// back in cache.bufRecords.
static int sortRecords(uint32_t recordCount)
{
    static uint32_t *histScratch = NULL;
    const uint32_t numBlocks = (recordCount + 1023u) / 1024u;
    const size_t histEntries = (size_t)256 * numBlocks;
    if (histScratch == NULL) {
        histScratch = malloc((size_t)256 * (OCL_RECORD_CAPACITY / 1024) * 4);
        if (histScratch == NULL)
            return 0;
    }

    cl_mem cur = cache.bufRecords;
    cl_mem alt = cache.bufRecordsAlt;
    const size_t sortGlobal[1] = { (size_t)numBlocks * 32 };
    const size_t sortLocal[1] = { 32 };

    for (int pass = 0; pass < 6; pass++) {
        const cl_uint shift = (cl_uint)(pass * 8);
        cl_int err = CL_SUCCESS;
        err |= clSetKernelArg(oclKernelRadixHist, 0, sizeof(cl_mem), &cur);
        err |= clSetKernelArg(oclKernelRadixHist, 1, sizeof(cl_uint), &recordCount);
        err |= clSetKernelArg(oclKernelRadixHist, 2, sizeof(cl_uint), &shift);
        err |= clSetKernelArg(oclKernelRadixHist, 3, sizeof(cl_uint), &numBlocks);
        err |= clSetKernelArg(oclKernelRadixHist, 4, sizeof(cl_mem), &cache.bufSortHist);
        if (err != CL_SUCCESS)
            return 0;
        err = clEnqueueNDRangeKernel(oclQueue, oclKernelRadixHist, 1, NULL, sortGlobal, sortLocal, 0, NULL, profSlot("radixHist"));
        if (err != CL_SUCCESS)
            return 0;
        err = clEnqueueReadBuffer(oclQueue, cache.bufSortHist, CL_TRUE, 0, histEntries * 4, histScratch, 0, NULL, NULL);
        if (err != CL_SUCCESS)
            return 0;
        uint32_t running = 0;
        for (size_t i = 0; i < histEntries; i++) {
            uint32_t v = histScratch[i];
            histScratch[i] = running;
            running += v;
        }
        err = clEnqueueWriteBuffer(oclQueue, cache.bufSortOffsets, CL_FALSE, 0, histEntries * 4, histScratch, 0, NULL, NULL);
        err |= clSetKernelArg(oclKernelRadixScatter, 0, sizeof(cl_mem), &cur);
        err |= clSetKernelArg(oclKernelRadixScatter, 1, sizeof(cl_uint), &recordCount);
        err |= clSetKernelArg(oclKernelRadixScatter, 2, sizeof(cl_uint), &shift);
        err |= clSetKernelArg(oclKernelRadixScatter, 3, sizeof(cl_uint), &numBlocks);
        err |= clSetKernelArg(oclKernelRadixScatter, 4, sizeof(cl_mem), &cache.bufSortOffsets);
        err |= clSetKernelArg(oclKernelRadixScatter, 5, sizeof(cl_mem), &alt);
        if (err != CL_SUCCESS)
            return 0;
        err = clEnqueueNDRangeKernel(oclQueue, oclKernelRadixScatter, 1, NULL, sortGlobal, sortLocal, 0, NULL, profSlot("radixScatter"));
        if (err != CL_SUCCESS)
            return 0;
        cl_mem tmp = cur;
        cur = alt;
        alt = tmp;
    }
    // Six passes: the final output landed back in cache.bufRecords.
    return cur == cache.bufRecords;
}

typedef struct {
    const uint64_t *records;
    uint32_t recStart, recEnd;
    zarray_t *clusters;
} SortedTask;

static void doSortedTask(void *p)
{
    SortedTask *task = (SortedTask *)p;
    zarray_t *cluster = NULL;
    uint64_t currentKey = 0;
    for (uint32_t i = task->recStart; i < task->recEnd; i++) {
        uint64_t key = task->records[2 * i];
        if (cluster == NULL || key != currentKey) {
            cluster = zarray_create(sizeof(OclPt));
            zarray_add(task->clusters, &cluster);
            currentKey = key;
        }
        appendPt(cluster, task->records[2 * i + 1]);
    }
}

// Cluster build over key-sorted records: groups are contiguous, so this is a
// linear walk with no hashing. Task ranges are aligned to key boundaries so
// no cluster spans two tasks.
static zarray_t *buildClustersSorted(apriltag_detector_t *td, const uint64_t *records, uint32_t recordCount)
{
    int taskCount = (td->wp != NULL && td->nthreads > 1) ? td->nthreads : 1;
    if (taskCount > 16)
        taskCount = 16;
    SortedTask tasks[16];

    uint32_t pos = 0;
    for (int t = 0; t < taskCount; t++) {
        tasks[t].records = records;
        tasks[t].recStart = pos;
        tasks[t].clusters = zarray_create(sizeof(zarray_t *));
        uint32_t target = (uint32_t)(((uint64_t)recordCount * (t + 1)) / taskCount);
        if (target < pos)
            target = pos;
        while (target < recordCount && target > 0 && records[2 * target] == records[2 * (target - 1)])
            target++;
        tasks[t].recEnd = target;
        pos = target;
    }
    tasks[taskCount - 1].recEnd = recordCount;

    if (taskCount == 1) {
        doSortedTask(&tasks[0]);
    } else {
        for (int t = 0; t < taskCount; t++)
            workerpool_add_task(td->wp, doSortedTask, &tasks[t]);
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

// Runs CCL + sizes + raster-ordered extraction over the threshold image in
// inputBuffer, then builds the cluster arrays on the CPU. Caller holds
// oclMutex and has a valid cache. labelsReady indicates the classify kernel
// already seeded the labels buffer.
static zarray_t *runClusterChain(apriltag_detector_t *td, cl_mem inputBuffer, cl_int cw, cl_int ch, cl_int cs, int labelsReady)
{
    const cl_uint minCluster = (cl_uint)td->qtp.min_cluster_pixels;
    const cl_uint capacity = OCL_RECORD_CAPACITY;
    const cl_int segsPerRow = (cw + OCL_SEG_WIDTH - 1) / OCL_SEG_WIDTH;
    const cl_int nSegs = segsPerRow * ch;
    zarray_t *clusters = NULL;
    cl_int err = CL_SUCCESS;
    const cl_uint zero = 0;
    const size_t pixelCount = (size_t)cw * (size_t)ch;

    if ((size_t)nSegs > OCL_SEG_COUNT)
        return NULL;

    err |= clEnqueueFillBuffer(oclQueue, cache.bufSizes, &zero, 4, 0, pixelCount * 4, 0, NULL, profSlot("fillSizes"));
    err |= clEnqueueFillBuffer(oclQueue, cache.bufSegCounts, &zero, 4, 0, OCL_SEG_COUNT * 4, 0, NULL, NULL);
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

    err |= clSetKernelArg(oclKernelCountSegments, 0, sizeof(cl_mem), &inputBuffer);
    err |= clSetKernelArg(oclKernelCountSegments, 1, sizeof(cl_mem), &cache.bufBigMap);
    err |= clSetKernelArg(oclKernelCountSegments, 2, sizeof(cl_int), &cs);
    err |= clSetKernelArg(oclKernelCountSegments, 3, sizeof(cl_int), &cw);
    err |= clSetKernelArg(oclKernelCountSegments, 4, sizeof(cl_int), &ch);
    err |= clSetKernelArg(oclKernelCountSegments, 5, sizeof(cl_int), &segsPerRow);
    err |= clSetKernelArg(oclKernelCountSegments, 6, sizeof(cl_int), &nSegs);
    err |= clSetKernelArg(oclKernelCountSegments, 7, sizeof(cl_mem), &cache.bufSegCounts);
    err |= clSetKernelArg(oclKernelCountSegments, 8, sizeof(cl_mem), &cache.bufMasks);

    err |= clSetKernelArg(oclKernelEmitSegments, 0, sizeof(cl_mem), &inputBuffer);
    err |= clSetKernelArg(oclKernelEmitSegments, 1, sizeof(cl_mem), &cache.bufMasks);
    err |= clSetKernelArg(oclKernelEmitSegments, 2, sizeof(cl_mem), &cache.bufLabels);
    err |= clSetKernelArg(oclKernelEmitSegments, 3, sizeof(cl_int), &cs);
    err |= clSetKernelArg(oclKernelEmitSegments, 4, sizeof(cl_int), &cw);
    err |= clSetKernelArg(oclKernelEmitSegments, 5, sizeof(cl_int), &ch);
    err |= clSetKernelArg(oclKernelEmitSegments, 6, sizeof(cl_int), &segsPerRow);
    err |= clSetKernelArg(oclKernelEmitSegments, 7, sizeof(cl_int), &nSegs);
    err |= clSetKernelArg(oclKernelEmitSegments, 8, sizeof(cl_mem), &cache.bufSegOffsets);
    err |= clSetKernelArg(oclKernelEmitSegments, 9, sizeof(cl_uint), &capacity);
    err |= clSetKernelArg(oclKernelEmitSegments, 10, sizeof(cl_mem), &cache.bufRecords);

    err |= clSetKernelArg(oclKernelScanLocal, 0, sizeof(cl_mem), &cache.bufSegCounts);
    err |= clSetKernelArg(oclKernelScanLocal, 1, sizeof(cl_mem), &cache.bufSegOffsets);
    err |= clSetKernelArg(oclKernelScanLocal, 2, sizeof(cl_mem), &cache.bufBlockSums);
    err |= clSetKernelArg(oclKernelScanBlocks, 0, sizeof(cl_mem), &cache.bufBlockSums);
    err |= clSetKernelArg(oclKernelAddBlockOffsets, 0, sizeof(cl_mem), &cache.bufSegOffsets);
    err |= clSetKernelArg(oclKernelAddBlockOffsets, 1, sizeof(cl_mem), &cache.bufBlockSums);
    if (err != CL_SUCCESS)
        goto done;

    const size_t global[2] = { roundUp((size_t)cw, 16), roundUp((size_t)ch, 16) };
    const size_t countGlobal[2] = { roundUp((size_t)cw, 128), (size_t)ch };
    const size_t countLocal[2] = { 128, 1 };
    const size_t segGlobal[1] = { (size_t)nSegs * 256 };
    const size_t segLocal[1] = { 256 };
    const size_t scanGlobal[1] = { OCL_SEG_COUNT };
    const size_t scanLocalSize[1] = { 256 };
    const size_t singleItem[1] = { 1 };

    if (!labelsReady)
        err |= clEnqueueNDRangeKernel(oclQueue, oclKernelInitLabels, 2, NULL, global, NULL, 0, NULL, profSlot("initLabels"));
    err |= clEnqueueNDRangeKernel(oclQueue, oclKernelMergeEdges, 2, NULL, global, NULL, 0, NULL, profSlot("mergeEdges"));
    err |= clEnqueueNDRangeKernel(oclQueue, oclKernelCompressAndCount, 2, NULL, countGlobal, countLocal, 0, NULL, profSlot("compressCount"));
    err |= clEnqueueNDRangeKernel(oclQueue, oclKernelBuildBigMap, 2, NULL, global, NULL, 0, NULL, profSlot("buildBigMap"));
    err |= clEnqueueNDRangeKernel(oclQueue, oclKernelCountSegments, 1, NULL, segGlobal, segLocal, 0, NULL, profSlot("countSegments"));
    err |= clEnqueueNDRangeKernel(oclQueue, oclKernelScanLocal, 1, NULL, scanGlobal, scanLocalSize, 0, NULL, profSlot("scanLocal"));
    err |= clEnqueueNDRangeKernel(oclQueue, oclKernelScanBlocks, 1, NULL, singleItem, NULL, 0, NULL, profSlot("scanBlocks"));
    err |= clEnqueueNDRangeKernel(oclQueue, oclKernelAddBlockOffsets, 1, NULL, scanGlobal, scanLocalSize, 0, NULL, profSlot("addBlockOffs"));
    if (err != CL_SUCCESS)
        goto done;

    // Read counts and pre-emit offsets for the host build walk; the in-order
    // queue keeps them ordered after the scan, and neither stalls the host.
    cl_event offsetsEvent = NULL;
    err |= clEnqueueReadBuffer(oclQueue, cache.bufSegCounts, CL_FALSE, 0, OCL_SEG_COUNT * 4, segCountsHost, 0, NULL, NULL);
    err |= clEnqueueReadBuffer(oclQueue, cache.bufSegOffsets, CL_FALSE, 0, OCL_SEG_COUNT * 4, segOffsetsHost, 0, NULL, &offsetsEvent);
    err |= clEnqueueNDRangeKernel(oclQueue, oclKernelEmitSegments, 1, NULL, segGlobal, segLocal, 0, NULL, profSlot("emitSegments"));
    if (err != CL_SUCCESS) {
        if (offsetsEvent != NULL)
            clReleaseEvent(offsetsEvent);
        goto done;
    }

    // Totals become available mid-chain; waiting here costs nothing extra
    // (the map below blocks on the emit anyway) and bounds the map size.
    clWaitForEvents(1, &offsetsEvent);
    clReleaseEvent(offsetsEvent);
    uint32_t recordCount = segOffsetsHost[nSegs - 1] + segCountsHost[nSegs - 1];
    if (recordCount > capacity) {
        oclDebugLog("record capacity exceeded");
        goto done;
    }
    if (recordCount == 0) {
        clFinish(oclQueue);
        clusters = zarray_create(sizeof(zarray_t *));
        goto done;
    }

    // P1 scaffolding for the GPU fit_quads port: sort records by cluster key
    // on the GPU so groups are contiguous (within-group raster order is
    // preserved by sort stability). Gated until the GPU fit lands.
    int useSorted = getenv("APRILTAG_OPENCL_SORTED") != NULL;
    if (useSorted && !sortRecords(recordCount))
        goto done;

    void *mapped = clEnqueueMapBuffer(oclQueue, cache.bufRecords, CL_TRUE, CL_MAP_READ, 0,
                                      (size_t)recordCount * 16, 0, NULL, profSlot("mapRecords"), &err);
    if (err != CL_SUCCESS)
        goto done;
    if (useSorted)
        clusters = buildClustersSorted(td, (const uint64_t *)mapped, recordCount);
    else
        clusters = buildClusters(td, (const uint64_t *)mapped, recordCount, segsPerRow, ch);
    clEnqueueUnmapMemObject(oclQueue, cache.bufRecords, mapped, 0, NULL, NULL);

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
