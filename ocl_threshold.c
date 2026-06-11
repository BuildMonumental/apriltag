#include "ocl_threshold.h"

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

static const char *sourceGather =
    // Materialize cluster-contiguous records on-device for the GPU fit
    // stages: the host build walk discovers the grouping anyway, so it emits
    // a permutation (source record index per output slot, clusters
    // contiguous, points in raster order) and one coalesced pass gathers the
    // records into that order. Replaces a full GPU key sort.
    "__kernel void gatherRecords(__global const ulong2 *in, __global const uint *perm,\n"
    "                            uint count, __global ulong2 *out) {\n"
    "    uint i = get_global_id(0);\n"
    "    if (i >= count) return;\n"
    "    out[i] = in[perm[i]];\n"
    "}\n";

static const char *sourceFitPrep =
    "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
    "#pragma OPENCL FP_CONTRACT OFF\n"
    // Per-cluster preparation replicating fit_quad's pre-sort steps with
    // identical arithmetic (the CPU build uses no FMA contraction, hence
    // FP_CONTRACT OFF): the do_quad_task/fit_quad filter cascade, bbox,
    // center (double then float, as the CPU's mixed expression evaluates),
    // per-point slopes, the gradient dot accumulated serially in point
    // order, and the slope sort. Keys are (orderedSlopeBits, point index);
    // comparisons use the slope half only and the sort replicates ptsort's
    // exact network — slope ties are pervasive (~88% of sorted clusters on
    // the vide frame), so matching ptsort's tie order is what keeps the
    // downstream fit bit-exact. Flag bits mirror the FIT_* defines in
    // ocl_threshold.c; FIT_SLM_CAP/FIT_BIG_CAP arrive via build options.
    "#define PROCESSED   (1u<<0)\n"
    "#define SKIP_MINPIX (1u<<1)\n"
    "#define SKIP_PERIM  (1u<<2)\n"
    "#define SKIP_AREA   (1u<<3)\n"
    "#define SKIP_BORDER (1u<<4)\n"
    "#define REVERSED    (1u<<5)\n"
    "#define SORT_GLOBAL (1u<<6)\n"
    "#define SKIP_TOOBIG (1u<<7)\n"
    "#define SORTED      (1u<<8)\n"
    "#define SORT_SLM    (1u<<9)\n"
    "inline uint orderedFloatBits(float f) {\n"
    "    uint u = as_uint(f);\n"
    "    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);\n"
    "}\n"
    "inline float cpuSlope(float fx, float fy, float cx, float cy) {\n"
    "    float dx = fx - cx;\n"
    "    float dy = fy - cy;\n"
    "    float quadrant = (dy > 0) ? ((dx > 0) ? 65536.0f : 131072.0f)\n"
    "                              : ((dx > 0) ? 0.0f : -65536.0f);\n"
    "    if (dy < 0) { dy = -dy; dx = -dx; }\n"
    "    if (dx < 0) { float tmp = dx; dx = dy; dy = -tmp; }\n"
    "    return quadrant + dy / dx;\n"
    "}\n"
    "inline void writeMeta(__global uint *meta, uint c, uint flags, float cx, float cy,\n"
    "                      float dot, uint xmin, uint xmax, uint ymin, uint ymax, uint n) {\n"
    "    __global uint *m = meta + 8u * c;\n"
    "    m[0] = flags; m[1] = as_uint(cx); m[2] = as_uint(cy); m[3] = as_uint(dot);\n"
    "    m[4] = xmin | (xmax << 16); m[5] = ymin | (ymax << 16); m[6] = n; m[7] = 0u;\n"
    "}\n"
    // All four bbox reductions share one tree (and its 8 barriers): the
    // max fields are stored complemented so a single min reduces them all.
    "inline void reduceBbox(__local uint *s4, int lid, uint xmin, uint xmax, uint ymin, uint ymax,\n"
    "                       uint *oXmin, uint *oXmax, uint *oYmin, uint *oYmax) {\n"
    "    s4[lid] = xmin; s4[256 + lid] = ~xmax; s4[512 + lid] = ymin; s4[768 + lid] = ~ymax;\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    for (int s = 128; s > 0; s >>= 1) {\n"
    "        if (lid < s) {\n"
    "            s4[lid] = min(s4[lid], s4[lid + s]);\n"
    "            s4[256 + lid] = min(s4[256 + lid], s4[256 + lid + s]);\n"
    "            s4[512 + lid] = min(s4[512 + lid], s4[512 + lid + s]);\n"
    "            s4[768 + lid] = min(s4[768 + lid], s4[768 + lid + s]);\n"
    "        }\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    }\n"
    "    *oXmin = s4[0]; *oXmax = ~s4[256]; *oYmin = s4[512]; *oYmax = ~s4[768];\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "}\n";

static const char *sourceFitSortHelpers =
    // ptsort's recursion tree is pure arithmetic on the cluster size:
    // floor-half/rest splits, terminating at five or fewer elements where
    // the CPU runs fixed sorting networks. ptsortNode locates the node for
    // a (depth, path) pair; net5 replicates the exact swap sequences and
    // ptsortMerge the exact right-biased two-pointer merge (ties take the
    // right run). Comparisons use only the ordered-slope half of the key.
    "#define KGT(a, b) ((uint)((a) >> 32) > (uint)((b) >> 32))\n"
    "#define KLT(a, b) ((uint)((a) >> 32) < (uint)((b) >> 32))\n"
    "inline int ptsortNode(uint n, uint depth, uint path, uint *outOff, uint *outSize) {\n"
    "    uint off = 0, m = n;\n"
    "    for (uint b = 0; b < depth; b++) {\n"
    "        if (m <= 5u) return 0;\n"
    "        uint hsz = m / 2u;\n"
    "        if ((path >> (depth - 1u - b)) & 1u) { off += hsz; m -= hsz; }\n"
    "        else { m = hsz; }\n"
    "    }\n"
    "    *outOff = off;\n"
    "    *outSize = m;\n"
    "    return 1;\n"
    "}\n"
    "#define SW(i, j) if (KGT(p[i], p[j])) { t = p[i]; p[i] = p[j]; p[j] = t; }\n"
    "#define NET5_BODY \\\n"
    "    ulong t; \\\n"
    "    if (m == 2u) { SW(0, 1) } \\\n"
    "    else if (m == 3u) { SW(0, 1) SW(1, 2) SW(0, 1) } \\\n"
    "    else if (m == 4u) { SW(0, 1) SW(2, 3) SW(0, 2) SW(1, 3) SW(1, 2) } \\\n"
    "    else if (m == 5u) { SW(0, 1) SW(3, 4) SW(1, 2) SW(0, 1) SW(0, 3) SW(2, 4) SW(1, 2) SW(2, 3) SW(1, 2) }\n"
    "inline void net5L(__local ulong *p, uint m) { NET5_BODY }\n"
    "inline void net5G(__global ulong *p, uint m) { NET5_BODY }\n"
    "#define MERGE_BODY \\\n"
    "    uint roff = loff + lsz; \\\n"
    "    uint i = 0, j = 0, o = loff; \\\n"
    "    while (i < lsz && j < rsz) { \\\n"
    "        ulong a = src[loff + i]; \\\n"
    "        ulong b = src[roff + j]; \\\n"
    "        if (KLT(a, b)) { dst[o++] = a; i++; } \\\n"
    "        else { dst[o++] = b; j++; } \\\n"
    "    } \\\n"
    "    while (i < lsz) dst[o++] = src[loff + i++]; \\\n"
    "    while (j < rsz) dst[o++] = src[roff + j++];\n"
    "inline void ptsortMergeL(__local const ulong *src, __local ulong *dst, uint loff, uint lsz, uint rsz) { MERGE_BODY }\n"
    "inline void ptsortMergeG(__global const ulong *src, __global ulong *dst, uint loff, uint lsz, uint rsz) { MERGE_BODY }\n";

static const char *sourceFitSortHelpers2 =
    // Merge-path split for lane-parallel merges: returns how many elements
    // of a the exact right-biased serial merge consumes among its first k
    // outputs, so a lane can start mid-merge and produce an identical
    // output chunk.
    "#define MPS_BODY \\\n"
    "    uint lo = (k > bsz) ? (k - bsz) : 0u; \\\n"
    "    uint hi = (k < asz) ? k : asz; \\\n"
    "    while (lo < hi) { \\\n"
    "        uint mid = (lo + hi) >> 1u; \\\n"
    "        if (KLT(a[mid], b[k - mid - 1u])) lo = mid + 1u; else hi = mid; \\\n"
    "    } \\\n"
    "    return lo;\n"
    "inline uint mergePathSearchG(__global const ulong *a, uint asz, __global const ulong *b, uint bsz, uint k) { MPS_BODY }\n"
    "inline uint mergePathSearchL(__local const ulong *a, uint asz, __local const ulong *b, uint bsz, uint k) { MPS_BODY }\n"
    // One ptsort depth pass: leaves copy (when the parity differs from the
    // input buffer) and run their network, internal nodes merge their
    // children from the opposite-parity buffer. Depth-d results land in
    // bufA when d is even, bufB when odd; the raster-order input lives in
    // bufA, and a leaf's bufA region is untouched by other nodes' merges
    // until its own depth is processed.
    "#define DEPTH_BODY(MERGEFN, NETFN) \\\n"
    "    for (uint node = (uint)lid; node < (1u << d); node += 256u) { \\\n"
    "        uint noff, nsz; \\\n"
    "        if (!ptsortNode(n, (uint)d, node, &noff, &nsz)) continue; \\\n"
    "        int even = (d & 1) == 0; \\\n"
    "        if (nsz <= 5u) { \\\n"
    "            if (!even) for (uint q = 0; q < nsz; q++) bufB[noff + q] = bufA[noff + q]; \\\n"
    "            if (even) NETFN(bufA + noff, nsz); else NETFN(bufB + noff, nsz); \\\n"
    "        } else { \\\n"
    "            uint hsz = nsz / 2u; \\\n"
    "            if (even) MERGEFN(bufB, bufA, noff, hsz, nsz - hsz); \\\n"
    "            else MERGEFN(bufA, bufB, noff, hsz, nsz - hsz); \\\n"
    "        } \\\n"
    "    }\n"
    // Shallow-depth pass (fewer nodes than lanes): each node's merge is
    // split across its lane group with the merge-path search, which lets a
    // lane start mid-merge and still produce the exact serial output. AS is
    // the buffers' address space.
    "#define SHALLOW_BODY(AS, MPSFN, NETFN) \\\n"
    "    uint lanesPerNode = 256u >> d; \\\n"
    "    uint node = (uint)lid / lanesPerNode; \\\n"
    "    uint lane = (uint)lid % lanesPerNode; \\\n"
    "    uint noff, nsz; \\\n"
    "    if (ptsortNode(n, (uint)d, node, &noff, &nsz)) { \\\n"
    "        int even = (d & 1) == 0; \\\n"
    "        AS ulong *src = even ? bufB : bufA; \\\n"
    "        AS ulong *dst = even ? bufA : bufB; \\\n"
    "        if (nsz <= 5u) { \\\n"
    "            if (lane == 0) { \\\n"
    "                if (!even) for (uint q = 0; q < nsz; q++) bufB[noff + q] = bufA[noff + q]; \\\n"
    "                if (even) NETFN(bufA + noff, nsz); else NETFN(bufB + noff, nsz); \\\n"
    "            } \\\n"
    "        } else { \\\n"
    "            uint hsz = nsz / 2u; \\\n"
    "            uint rsz = nsz - hsz; \\\n"
    "            uint chunk = (nsz + lanesPerNode - 1u) / lanesPerNode; \\\n"
    "            uint k0 = lane * chunk; \\\n"
    "            if (k0 < nsz) { \\\n"
    "                uint k1 = (k0 + chunk < nsz) ? (k0 + chunk) : nsz; \\\n"
    "                uint ai = MPSFN(src + noff, hsz, src + noff + hsz, rsz, k0); \\\n"
    "                uint bi = k0 - ai; \\\n"
    "                for (uint k = k0; k < k1; k++) { \\\n"
    "                    int takeA = (ai < hsz) && ((bi >= rsz) || KLT(src[noff + ai], src[noff + hsz + bi])); \\\n"
    "                    if (takeA) { dst[noff + k] = src[noff + ai]; ai++; } \\\n"
    "                    else { dst[noff + k] = src[noff + hsz + bi]; bi++; } \\\n"
    "                } \\\n"
    "            } \\\n"
    "        } \\\n"
    "    }\n";

static const char *sourceFitPrep2 =
    // Preparation only — the sort runs in fitSortSlm/fitSortBig so this
    // kernel keeps a small SLM footprint and full occupancy. The gradient
    // dot's per-point terms are computed chunk by chunk into local memory
    // (the chunked walk has the same coalesced access pattern as a strided
    // one); one lane sums each chunk in cluster point order, which
    // reproduces the CPU's float accumulation exactly without a global
    // round trip for the terms.
    "__kernel void fitPrep(__global const ulong2 *records, __global const uint2 *desc,\n"
    "                      uint clusterCount, int minClusterPixels, int perimCap, int tagWidth,\n"
    "                      int normalAllowed, int reversedAllowed,\n"
    "                      __global ulong *keys, __global uint *meta) {\n"
    "    uint c = get_group_id(0);\n"
    "    if (c >= clusterCount) return;\n"
    "    uint off = desc[c].x;\n"
    "    uint n = desc[c].y;\n"
    "    int lid = get_local_id(0);\n"
    "    __local uint s4[1024];\n"
    "    __local float sterms[256];\n"
    "    uint flags = PROCESSED;\n"
    "    if ((int)n < minClusterPixels) flags |= SKIP_MINPIX;\n"
    "    else if ((int)n > perimCap) flags |= SKIP_PERIM;\n"
    "    if (flags != PROCESSED) {\n"
    "        if (lid == 0) writeMeta(meta, c, flags, 0.0f, 0.0f, 0.0f, 0u, 0u, 0u, 0u, n);\n"
    "        return;\n"
    "    }\n"
    "    uint lxmin = 65535u, lxmax = 0u, lymin = 65535u, lymax = 0u;\n"
    "    for (uint i = (uint)lid; i < n; i += 256u) {\n"
    "        ulong payload = records[off + i].y;\n"
    "        uint px = (uint)((payload >> 48) & 0xFFFFul);\n"
    "        uint py = (uint)((payload >> 32) & 0xFFFFul);\n"
    "        lxmin = min(lxmin, px); lxmax = max(lxmax, px);\n"
    "        lymin = min(lymin, py); lymax = max(lymax, py);\n"
    "    }\n"
    "    uint xmin, xmax, ymin, ymax;\n"
    "    reduceBbox(s4, lid, lxmin, lxmax, lymin, lymax, &xmin, &xmax, &ymin, &ymax);\n"
    "    if ((int)(xmax - xmin) * (int)(ymax - ymin) < tagWidth) {\n"
    "        if (lid == 0) writeMeta(meta, c, flags | SKIP_AREA, 0.0f, 0.0f, 0.0f, xmin, xmax, ymin, ymax, n);\n"
    "        return;\n"
    "    }\n"
    "    float cx = (float)((xmin + xmax) * 0.5 + 0.05118);\n"
    "    float cy = (float)((ymin + ymax) * 0.5 - 0.028581);\n"
    "    float dot = 0.0f;\n"
    "    for (uint chunk = 0; chunk < n; chunk += 256u) {\n"
    "        uint i = chunk + (uint)lid;\n"
    "        if (i < n) {\n"
    "            ulong payload = records[off + i].y;\n"
    "            float fx = (float)((payload >> 48) & 0xFFFFul);\n"
    "            float fy = (float)((payload >> 32) & 0xFFFFul);\n"
    "            float gx = (float)as_short((ushort)((payload >> 16) & 0xFFFFul));\n"
    "            float gy = (float)as_short((ushort)(payload & 0xFFFFul));\n"
    "            float slope = cpuSlope(fx, fy, cx, cy);\n"
    "            keys[off + i] = ((ulong)orderedFloatBits(slope) << 32) | (ulong)i;\n"
    "            float dx = fx - cx;\n"
    "            float dy = fy - cy;\n"
    "            sterms[lid] = dx * gx + dy * gy;\n"
    "        }\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "        if (lid == 0) {\n"
    "            uint m = min(n - chunk, 256u);\n"
    "            for (uint j = 0; j < m; j++) dot += sterms[j];\n"
    "        }\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    }\n"
    "    if (lid == 0) {\n"
    "        int rev = dot < 0;\n"
    "        if (rev) flags |= REVERSED;\n"
    "        if (rev ? !reversedAllowed : !normalAllowed) flags |= SKIP_BORDER;\n"
    "        if ((flags & SKIP_BORDER) == 0) {\n"
    "            if (n <= (uint)FIT_SLM_CAP) flags |= SORT_SLM;\n"
    "            else if (n <= (uint)FIT_BIG_CAP) flags |= SORT_GLOBAL;\n"
    "            else flags |= SKIP_TOOBIG;\n"
    "        }\n"
    "        writeMeta(meta, c, flags, cx, cy, dot, xmin, xmax, ymin, ymax, n);\n"
    "    }\n"
    "}\n";

static const char *sourceFitSortSlm =
    // Slope sort for SLM-sized clusters (ids in sortList): the ptsort
    // replica over two SLM buffers. Deep levels assign one lane per node;
    // shallow levels (few, large merges) split each merge across the
    // node's lane group with the merge-path search.
    "__kernel void fitSortSlm(__global ulong *keys, __global const uint2 *desc,\n"
    "                         __global uint *meta, __global const uint *sortList, uint count) {\n"
    "    uint g = get_group_id(0);\n"
    "    if (g >= count) return;\n"
    "    uint c = sortList[g];\n"
    "    uint flags = meta[8u * c];\n"
    "    if ((flags & SORT_SLM) == 0u) return;\n"
    "    uint off = desc[c].x;\n"
    "    uint n = desc[c].y;\n"
    "    int lid = get_local_id(0);\n"
    "    __local ulong skeysA[FIT_SLM_CAP];\n"
    "    __local ulong skeysB[FIT_SLM_CAP];\n"
    "    for (uint i = (uint)lid; i < n; i += 256u) skeysA[i] = keys[off + i];\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    __local ulong *bufA = skeysA;\n"
    "    __local ulong *bufB = skeysB;\n"
    "    for (int d = 9; d >= 0; d--) {\n"
    "        if ((1u << d) >= 256u) {\n"
    "            DEPTH_BODY(ptsortMergeL, net5L)\n"
    "        } else {\n"
    "            SHALLOW_BODY(__local, mergePathSearchL, net5L)\n"
    "        }\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    }\n"
    "    for (uint i = (uint)lid; i < n; i += 256u) keys[off + i] = skeysA[i];\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    if (lid == 0) meta[8u * c] = (flags & ~SORT_SLM) | SORTED;\n"
    "}\n";

static const char *sourceFitSort =
    // Slope sort for clusters too large for SLM: one workgroup per
    // oversized cluster (ids in the big segment of sortList, one batch per
    // launch), running the same ptsort replica in global memory. The
    // cluster's keys segment is bufA (raster-order input and final output),
    // the workgroup's scratch slice is bufB. Deep levels assign one lane
    // per node; shallow levels (few, large merges) split each merge across
    // lanes with mergePathSearch, which reproduces the exact serial output.
    "__kernel void fitSortBig(__global ulong *keys, __global const uint2 *desc,\n"
    "                         __global uint *meta, __global const uint *sortList,\n"
    "                         uint baseIdx, uint count, __global ulong *scratch) {\n"
    "    uint g = get_group_id(0);\n"
    "    if (g >= count) return;\n"
    "    uint c = sortList[baseIdx + g];\n"
    "    uint flags = meta[8u * c];\n"
    "    if ((flags & SORT_GLOBAL) == 0u) return;\n"
    "    uint n = desc[c].y;\n"
    "    int lid = get_local_id(0);\n"
    "    __global ulong *bufA = keys + desc[c].x;\n"
    "    __global ulong *bufB = scratch + (size_t)g * (size_t)FIT_BIG_CAP;\n"
    "    for (int d = 13; d >= 0; d--) {\n"
    "        if ((1u << d) >= 256u) {\n"
    "            DEPTH_BODY(ptsortMergeG, net5G)\n"
    "        } else {\n"
    "            SHALLOW_BODY(__global, mergePathSearchG, net5G)\n"
    "        }\n"
    "        barrier(CLK_GLOBAL_MEM_FENCE);\n"
    "    }\n"
    "    if (lid == 0) meta[8u * c] = (flags & ~SORT_GLOBAL) | SORTED;\n"
    "}\n";

static const char *sourceFitLfps =
    // compute_lfps replica over the sorted point order (P3), fused: each
    // cluster's workgroup resolves the sorted indirection, samples the
    // grayscale weight, and computes one 256-point chunk of the six moment
    // TERMS — the exact per-statement products the CPU forms (W*fx,
    // (W*fx)*fx, ...) — into local memory; lanes 0-5 (one per field, SIMD
    // lockstep) then extend the six cumulative sums in CPU accumulation
    // order, the minimal serial work the exactness contract allows. The
    // raw terms never travel through global memory (the split prep/scan
    // pair moved them out and back — two thirds of the chain's traffic)
    // and each cluster flows through prep and scan independently. Output
    // planes (six per-field planes of lfStride doubles) are unchanged.
    "__kernel void fitLfps(__global const ulong2 *records, __global const ulong *keys,\n"
    "                      __global const uint2 *desc, __global const uint2 *fitList, uint count,\n"
    "                      __global const uchar *im, int imW, int imH, int imS,\n"
    "                      uint lfStride, __global double *lfps) {\n"
    "    uint g = get_group_id(0);\n"
    "    if (g >= count) return;\n"
    "    uint c = fitList[g].x;\n"
    "    uint lo = fitList[g].y;\n"
    "    uint off = desc[c].x;\n"
    "    uint n = desc[c].y;\n"
    "    int lid = get_local_id(0);\n"
    "    __local double terms[256 * 6];\n"
    "    double acc = 0.0;\n"
    "    for (uint chunk = 0; chunk < n; chunk += 256u) {\n"
    "        uint i = chunk + (uint)lid;\n"
    "        if (i < n) {\n"
    "            ulong payload = records[off + (uint)(keys[off + i] & 0xFFFFFFFFul)].y;\n"
    "            int px = (int)((payload >> 48) & 0xFFFFul);\n"
    "            int py = (int)((payload >> 32) & 0xFFFFul);\n"
    "            double x = px * 0.5 + 0.5;\n"
    "            double y = py * 0.5 + 0.5;\n"
    "            int ix = (int)x, iy = (int)y;\n"
    "            double W = 1.0;\n"
    "            if (ix > 0 && ix + 1 < imW && iy > 0 && iy + 1 < imH) {\n"
    "                int gradX = (int)im[iy * imS + ix + 1] - (int)im[iy * imS + ix - 1];\n"
    "                int gradY = (int)im[(iy + 1) * imS + ix] - (int)im[(iy - 1) * imS + ix];\n"
    "                W = sqrt((double)(gradX * gradX + gradY * gradY)) + 1.0;\n"
    "            }\n"
    "            double fx = x, fy = y;\n"
    "            __local double *t = terms + 6 * lid;\n"
    "            t[0] = W * fx;\n"
    "            t[1] = W * fy;\n"
    "            t[2] = W * fx * fx;\n"
    "            t[3] = W * fx * fy;\n"
    "            t[4] = W * fy * fy;\n"
    "            t[5] = W;\n"
    "        }\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "        if (lid < 6) {\n"
    "            uint m = min(n - chunk, 256u);\n"
    "            __global double *plane = lfps + (uint)lid * lfStride + lo + chunk;\n"
    "            for (uint j = 0; j < m; j++) {\n"
    "                acc += terms[6u * j + (uint)lid];\n"
    "                plane[j] = acc;\n"
    "            }\n"
    "        }\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    }\n"
    "}\n";

static const char *sourceFitLine =
    // fit_line replicas over the cumulative moments, laid out as 6 doubles
    // per point [Mx My Mxx Mxy Myy W]. The CPU branches and operation order
    // are kept exactly: one subtraction when i0 > 0, last-minus-prev plus i1
    // on wraparound, divides in double, and sqrtf-on-double narrowing (the
    // CPU calls sqrtf on double expressions) as (double)sqrt((float)x).
    "#define LFP(p, i) base[(p) * lfStride + (uint)(i)]\n"
    "inline double fitErrAt(__global const double *base, uint lfStride, int sz, int i0, int i1) {\n"
    "    double mx, my, mxx, mxy, myy, mw;\n"
    "    int N;\n"
    "    if (i0 < i1) {\n"
    "        mx = LFP(0u, i1); my = LFP(1u, i1); mxx = LFP(2u, i1);\n"
    "        mxy = LFP(3u, i1); myy = LFP(4u, i1); mw = LFP(5u, i1);\n"
    "        if (i0 > 0) {\n"
    "            mx -= LFP(0u, i0 - 1); my -= LFP(1u, i0 - 1); mxx -= LFP(2u, i0 - 1);\n"
    "            mxy -= LFP(3u, i0 - 1); myy -= LFP(4u, i0 - 1); mw -= LFP(5u, i0 - 1);\n"
    "        }\n"
    "        N = i1 - i0 + 1;\n"
    "    } else {\n"
    "        mx = LFP(0u, sz - 1) - LFP(0u, i0 - 1); my = LFP(1u, sz - 1) - LFP(1u, i0 - 1);\n"
    "        mxx = LFP(2u, sz - 1) - LFP(2u, i0 - 1); mxy = LFP(3u, sz - 1) - LFP(3u, i0 - 1);\n"
    "        myy = LFP(4u, sz - 1) - LFP(4u, i0 - 1); mw = LFP(5u, sz - 1) - LFP(5u, i0 - 1);\n"
    "        mx += LFP(0u, i1); my += LFP(1u, i1); mxx += LFP(2u, i1);\n"
    "        mxy += LFP(3u, i1); myy += LFP(4u, i1); mw += LFP(5u, i1);\n"
    "        N = sz - i0 + i1 + 1;\n"
    "    }\n"
    "    double ex = mx / mw;\n"
    "    double ey = my / mw;\n"
    "    double cxx = mxx / mw - ex * ex;\n"
    "    double cxy = mxy / mw - ex * ey;\n"
    "    double cyy = myy / mw - ey * ey;\n"
    "    double eigSmall = 0.5 * (cxx + cyy - (double)sqrt((float)((cxx - cyy) * (cxx - cyy) + 4.0 * cxy * cxy)));\n"
    "    return N * eigSmall;\n"
    "}\n"
    // Cached variant for the combo search: the at/prev moment rows of the
    // (at most FIT_MAX_K) maxima plus the last row live in local memory,
    // indexed by maxima slot. Arithmetic identical to fitErrAt.
    "inline void fitLineC(__local const double *lfA, __local const double *lfP, __local const double *lfL,\n"
    "                     int sz, int i0, int i1, int k0, int k1,\n"
    "                     double *lineparm, double *err, double *mse) {\n"
    "    double mx, my, mxx, mxy, myy, mw;\n"
    "    int N;\n"
    "    __local const double *b = lfA + 6 * k1;\n"
    "    __local const double *a = lfP + 6 * k0;\n"
    "    if (i0 < i1) {\n"
    "        mx = b[0]; my = b[1]; mxx = b[2]; mxy = b[3]; myy = b[4]; mw = b[5];\n"
    "        if (i0 > 0) {\n"
    "            mx -= a[0]; my -= a[1]; mxx -= a[2]; mxy -= a[3]; myy -= a[4]; mw -= a[5];\n"
    "        }\n"
    "        N = i1 - i0 + 1;\n"
    "    } else {\n"
    "        mx = lfL[0] - a[0]; my = lfL[1] - a[1]; mxx = lfL[2] - a[2];\n"
    "        mxy = lfL[3] - a[3]; myy = lfL[4] - a[4]; mw = lfL[5] - a[5];\n"
    "        mx += b[0]; my += b[1]; mxx += b[2]; mxy += b[3]; myy += b[4]; mw += b[5];\n"
    "        N = sz - i0 + i1 + 1;\n"
    "    }\n"
    "    double ex = mx / mw;\n"
    "    double ey = my / mw;\n"
    "    double cxx = mxx / mw - ex * ex;\n"
    "    double cxy = mxy / mw - ex * ey;\n"
    "    double cyy = myy / mw - ey * ey;\n"
    "    double sq = (double)sqrt((float)((cxx - cyy) * (cxx - cyy) + 4.0 * cxy * cxy));\n"
    "    double eigSmall = 0.5 * (cxx + cyy - sq);\n"
    "    if (lineparm) {\n"
    "        lineparm[0] = ex;\n"
    "        lineparm[1] = ey;\n"
    "        double eig = 0.5 * (cxx + cyy + sq);\n"
    "        double nx1 = cxx - eig, ny1 = cxy;\n"
    "        double m1 = nx1 * nx1 + ny1 * ny1;\n"
    "        double nx2 = cxy, ny2 = cyy - eig;\n"
    "        double m2 = nx2 * nx2 + ny2 * ny2;\n"
    "        double nx, ny, mm;\n"
    "        if (m1 > m2) { nx = nx1; ny = ny1; mm = m1; } else { nx = nx2; ny = ny2; mm = m2; }\n"
    "        double length = (double)sqrt((float)mm);\n"
    "        if (fabs(length) < 1e-12) { lineparm[2] = 0; lineparm[3] = 0; }\n"
    "        else { lineparm[2] = nx / length; lineparm[3] = ny / length; }\n"
    "    }\n"
    "    *err = N * eigSmall;\n"
    "    *mse = eigSmall;\n"
    "}\n";

static const char *sourceFitErrs =
    // quad_segment_maxima's front half (P3): windowed errors per point, the
    // fixed 7-tap low-pass (FILT* baked in from the host's libm exp, the
    // values the CPU computes at runtime), maxima detection, then the
    // max_nmaxima cut. The cut's qsort is order-irrelevant on the CPU — only
    // the value at index max_nmaxima is read as a strict threshold — so a
    // top-K multiset selection reproduces it exactly.
    "__kernel void fitErrs(__global const double *lfps, uint lfStride, __global const uint2 *desc,\n"
    "                      __global const uint2 *fitList, uint count, int maxNmaxima,\n"
    "                      __global double *errsRaw, __global double *errsSmooth,\n"
    "                      __global uint *maximaScratch, __global uint *maximaOut,\n"
    "                      __global uint *fitOut) {\n"
    "    uint g = get_group_id(0);\n"
    "    if (g >= count) return;\n"
    "    uint c = fitList[g].x;\n"
    "    uint lo = fitList[g].y;\n"
    "    uint n = desc[c].y;\n"
    "    int lid = get_local_id(0);\n"
    "    __local uint scanBuf[256];\n"
    "    int ksz = min(20, (int)n / 12);\n"
    "    if (ksz < 2) {\n"
    "        if (lid == 0) fitOut[g * FIT_OUT_STRIDE] = 2u;\n"
    "        return;\n"
    "    }\n"
    "    __global const double *lf = lfps + lo;\n"
    "    for (uint i = (uint)lid; i < n; i += 256u)\n"
    "        errsRaw[lo + i] = fitErrAt(lf, lfStride, (int)n, (int)((i + n - (uint)ksz) % n), (int)((i + (uint)ksz) % n));\n"
    "    barrier(CLK_GLOBAL_MEM_FENCE);\n"
    "    for (uint i = (uint)lid; i < n; i += 256u) {\n"
    "        double acc = 0;\n"
    "        acc += errsRaw[lo + (i + n - 3u) % n] * FILT0;\n"
    "        acc += errsRaw[lo + (i + n - 2u) % n] * FILT1;\n"
    "        acc += errsRaw[lo + (i + n - 1u) % n] * FILT2;\n"
    "        acc += errsRaw[lo + i] * FILT3;\n"
    "        acc += errsRaw[lo + (i + 1u) % n] * FILT4;\n"
    "        acc += errsRaw[lo + (i + 2u) % n] * FILT5;\n"
    "        acc += errsRaw[lo + (i + 3u) % n] * FILT6;\n"
    "        errsSmooth[lo + i] = acc;\n"
    "    }\n"
    "    barrier(CLK_GLOBAL_MEM_FENCE);\n"
    // Order-preserving compaction of maxima indices: per-256 chunk local
    // scan with a running base, so maximaScratch holds ascending indices.
    "    uint base = 0;\n"
    "    for (uint chunk = 0; chunk < n; chunk += 256u) {\n"
    "        uint i = chunk + (uint)lid;\n"
    "        int isMax = 0;\n"
    "        if (i < n) {\n"
    "            double v = errsSmooth[lo + i];\n"
    "            isMax = (v > errsSmooth[lo + (i + 1u) % n]) && (v > errsSmooth[lo + (i + n - 1u) % n]);\n"
    "        }\n"
    "        scanBuf[lid] = (uint)isMax;\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "        for (int s = 1; s < 256; s <<= 1) {\n"
    "            uint v2 = (lid >= s) ? scanBuf[lid - s] : 0u;\n"
    "            barrier(CLK_LOCAL_MEM_FENCE);\n"
    "            scanBuf[lid] += v2;\n"
    "            barrier(CLK_LOCAL_MEM_FENCE);\n"
    "        }\n"
    "        if (isMax) maximaScratch[lo + base + scanBuf[lid] - 1u] = i;\n"
    "        base += scanBuf[255];\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    }\n"
    "    uint nmax = base;\n"
    "    if (nmax < 4u) {\n"
    "        if (lid == 0) fitOut[g * FIT_OUT_STRIDE] = 2u;\n"
    "        return;\n"
    "    }\n"
    "    if (lid != 0) return;\n"
    "    uint m = 0;\n"
    "    __global uint *outIdx = maximaOut + g * FIT_MAXIMA_STRIDE + 1u;\n"
    "    if (nmax > (uint)maxNmaxima) {\n"
    "        double top[FIT_MAX_K + 1];\n"
    "        int kOne = maxNmaxima + 1;\n"
    "        int filled = 0;\n"
    "        for (uint j = 0; j < nmax; j++) {\n"
    "            double v = errsSmooth[lo + maximaScratch[lo + j]];\n"
    "            if (filled == kOne && !(v > top[kOne - 1])) continue;\n"
    "            int pos = (filled < kOne) ? filled : (kOne - 1);\n"
    "            while (pos > 0 && top[pos - 1] < v) { top[pos] = top[pos - 1]; pos--; }\n"
    "            top[pos] = v;\n"
    "            if (filled < kOne) filled++;\n"
    "        }\n"
    "        double thr = top[maxNmaxima];\n"
    "        for (uint j = 0; j < nmax; j++) {\n"
    "            uint idx = maximaScratch[lo + j];\n"
    "            if (errsSmooth[lo + idx] <= thr) continue;\n"
    "            outIdx[m] = idx; m++;\n"
    "        }\n"
    "    } else {\n"
    "        for (uint j = 0; j < nmax; j++) { outIdx[m] = maximaScratch[lo + j]; m++; }\n"
    "    }\n"
    "    maximaOut[g * FIT_MAXIMA_STRIDE] = m;\n"
    "}\n";

static const char *sourceFitCombosA =
    // Combo search + final quad (P3 tail). Combos are evaluated in the
    // CPU's lexicographic (m0,m1,m2,m3) order via rank decoding; the argmin
    // reduce breaks exact err ties by lower rank, replicating the CPU's
    // strict first-wins scan.
    "inline uint chooseN(uint n, uint k) {\n"
    "    if (n < k) return 0u;\n"
    "    if (k == 1u) return n;\n"
    "    if (k == 2u) return n * (n - 1u) / 2u;\n"
    "    return n * (n - 1u) * (n - 2u) / 6u;\n"
    "}\n"
    "inline void decodeCombo(uint rank, uint m, uint *a, uint *b, uint *c, uint *d) {\n"
    "    uint r = rank;\n"
    "    uint i = 0;\n"
    "    while (chooseN(m - 1u - i, 3u) <= r) { r -= chooseN(m - 1u - i, 3u); i++; }\n"
    "    *a = i; i++;\n"
    "    while (chooseN(m - 1u - i, 2u) <= r) { r -= chooseN(m - 1u - i, 2u); i++; }\n"
    "    *b = i; i++;\n"
    "    while (chooseN(m - 1u - i, 1u) <= r) { r -= chooseN(m - 1u - i, 1u); i++; }\n"
    "    *c = i; i++;\n"
    "    *d = i + r;\n"
    "}\n"
    // The post-corner checks subtract corners in FLOAT (the CPU reads float
    // quad->p fields) before promoting to double — sqf keeps that order.
    "inline double sqf(float d) { double v = (double)d; return v * v; }\n"
    "inline uint pairIdx(uint k0, uint k1, uint m) { return k0 * (2u * m - k0 - 1u) / 2u + (k1 - k0 - 1u); }\n"
    "#define FIT_PAIRS (FIT_MAX_K * (FIT_MAX_K - 1) / 2)\n"
    "__kernel void fitCombos(__global const double *lfps, uint lfStride, __global const uint2 *desc,\n"
    "                        __global const uint2 *fitList, uint count,\n"
    "                        __global const uint *maximaOut, double maxDot, double maxMse,\n"
    "                        int tagWidth, __global uint *fitOut) {\n"
    "    uint g = get_group_id(0);\n"
    "    if (g >= count) return;\n"
    "    int lid = get_local_id(0);\n"
    "    __local uint midx[FIT_MAX_K];\n"
    "    __local double lfA[FIT_MAX_K * 6];\n"
    "    __local double lfP[FIT_MAX_K * 6];\n"
    "    __local double lfL[6];\n"
    "    __local double pairErrF[FIT_PAIRS];\n"
    "    __local double pairMseF[FIT_PAIRS];\n"
    "    __local double pairNx[FIT_PAIRS];\n"
    "    __local double pairNy[FIT_PAIRS];\n"
    "    __local double pairErrW[FIT_PAIRS];\n"
    "    __local double pairMseW[FIT_PAIRS];\n"
    "    __local double redErr[256];\n"
    "    __local uint redRank[256];\n"
    "    __global uint *out = fitOut + g * FIT_OUT_STRIDE;\n"
    "    if (out[0] != 0u) return;\n"
    "    uint c = fitList[g].x;\n"
    "    uint lo = fitList[g].y;\n"
    "    int sz = (int)desc[c].y;\n"
    "    uint m = maximaOut[g * FIT_MAXIMA_STRIDE];\n"
    "    __global const double *lf = lfps + lo;\n"
    "    if (lid < (int)m) {\n"
    "        uint ii = maximaOut[g * FIT_MAXIMA_STRIDE + 1u + (uint)lid];\n"
    "        midx[lid] = ii;\n"
    "        for (uint q = 0; q < 6u; q++) lfA[lid * 6 + (int)q] = lf[q * lfStride + ii];\n"
    "        for (uint q = 0; q < 6u; q++) lfP[lid * 6 + (int)q] = (ii > 0u) ? lf[q * lfStride + (ii - 1u)] : 0.0;\n"
    "    }\n"
    "    if (lid == 0) for (uint q = 0; q < 6u; q++) lfL[q] = lf[q * lfStride + (uint)(sz - 1)];\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n";

static const char *sourceFitCombosA2 =
    // Every line fit the combo scan can request is one of the C(m,2)
    // forward segments or the C(m,2) wraparound closers (i_high -> i_low);
    // fit them all once — the same fit_line(i0, i1) values the CPU
    // recomputes inside its loop nest — and turn the combo scan into pure
    // table lookups.
    "    uint nPairs = m * (m - 1u) / 2u;\n"
    "    int isFwd = lid < (int)nPairs;\n"
    "    int isWrap = lid >= 128 && lid < 128 + (int)nPairs;\n"
    "    if (isFwd || isWrap) {\n"
    "        uint p = isFwd ? (uint)lid : (uint)(lid - 128);\n"
    "        uint k0 = 0, rem = p;\n"
    "        while (rem >= m - 1u - k0) { rem -= m - 1u - k0; k0++; }\n"
    "        uint k1 = k0 + 1u + rem;\n"
    "        double prm[4], e, ms;\n"
    "        if (isFwd) {\n"
    "            fitLineC(lfA, lfP, lfL, sz, (int)midx[k0], (int)midx[k1], (int)k0, (int)k1, prm, &e, &ms);\n"
    "            pairErrF[p] = e; pairMseF[p] = ms; pairNx[p] = prm[2]; pairNy[p] = prm[3];\n"
    "        } else {\n"
    "            fitLineC(lfA, lfP, lfL, sz, (int)midx[k1], (int)midx[k0], (int)k1, (int)k0, 0, &e, &ms);\n"
    "            pairErrW[p] = e; pairMseW[p] = ms;\n"
    "        }\n"
    "    }\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    uint nc = (m >= 4u) ? (m * (m - 1u) * (m - 2u) * (m - 3u) / 24u) : 0u;\n"
    "    double bestErr = INFINITY;\n"
    "    uint bestRank = 0xFFFFFFFFu;\n"
    "    for (uint rank = (uint)lid; rank < nc; rank += 256u) {\n"
    "        uint a, b, c2, d;\n"
    "        decodeCombo(rank, m, &a, &b, &c2, &d);\n"
    "        uint p01 = pairIdx(a, b, m), p12 = pairIdx(b, c2, m);\n"
    "        uint p23 = pairIdx(c2, d, m), p30 = pairIdx(a, d, m);\n"
    "        if (pairMseF[p01] > maxMse) continue;\n"
    "        if (pairMseF[p12] > maxMse) continue;\n"
    "        double dotv = pairNx[p01] * pairNx[p12] + pairNy[p01] * pairNy[p12];\n"
    "        if (fabs(dotv) > maxDot) continue;\n"
    "        if (pairMseF[p23] > maxMse) continue;\n"
    "        if (pairMseW[p30] > maxMse) continue;\n"
    "        double e = pairErrF[p01] + pairErrF[p12] + pairErrF[p23] + pairErrW[p30];\n"
    "        if (e < bestErr) { bestErr = e; bestRank = rank; }\n"
    "    }\n"
    "    redErr[lid] = bestErr;\n"
    "    redRank[lid] = bestRank;\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    for (int s = 128; s > 0; s >>= 1) {\n"
    "        if (lid < s) {\n"
    "            int take = (redErr[lid + s] < redErr[lid]) ||\n"
    "                       (redErr[lid + s] == redErr[lid] && redRank[lid + s] < redRank[lid]);\n"
    "            if (take) { redErr[lid] = redErr[lid + s]; redRank[lid] = redRank[lid + s]; }\n"
    "        }\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    }\n"
    "    if (lid != 0) return;\n"
    "    if (!(redErr[0] / (double)sz < maxMse)) { out[0] = 2u; return; }\n";

static const char *sourceFitCombosB =
    // Lane 0 continues: the winning combo's four line fits, intersections,
    // float corner narrowing exactly where the CPU assigns quad->p, then the
    // area and angle rejections over those float corners.
    "    uint wa, wb, wc, wd;\n"
    "    decodeCombo(redRank[0], m, &wa, &wb, &wc, &wd);\n"
    "    int idx4[4]; int slot4[4];\n"
    "    idx4[0] = (int)midx[wa]; slot4[0] = (int)wa;\n"
    "    idx4[1] = (int)midx[wb]; slot4[1] = (int)wb;\n"
    "    idx4[2] = (int)midx[wc]; slot4[2] = (int)wc;\n"
    "    idx4[3] = (int)midx[wd]; slot4[3] = (int)wd;\n"
    "    double lines[4][4];\n"
    "    for (int i = 0; i < 4; i++) {\n"
    "        double e, ms;\n"
    "        fitLineC(lfA, lfP, lfL, sz, idx4[i], idx4[(i + 1) & 3], slot4[i], slot4[(i + 1) & 3],\n"
    "                 lines[i], &e, &ms);\n"
    "        if (ms > maxMse) { out[0] = 2u; return; }\n"
    "    }\n"
    "    float pf[8];\n"
    "    for (int i = 0; i < 4; i++) {\n"
    "        double A00 = lines[i][3], A01 = -lines[(i + 1) & 3][3];\n"
    "        double A10 = -lines[i][2], A11 = lines[(i + 1) & 3][2];\n"
    "        double B0 = -lines[i][0] + lines[(i + 1) & 3][0];\n"
    "        double B1 = -lines[i][1] + lines[(i + 1) & 3][1];\n"
    "        double det = A00 * A11 - A10 * A01;\n"
    "        if (fabs(det) < 0.001) { out[0] = 2u; return; }\n"
    "        double W00 = A11 / det, W01 = -A01 / det;\n"
    "        double L0 = W00 * B0 + W01 * B1;\n"
    "        pf[2 * i] = (float)(lines[i][0] + L0 * A00);\n"
    "        pf[2 * i + 1] = (float)(lines[i][1] + L0 * A10);\n"
    "    }\n"
    "    double area = 0;\n"
    "    double l0 = sqrt(sqf(pf[2] - pf[0]) + sqf(pf[3] - pf[1]));\n"
    "    double l1 = sqrt(sqf(pf[4] - pf[2]) + sqf(pf[5] - pf[3]));\n"
    "    double l2 = sqrt(sqf(pf[0] - pf[4]) + sqf(pf[1] - pf[5]));\n"
    "    double p = (l0 + l1 + l2) / 2;\n"
    "    area += sqrt(p * (p - l0) * (p - l1) * (p - l2));\n"
    "    l0 = sqrt(sqf(pf[6] - pf[4]) + sqf(pf[7] - pf[5]));\n"
    "    l1 = sqrt(sqf(pf[0] - pf[6]) + sqf(pf[1] - pf[7]));\n"
    "    l2 = sqrt(sqf(pf[4] - pf[0]) + sqf(pf[5] - pf[1]));\n"
    "    p = (l0 + l1 + l2) / 2;\n"
    "    area += sqrt(p * (p - l0) * (p - l1) * (p - l2));\n"
    "    if (area < 0.95 * tagWidth * tagWidth) { out[0] = 2u; return; }\n"
    "    for (int i = 0; i < 4; i++) {\n"
    "        int j1 = (i + 1) & 3, j2 = (i + 2) & 3;\n"
    "        double dx1 = (double)(pf[2 * j1] - pf[2 * i]);\n"
    "        double dy1 = (double)(pf[2 * j1 + 1] - pf[2 * i + 1]);\n"
    "        double dx2 = (double)(pf[2 * j2] - pf[2 * j1]);\n"
    "        double dy2 = (double)(pf[2 * j2 + 1] - pf[2 * j1 + 1]);\n"
    "        double denom = sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2));\n"
    "        if (denom == 0) { out[0] = 2u; return; }\n"
    "        double cosDt = (dx1 * dx2 + dy1 * dy2) / denom;\n"
    "        if ((cosDt > maxDot || cosDt < -maxDot) || dx1 * dy2 < dy1 * dx2) { out[0] = 2u; return; }\n"
    "    }\n"
    "    out[0] = 1u;\n"
    "    for (int i = 0; i < 8; i++) out[1 + i] = as_uint(pf[i]);\n"
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
// 16 build tasks x at most OCL_HASH_SIZE/2 clusters each.
#define OCL_MAX_CLUSTERS (1u << 19)

// Mirrors the flag defines in sourceFitPrep/sourceFitSort — keep in sync.
#define FIT_PROCESSED (1u << 0)
#define FIT_SKIP_MINPIX (1u << 1)
#define FIT_SKIP_PERIM (1u << 2)
#define FIT_SKIP_AREA (1u << 3)
#define FIT_SKIP_BORDER (1u << 4)
#define FIT_REVERSED (1u << 5)
#define FIT_SORT_GLOBAL (1u << 6)
#define FIT_SKIP_TOOBIG (1u << 7)
#define FIT_SORTED (1u << 8)
#define FIT_SORT_SLM (1u << 9)
// Passed to the fit program as build options.
#define FIT_SLM_CAP 512
#define FIT_BIG_CAP 32768
// Scratch slices (and so workgroups) per fitSortBig launch.
#define FIT_BATCH 256
// P3 chain limits, also passed to the fit program as build options.
// FIT_MAX_K bounds max_nmaxima; detectors configured above it fall back to
// the CPU fit. The strides are in uints.
#define FIT_MAX_K 16
#define FIT_OUT_STRIDE 12
#define FIT_MAXIMA_STRIDE 18
// Upper bound on summed fit-cluster points per frame; clusters past the cap
// fall back to the CPU fit (bounds lfps/errs scratch at ~256 MB).
#define FIT_POINT_CAP (4u * 1024u * 1024u)
// fitOut status values.
#define FIT_QUAD_PENDING 0u
#define FIT_QUAD_ACCEPT 1u
#define FIT_QUAD_REJECT 2u

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
static cl_kernel oclKernelGatherRecords;
static int oclFitReady = 0;
static cl_kernel oclKernelFitPrep;
static cl_kernel oclKernelFitSortSlm;
static cl_kernel oclKernelFitSortBig;
static cl_kernel oclKernelFitLfps;
static cl_kernel oclKernelFitErrs;
static cl_kernel oclKernelFitCombos;

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
    cl_mem bufPerm;
    cl_mem bufClusterDesc;
    cl_mem bufSortKeys;
    cl_mem bufFitMeta;
    cl_mem bufSortScratch;
    cl_mem bufSortList;
    cl_mem bufMaximaScratch;
    // P3 chain buffers, grow-only, sized by the frame's fit-cluster load
    // rather than the frame geometry.
    cl_mem bufFitList;
    size_t fitListCap;
    cl_mem bufLfps;
    size_t lfpsCap;
    cl_mem bufErrsRaw;
    size_t errsRawCap;
    cl_mem bufErrsSmooth;
    size_t errsSmoothCap;
    cl_mem bufMaxima;
    size_t maximaCap;
    cl_mem bufFitOut;
    size_t fitOutCap;
} OclBufferCache;

static OclBufferCache cache;

// One-shot handoff from runClusterChain (which enqueues the gather + fit
// preparation) to oclFitQuads (called later from fit_quads): valid only when
// the decimated grayscale this frame's lfps must sample is still resident in
// bufIm, i.e. on the oclFrontend path. Guarded by oclMutex; every other
// entry point invalidates it.
typedef struct {
    int valid;
    // The walk ran in slim mode: the handed-off clusters are size-only
    // shells (data == NULL) and any cluster the GPU fit does not decide
    // must be materialized from the gathered records before the CPU can
    // touch it.
    int slim;
    const zarray_t *clusters;
    cl_int cw, ch, cs;
} PendingFit;

static PendingFit pendingFit;
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

static double hostNowUs(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static void profHost(const char *name, double startUs)
{
    if (profEnabled)
        fprintf(stderr, "  host %-11s %8.1f us\n", name, hostNowUs() - startUs);
}

// The fit kernels need fp64 (for fit_quad's double-evaluated center) and
// correctly-rounded fp32 division (for bit-identical slopes), so they live
// in their own program: a device without either degrades the fit path only,
// never the frontend.
static void oclInitFitProgram(cl_device_id device)
{
    cl_device_fp_config doubleConfig = 0;
    clGetDeviceInfo(device, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(doubleConfig), &doubleConfig, NULL);
    if (doubleConfig == 0) {
        oclDebugLog("no fp64: fit kernels disabled");
        return;
    }
    cl_device_fp_config singleConfig = 0;
    clGetDeviceInfo(device, CL_DEVICE_SINGLE_FP_CONFIG, sizeof(singleConfig), &singleConfig, NULL);
    const int exactDivide = (singleConfig & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT) != 0;
    if (!exactDivide)
        oclDebugLog("no correctly-rounded fp32 divide: slopes may differ in the last ulp");

    // quad_segment_maxima's low-pass kernel, computed with the exact CPU
    // expressions (sigma = 1, cutoff = 0.05) and this process's libm — the
    // same values the CPU path computes at runtime — then baked into the
    // program as exact hex float literals.
    const double sigma = 1.0, cutoff = 0.05;
    int fsz = sqrt(-log(cutoff) * 2 * sigma * sigma) + 1;
    fsz = 2 * fsz + 1;
    if (fsz != 7) {
        oclDebugLog("unexpected smoothing kernel size: fit kernels disabled");
        return;
    }
    float filt[7];
    for (int i = 0; i < 7; i++) {
        int j = i - fsz / 2;
        filt[i] = exp(-j * j / (2 * sigma * sigma));
    }

    char options[640];
    snprintf(options, sizeof(options),
             "%s -DFIT_SLM_CAP=%d -DFIT_BIG_CAP=%d -DFIT_MAX_K=%d -DFIT_OUT_STRIDE=%d"
             " -DFIT_MAXIMA_STRIDE=%d -DFILT0=%af -DFILT1=%af -DFILT2=%af -DFILT3=%af"
             " -DFILT4=%af -DFILT5=%af -DFILT6=%af",
             exactDivide ? "-cl-fp32-correctly-rounded-divide-sqrt" : "", FIT_SLM_CAP, FIT_BIG_CAP,
             FIT_MAX_K, FIT_OUT_STRIDE, FIT_MAXIMA_STRIDE,
             (double)filt[0], (double)filt[1], (double)filt[2], (double)filt[3],
             (double)filt[4], (double)filt[5], (double)filt[6]);

    cl_int err = CL_SUCCESS;
    const char *sources[12] = { sourceFitPrep, sourceFitSortHelpers, sourceFitSortHelpers2, sourceFitPrep2,
                                sourceFitSortSlm, sourceFitSort, sourceFitLfps, sourceFitLine, sourceFitErrs,
                                sourceFitCombosA, sourceFitCombosA2, sourceFitCombosB };
    cl_program program = clCreateProgramWithSource(oclContext, 12, sources, NULL, &err);
    if (err != CL_SUCCESS)
        return;
    err = clBuildProgram(program, 1, &device, options, NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[8192] = { 0 };
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log) - 1, log, NULL);
        oclDebugLog(log);
        clReleaseProgram(program);
        return;
    }
    struct { cl_kernel *handle; const char *name; } fitKernels[] = {
        { &oclKernelFitPrep, "fitPrep" },
        { &oclKernelFitSortSlm, "fitSortSlm" },
        { &oclKernelFitSortBig, "fitSortBig" },
        { &oclKernelFitLfps, "fitLfps" },
        { &oclKernelFitErrs, "fitErrs" },
        { &oclKernelFitCombos, "fitCombos" },
    };
    int failed = 0;
    for (size_t i = 0; i < sizeof(fitKernels) / sizeof(fitKernels[0]); i++) {
        *fitKernels[i].handle = clCreateKernel(program, fitKernels[i].name, &err);
        if (err != CL_SUCCESS)
            failed = 1;
    }
    clReleaseProgram(program);
    if (!failed)
        oclFitReady = 1;
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

    const char *sources[8] = { sourceThreshold, sourceCcl, sourceCompress, sourceExtract, sourceEmit, sourceSort, sourceScan, sourceGather };
    cl_program program = clCreateProgramWithSource(oclContext, 8, sources, NULL, &err);
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
        { &oclKernelGatherRecords, "gatherRecords" },
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
    oclInitFitProgram(device);
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
    releaseBuffer(cache.bufPerm);
    releaseBuffer(cache.bufClusterDesc);
    releaseBuffer(cache.bufSortKeys);
    releaseBuffer(cache.bufFitMeta);
    releaseBuffer(cache.bufSortScratch);
    releaseBuffer(cache.bufSortList);
    releaseBuffer(cache.bufMaximaScratch);
    releaseBuffer(cache.bufFitList);
    releaseBuffer(cache.bufLfps);
    releaseBuffer(cache.bufErrsRaw);
    releaseBuffer(cache.bufErrsSmooth);
    releaseBuffer(cache.bufMaxima);
    releaseBuffer(cache.bufFitOut);
    memset(&cache, 0, sizeof(cache));
    pendingFit.valid = 0;
}

static cl_mem createOrFail(cl_mem_flags flags, size_t bytes, void *host, int *failed)
{
    cl_int err = CL_SUCCESS;
    cl_mem buffer = clCreateBuffer(oclContext, flags, bytes, host, &err);
    if (err != CL_SUCCESS)
        *failed = 1;
    return buffer;
}

// Grow-only allocation for the P3 chain scratch: kept across frames, grown
// with headroom when a frame needs more.
static int ensureChainBuffer(cl_mem *buffer, size_t *capacity, cl_mem_flags flags, size_t bytes)
{
    if (*buffer != NULL && *capacity >= bytes)
        return 1;
    releaseBuffer(*buffer);
    *buffer = NULL;
    *capacity = 0;
    int failed = 0;
    const size_t grown = bytes + bytes / 4;
    cl_mem created = createOrFail(flags, grown, NULL, &failed);
    if (failed) {
        releaseBuffer(created);
        return 0;
    }
    *buffer = created;
    *capacity = grown;
    return 1;
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
    cache.bufPerm = createOrFail(CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, (size_t)OCL_RECORD_CAPACITY * 4, NULL, &failed);
    cache.bufClusterDesc = createOrFail(CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, (size_t)OCL_MAX_CLUSTERS * 8, NULL, &failed);
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

static void destroyClusterList(zarray_t *clusters)
{
    for (int i = 0; i < zarray_size(clusters); i++) {
        zarray_t *cluster;
        zarray_get(clusters, i, &cluster);
        zarray_destroy(cluster);
    }
    zarray_destroy(clusters);
}

// The slim build walk hands the GPU fit size-only cluster shells. Any
// cluster the GPU does not decide must get its point data back before the
// CPU fit may touch it: the gathered records are cluster-contiguous in the
// CPU emitter's point order and their payloads carry exactly the struct pt
// fields, so a shell rebuilds from its descriptor range. skip[c] != 0
// keeps that cluster a shell (already decided); NULL materializes every
// shell. On failure the remaining shells are emptied (size 0) so the CPU
// path skips them instead of dereferencing NULL data. Caller holds
// oclMutex.
static int materializeShells(zarray_t *clusters, const uint8_t *skip)
{
    const uint32_t clusterCount = (uint32_t)zarray_size(clusters);
    if (clusterCount == 0)
        return 1;

    cl_int err = CL_SUCCESS;
    int ok = 1;
    const uint32_t *desc = clEnqueueMapBuffer(oclQueue, cache.bufClusterDesc, CL_TRUE, CL_MAP_READ, 0,
                                              (size_t)clusterCount * 8, 0, NULL, NULL, &err);
    if (err != CL_SUCCESS)
        desc = NULL;
    const uint64_t *gathered = NULL;
    if (desc != NULL) {
        const uint32_t total = desc[2 * (clusterCount - 1)] + desc[2 * (clusterCount - 1) + 1];
        gathered = clEnqueueMapBuffer(oclQueue, cache.bufRecordsAlt, CL_TRUE, CL_MAP_READ, 0,
                                      (size_t)total * 16, 0, NULL, NULL, &err);
        if (err != CL_SUCCESS)
            gathered = NULL;
    }

    for (uint32_t c = 0; c < clusterCount; c++) {
        zarray_t *cluster;
        zarray_get(clusters, (int)c, &cluster);
        if (cluster->data != NULL || cluster->size == 0)
            continue;
        if (skip != NULL && skip[c] != 0)
            continue;
        char *data = NULL;
        if (gathered != NULL && desc[2 * c + 1] == (uint32_t)cluster->size)
            data = malloc((size_t)cluster->size * cluster->el_sz);
        if (data == NULL) {
            cluster->size = 0;
            ok = 0;
            continue;
        }
        const uint32_t off = desc[2 * c];
        for (int j = 0; j < cluster->size; j++) {
            const uint64_t payload = gathered[2 * ((size_t)off + (size_t)j) + 1];
            OclPt *pt = (OclPt *)(data + (size_t)j * cluster->el_sz);
            pt->x = (uint16_t)(payload >> 48);
            pt->y = (uint16_t)(payload >> 32);
            pt->gx = (int16_t)(uint16_t)(payload >> 16);
            pt->gy = (int16_t)(uint16_t)payload;
            pt->slope = 0.0f;
        }
        cluster->data = data;
        cluster->alloc = cluster->size;
    }

    if (gathered != NULL)
        clEnqueueUnmapMemObject(oclQueue, cache.bufRecordsAlt, (void *)gathered, 0, NULL, NULL);
    if (desc != NULL)
        clEnqueueUnmapMemObject(oclQueue, cache.bufClusterDesc, (void *)desc, 0, NULL, NULL);
    if (!ok)
        oclDebugLog("shell materialize failed: dropped unmaterialized clusters");
    return ok;
}

// A pending slim handoff still owes its clusters their point data (they
// are alive — the owning detect call has not reached fit_quads yet);
// materialize before this entry point's work invalidates the device
// buffers the shells depend on.
static void flushPendingFit(void)
{
    if (pendingFit.valid && pendingFit.slim)
        materializeShells((zarray_t *)pendingFit.clusters, NULL);
    pendingFit.valid = 0;
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
    flushPendingFit();
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

// Bookkeeping that turns the build walk's grouping into the gather
// permutation as a counting sort, with no per-cluster index storage:
// the walk records each record's task-local cluster index (one flat
// uint32 per record); the merge records, per task-local cluster, the
// final cluster index and the chunk's start offset within that final
// cluster (task-order concatenation); once final cluster offsets are
// known, a parallel pass computes each record's output slot directly.
typedef struct {
    uint32_t recStart, recEnd;
    int localCount;
    uint32_t *finalIdx;
    uint32_t *chunkStart;
} GatherTaskPlan;

typedef struct {
    int taskCount;
    const uint32_t *recCluster;
    GatherTaskPlan tasks[16];
} GatherPlan;

typedef struct {
    const uint64_t *records;
    uint32_t recStart, recEnd;
    zarray_t *clusters;
    // Per-record task-local cluster index slots (the whole flat array,
    // indexed by absolute record index). NULL when no permutation is
    // requested.
    uint32_t *recCluster;
    uint64_t *clusterKeys;
    int clusterCap;
    // Count-only walk for the GPU fit path: clusters keep their exact
    // sizes but no point data (the GPU consumes the gathered records
    // instead; stragglers are materialized from them on demand).
    int slim;
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
        if (task->recCluster != NULL)
            task->recCluster[i] = table[slot].clusterIdx;
        zarray_t *cluster;
        zarray_get(task->clusters, (int)table[slot].clusterIdx, &cluster);
        if (task->slim)
            cluster->size++;
        else
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

static void destroyGatherPlan(GatherPlan *plan)
{
    for (int t = 0; t < plan->taskCount; t++) {
        free(plan->tasks[t].finalIdx);
        free(plan->tasks[t].chunkStart);
    }
    plan->taskCount = 0;
}

static void concatAndDestroy(zarray_t *dst, zarray_t *src)
{
    zarray_ensure_capacity(dst, dst->size + src->size);
    memcpy(dst->data + (size_t)dst->size * dst->el_sz, src->data,
           (size_t)src->size * src->el_sz);
    dst->size += src->size;
    zarray_destroy(src);
}

// Merge per-task clusters in task order: tasks cover ascending row ranges,
// so concatenation preserves raster point order within each cluster. When
// a gather plan is requested, the merge also records where each task-local
// cluster lands: its final cluster index and its chunk's start offset
// within that final cluster.
static zarray_t *mergeTaskClusters(BuildTask *tasks, int taskCount, GatherPlan *plan, int slim)
{
    zarray_t *clusters = zarray_create(sizeof(zarray_t *));
    HashEntry *table = calloc(OCL_HASH_SIZE, sizeof(HashEntry));
    if (table == NULL) {
        for (int t = 0; t < taskCount; t++)
            destroyTaskClusters(&tasks[t]);
        if (plan != NULL)
            plan->taskCount = 0;
        return clusters;
    }

    for (int t = 0; t < taskCount; t++) {
        const int localCount = zarray_size(tasks[t].clusters);
        GatherTaskPlan *taskPlan = NULL;
        if (plan != NULL) {
            taskPlan = &plan->tasks[t];
            taskPlan->recStart = tasks[t].recStart;
            taskPlan->recEnd = tasks[t].recEnd;
            taskPlan->localCount = localCount;
            taskPlan->finalIdx = malloc(sizeof(uint32_t) * (size_t)(localCount > 0 ? localCount : 1));
            taskPlan->chunkStart = malloc(sizeof(uint32_t) * (size_t)(localCount > 0 ? localCount : 1));
        }
        for (int i = 0; i < localCount; i++) {
            zarray_t *cluster;
            zarray_get(tasks[t].clusters, i, &cluster);
            uint64_t key = tasks[t].clusterKeys[i];
            uint32_t slot = hashSlot(key);
            while (table[slot].key != 0 && table[slot].key != key)
                slot = (slot + 1) & (OCL_HASH_SIZE - 1);
            if (table[slot].key == 0) {
                table[slot].key = key;
                table[slot].clusterIdx = (uint32_t)zarray_size(clusters);
                if (taskPlan != NULL) {
                    taskPlan->finalIdx[i] = table[slot].clusterIdx;
                    taskPlan->chunkStart[i] = 0;
                }
                zarray_add(clusters, &cluster);
            } else {
                zarray_t *dst;
                zarray_get(clusters, (int)table[slot].clusterIdx, &dst);
                if (taskPlan != NULL) {
                    taskPlan->finalIdx[i] = table[slot].clusterIdx;
                    taskPlan->chunkStart[i] = (uint32_t)dst->size;
                }
                if (slim) {
                    dst->size += cluster->size;
                    zarray_destroy(cluster);
                } else {
                    concatAndDestroy(dst, cluster);
                }
            }
        }
        zarray_destroy(tasks[t].clusters);
        free(tasks[t].clusterKeys);
    }
    free(table);
    if (plan != NULL)
        plan->taskCount = taskCount;
    return clusters;
}

static uint32_t *recClusterScratch = NULL;
static uint32_t recClusterScratchCap = 0;
static uint32_t *sortListScratch = NULL;
static uint32_t sortListScratchCap = 0;

static zarray_t *buildClusters(apriltag_detector_t *td, const uint64_t *records, uint32_t recordCount,
                               int segsPerRow, cl_int h, GatherPlan *planOut, int slim)
{
    int taskCount = (td->wp != NULL && td->nthreads > 1) ? td->nthreads : 1;
    if (taskCount > 16)
        taskCount = 16;
    BuildTask tasks[16];

    uint32_t *recCluster = NULL;
    if (planOut != NULL) {
        planOut->taskCount = 0;
        if (recClusterScratchCap < recordCount) {
            free(recClusterScratch);
            recClusterScratch = malloc(sizeof(uint32_t) * (size_t)recordCount);
            recClusterScratchCap = (recClusterScratch != NULL) ? recordCount : 0;
        }
        recCluster = recClusterScratch;
        planOut->recCluster = recCluster;
    }

    // Split rows into contiguous ranges balanced by record count; row r's
    // records start at segOffsetsHost[r * segsPerRow].
    uint32_t targetPerTask = recordCount / (uint32_t)taskCount + 1;
    cl_int row = 0;
    for (int t = 0; t < taskCount; t++) {
        uint32_t recStart = (row < h) ? segOffsetsHost[(size_t)row * segsPerRow] : recordCount;
        tasks[t].records = records;
        tasks[t].recStart = recStart;
        tasks[t].clusters = zarray_create(sizeof(zarray_t *));
        tasks[t].recCluster = recCluster;
        tasks[t].clusterCap = 256;
        tasks[t].clusterKeys = malloc(sizeof(uint64_t) * tasks[t].clusterCap);
        tasks[t].slim = slim;
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
    return mergeTaskClusters(tasks, taskCount, (recCluster != NULL) ? planOut : NULL, slim);
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
    const GatherTaskPlan *taskPlan;
    const uint32_t *recCluster;
    const uint32_t *finalOffsets;
    uint32_t *perm;
    int failed;
} PermTask;

// Each record's output slot: its chunk's absolute start (final cluster
// offset + chunk offset within the cluster) plus its rank within the
// chunk, which ascending record order provides for free. Tasks own
// disjoint record ranges and disjoint output chunks.
static void doPermTask(void *p)
{
    PermTask *task = (PermTask *)p;
    const GatherTaskPlan *taskPlan = task->taskPlan;
    const int localCount = taskPlan->localCount > 0 ? taskPlan->localCount : 1;
    uint32_t *cursor = malloc(sizeof(uint32_t) * (size_t)localCount);
    if (cursor == NULL) {
        task->failed = 1;
        return;
    }
    for (int c = 0; c < taskPlan->localCount; c++)
        cursor[c] = task->finalOffsets[taskPlan->finalIdx[c]] + taskPlan->chunkStart[c];
    for (uint32_t i = taskPlan->recStart; i < taskPlan->recEnd; i++)
        task->perm[cursor[task->recCluster[i]]++] = i;
    free(cursor);
}

// P1b of the GPU fit_quads port: turn the build walk's grouping into a
// permutation array plus per-cluster (offset, count) descriptors in the
// mapped staging buffers, then gather the records into cluster-contiguous
// order in cache.bufRecordsAlt — the layout the GPU fit stages consume.
// Caller holds oclMutex. Returns 0 on failure.
static int gatherClusterRecords(apriltag_detector_t *td, zarray_t *clusters, GatherPlan *plan,
                                uint32_t recordCount)
{
    const uint32_t clusterCount = (uint32_t)zarray_size(clusters);
    if (clusterCount == 0 || clusterCount > OCL_MAX_CLUSTERS || plan->taskCount == 0)
        return 0;
    uint32_t *finalOffsets = malloc(sizeof(uint32_t) * (size_t)clusterCount);
    if (finalOffsets == NULL)
        return 0;

    cl_int err = CL_SUCCESS;
    double t = hostNowUs();
    uint32_t *perm = clEnqueueMapBuffer(oclQueue, cache.bufPerm, CL_TRUE,
                                        CL_MAP_WRITE_INVALIDATE_REGION, 0,
                                        (size_t)recordCount * 4, 0, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        free(finalOffsets);
        return 0;
    }
    uint32_t *desc = clEnqueueMapBuffer(oclQueue, cache.bufClusterDesc, CL_TRUE,
                                        CL_MAP_WRITE_INVALIDATE_REGION, 0,
                                        (size_t)clusterCount * 8, 0, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        clEnqueueUnmapMemObject(oclQueue, cache.bufPerm, perm, 0, NULL, NULL);
        free(finalOffsets);
        return 0;
    }
    profHost("mapPerm", t);

    t = hostNowUs();
    uint32_t slot = 0;
    for (uint32_t c = 0; c < clusterCount; c++) {
        zarray_t *cluster;
        zarray_get(clusters, (int)c, &cluster);
        finalOffsets[c] = slot;
        desc[2 * c] = slot;
        desc[2 * c + 1] = (uint32_t)cluster->size;
        slot += (uint32_t)cluster->size;
    }
    int complete = slot == recordCount;
    profHost("descFill", t);

    t = hostNowUs();
    PermTask permTasks[16];
    if (complete) {
        for (int pt = 0; pt < plan->taskCount; pt++) {
            permTasks[pt].taskPlan = &plan->tasks[pt];
            permTasks[pt].recCluster = plan->recCluster;
            permTasks[pt].finalOffsets = finalOffsets;
            permTasks[pt].perm = perm;
            permTasks[pt].failed = 0;
        }
        if (plan->taskCount == 1) {
            doPermTask(&permTasks[0]);
        } else {
            for (int pt = 0; pt < plan->taskCount; pt++)
                workerpool_add_task(td->wp, doPermTask, &permTasks[pt]);
            workerpool_run(td->wp);
        }
        for (int pt = 0; pt < plan->taskCount; pt++)
            complete &= permTasks[pt].failed == 0;
    }
    free(finalOffsets);
    profHost("permPass", t);

    t = hostNowUs();
    err = clEnqueueUnmapMemObject(oclQueue, cache.bufPerm, perm, 0, NULL, NULL);
    err |= clEnqueueUnmapMemObject(oclQueue, cache.bufClusterDesc, desc, 0, NULL, NULL);
    if (err != CL_SUCCESS || !complete)
        return 0;
    profHost("unmapPerm", t);

    err |= clSetKernelArg(oclKernelGatherRecords, 0, sizeof(cl_mem), &cache.bufRecords);
    err |= clSetKernelArg(oclKernelGatherRecords, 1, sizeof(cl_mem), &cache.bufPerm);
    err |= clSetKernelArg(oclKernelGatherRecords, 2, sizeof(cl_uint), &recordCount);
    err |= clSetKernelArg(oclKernelGatherRecords, 3, sizeof(cl_mem), &cache.bufRecordsAlt);
    if (err != CL_SUCCESS)
        return 0;
    t = hostNowUs();
    const size_t gatherGlobal[1] = { roundUp(recordCount, 256) };
    err = clEnqueueNDRangeKernel(oclQueue, oclKernelGatherRecords, 1, NULL, gatherGlobal, NULL,
                                 0, NULL, profSlot("gather"));
    if (err != CL_SUCCESS)
        return 0;
    profHost("gatherEnq", t);
    if (profEnabled)
        clFinish(oclQueue);
    return 1;
}

// Development gate for the gather path: read the gathered records back and
// check each cluster's range carries a uniform key and reproduces the
// cluster's points in order. Reports on stderr.
static void validateGather(zarray_t *clusters, uint32_t recordCount)
{
    cl_int err = CL_SUCCESS;
    const uint64_t *gathered = clEnqueueMapBuffer(oclQueue, cache.bufRecordsAlt, CL_TRUE, CL_MAP_READ,
                                                  0, (size_t)recordCount * 16, 0, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "apriltag opencl: gather validate: map failed\n");
        return;
    }

    uint64_t mismatches = 0;
    uint32_t slot = 0;
    for (int c = 0; c < zarray_size(clusters); c++) {
        zarray_t *cluster;
        zarray_get(clusters, c, &cluster);
        const uint64_t clusterKey = gathered[2 * slot];
        for (int j = 0; j < zarray_size(cluster); j++, slot++) {
            const OclPt *pt = (const OclPt *)(cluster->data + (size_t)j * cluster->el_sz);
            const uint64_t key = gathered[2 * slot];
            const uint64_t payload = gathered[2 * slot + 1];
            const int ok = key == clusterKey &&
                           pt->x == (uint16_t)(payload >> 48) &&
                           pt->y == (uint16_t)(payload >> 32) &&
                           pt->gx == (int16_t)(uint16_t)(payload >> 16) &&
                           pt->gy == (int16_t)(uint16_t)payload;
            if (!ok)
                mismatches++;
        }
    }
    clEnqueueUnmapMemObject(oclQueue, cache.bufRecordsAlt, (void *)gathered, 0, NULL, NULL);

    if (mismatches == 0 && slot == recordCount)
        fprintf(stderr, "apriltag opencl: gather validate: PASS (%d clusters, %u records)\n",
                zarray_size(clusters), recordCount);
    else
        fprintf(stderr, "apriltag opencl: gather validate: FAIL (%llu mismatches, %u/%u records)\n",
                (unsigned long long)mismatches, slot, recordCount);
}

static int ensureFitBuffers(void)
{
    if (cache.bufSortKeys != NULL)
        return 1;
    int failed = 0;
    cache.bufSortKeys = createOrFail(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, (size_t)OCL_RECORD_CAPACITY * 8, NULL, &failed);
    cache.bufFitMeta = createOrFail(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, (size_t)OCL_MAX_CLUSTERS * 32, NULL, &failed);
    cache.bufSortScratch = createOrFail(CL_MEM_READ_WRITE, (size_t)FIT_BATCH * FIT_BIG_CAP * 8, NULL, &failed);
    cache.bufSortList = createOrFail(CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, (size_t)OCL_MAX_CLUSTERS * 4, NULL, &failed);
    cache.bufMaximaScratch = createOrFail(CL_MEM_READ_WRITE, (size_t)OCL_RECORD_CAPACITY * 4, NULL, &failed);
    if (failed) {
        releaseBuffer(cache.bufSortKeys);
        releaseBuffer(cache.bufFitMeta);
        releaseBuffer(cache.bufSortScratch);
        releaseBuffer(cache.bufSortList);
        releaseBuffer(cache.bufMaximaScratch);
        cache.bufSortKeys = NULL;
        cache.bufFitMeta = NULL;
        cache.bufSortScratch = NULL;
        cache.bufSortList = NULL;
        cache.bufMaximaScratch = NULL;
        return 0;
    }
    return 1;
}

typedef struct {
    cl_int minClusterPixels, perimCap, tagWidth, normalAllowed, reversedAllowed;
} FitParams;

// Replicates fit_quads' per-call parameters (apriltag_quad_thresh.c)
// expression for expression, including the int-by-float decimate division.
static void computeFitParams(apriltag_detector_t *td, cl_int cw, cl_int ch, FitParams *params)
{
    int normalAllowed = 0, reversedAllowed = 0;
    int minTagWidth = 1000000;
    for (int i = 0; i < zarray_size(td->tag_families); i++) {
        apriltag_family_t *family;
        zarray_get(td->tag_families, i, &family);
        if (family->width_at_border < minTagWidth)
            minTagWidth = family->width_at_border;
        normalAllowed |= !family->reversed_border;
        reversedAllowed |= family->reversed_border;
    }
    if (td->quad_decimate > 1)
        minTagWidth /= td->quad_decimate;
    if (minTagWidth < 3)
        minTagWidth = 3;
    params->minClusterPixels = td->qtp.min_cluster_pixels;
    params->perimCap = 2 * (2 * cw + 2 * ch);
    params->tagWidth = minTagWidth;
    params->normalAllowed = normalAllowed;
    params->reversedAllowed = reversedAllowed;
}

// P2 of the GPU fit_quads port: enqueues the per-cluster preparation
// (filter cascade, bbox, center, slopes, gradient dot) and the slope sort
// over the gathered cluster-contiguous records. SLM-sized clusters sort
// inside fitPrep; bigger ones get one fitSortBig launch each through the
// shared scratch buffer. Caller holds oclMutex. Returns 0 on failure.
static int fitPrepSort(apriltag_detector_t *td, zarray_t *clusters, cl_int cw, cl_int ch)
{
    const cl_uint clusterCount = (cl_uint)zarray_size(clusters);
    if (oclFitReady == 0 || clusterCount == 0 || !ensureFitBuffers())
        return 0;

    FitParams params;
    computeFitParams(td, cw, ch, &params);

    // The sort runs over host-prefiltered id lists: SLM-sized candidates in
    // sortList[0..slmCount), oversized ones after them. Only host-checkable
    // size filters apply here; the sort kernels skip area- and
    // border-rejected ids via the meta flags fitPrep writes. The list is
    // staged in a persistent host scratch and uploaded with a non-blocking
    // write — a blocking map here would stall the host behind the gather
    // kernel on the in-order queue. The scratch stays untouched until the
    // next fitPrepSort call, by which time every entry point has issued a
    // blocking map that drained this write.
    if (sortListScratchCap < clusterCount) {
        free(sortListScratch);
        sortListScratch = malloc(sizeof(uint32_t) * (size_t)clusterCount);
        sortListScratchCap = (sortListScratch != NULL) ? clusterCount : 0;
        if (sortListScratch == NULL)
            return 0;
    }
    uint32_t *sortList = sortListScratch;
    cl_uint slmCount = 0, bigCount = 0;
    for (cl_uint c = 0; c < clusterCount; c++) {
        zarray_t *cluster;
        zarray_get(clusters, (int)c, &cluster);
        const int clusterSize = cluster->size;
        if (clusterSize >= params.minClusterPixels && clusterSize <= FIT_SLM_CAP)
            sortList[slmCount++] = c;
    }
    for (cl_uint c = 0; c < clusterCount; c++) {
        zarray_t *cluster;
        zarray_get(clusters, (int)c, &cluster);
        const int clusterSize = cluster->size;
        if (clusterSize > FIT_SLM_CAP && clusterSize <= FIT_BIG_CAP && clusterSize <= params.perimCap)
            sortList[slmCount + bigCount++] = c;
    }
    cl_int err = CL_SUCCESS;
    if (slmCount + bigCount > 0)
        err = clEnqueueWriteBuffer(oclQueue, cache.bufSortList, CL_FALSE, 0,
                                   (size_t)(slmCount + bigCount) * 4, sortList, 0, NULL, NULL);
    if (err != CL_SUCCESS)
        return 0;

    err |= clSetKernelArg(oclKernelFitPrep, 0, sizeof(cl_mem), &cache.bufRecordsAlt);
    err |= clSetKernelArg(oclKernelFitPrep, 1, sizeof(cl_mem), &cache.bufClusterDesc);
    err |= clSetKernelArg(oclKernelFitPrep, 2, sizeof(cl_uint), &clusterCount);
    err |= clSetKernelArg(oclKernelFitPrep, 3, sizeof(cl_int), &params.minClusterPixels);
    err |= clSetKernelArg(oclKernelFitPrep, 4, sizeof(cl_int), &params.perimCap);
    err |= clSetKernelArg(oclKernelFitPrep, 5, sizeof(cl_int), &params.tagWidth);
    err |= clSetKernelArg(oclKernelFitPrep, 6, sizeof(cl_int), &params.normalAllowed);
    err |= clSetKernelArg(oclKernelFitPrep, 7, sizeof(cl_int), &params.reversedAllowed);
    err |= clSetKernelArg(oclKernelFitPrep, 8, sizeof(cl_mem), &cache.bufSortKeys);
    err |= clSetKernelArg(oclKernelFitPrep, 9, sizeof(cl_mem), &cache.bufFitMeta);
    if (err != CL_SUCCESS)
        return 0;
    const size_t prepGlobal[1] = { (size_t)clusterCount * 256 };
    const size_t wgSize[1] = { 256 };
    err = clEnqueueNDRangeKernel(oclQueue, oclKernelFitPrep, 1, NULL, prepGlobal, wgSize, 0, NULL, profSlot("fitPrep"));
    if (err != CL_SUCCESS)
        return 0;

    if (slmCount > 0) {
        err |= clSetKernelArg(oclKernelFitSortSlm, 0, sizeof(cl_mem), &cache.bufSortKeys);
        err |= clSetKernelArg(oclKernelFitSortSlm, 1, sizeof(cl_mem), &cache.bufClusterDesc);
        err |= clSetKernelArg(oclKernelFitSortSlm, 2, sizeof(cl_mem), &cache.bufFitMeta);
        err |= clSetKernelArg(oclKernelFitSortSlm, 3, sizeof(cl_mem), &cache.bufSortList);
        err |= clSetKernelArg(oclKernelFitSortSlm, 4, sizeof(cl_uint), &slmCount);
        const size_t slmGlobal[1] = { (size_t)slmCount * 256 };
        err |= clEnqueueNDRangeKernel(oclQueue, oclKernelFitSortSlm, 1, NULL, slmGlobal, wgSize, 0, NULL, profSlot("fitSortSlm"));
        if (err != CL_SUCCESS)
            return 0;
    }

    err |= clSetKernelArg(oclKernelFitSortBig, 0, sizeof(cl_mem), &cache.bufSortKeys);
    err |= clSetKernelArg(oclKernelFitSortBig, 1, sizeof(cl_mem), &cache.bufClusterDesc);
    err |= clSetKernelArg(oclKernelFitSortBig, 2, sizeof(cl_mem), &cache.bufFitMeta);
    err |= clSetKernelArg(oclKernelFitSortBig, 3, sizeof(cl_mem), &cache.bufSortList);
    err |= clSetKernelArg(oclKernelFitSortBig, 6, sizeof(cl_mem), &cache.bufSortScratch);
    for (cl_uint base = 0; base < bigCount && err == CL_SUCCESS; base += FIT_BATCH) {
        const cl_uint batch = (bigCount - base < FIT_BATCH) ? bigCount - base : FIT_BATCH;
        const cl_uint listBase = slmCount + base;
        const size_t batchGlobal[1] = { (size_t)batch * 256 };
        err |= clSetKernelArg(oclKernelFitSortBig, 4, sizeof(cl_uint), &listBase);
        err |= clSetKernelArg(oclKernelFitSortBig, 5, sizeof(cl_uint), &batch);
        err |= clEnqueueNDRangeKernel(oclQueue, oclKernelFitSortBig, 1, NULL, batchGlobal, wgSize, 0, NULL, profSlot("fitSortBig"));
    }
    return err == CL_SUCCESS;
}

static uint32_t hostOrderedFloatBits(float f)
{
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// fit_quad's slope expression, operation for operation.
static float hostSlope(float fx, float fy, float cx, float cy)
{
    float dx = fx - cx;
    float dy = fy - cy;
    float quadrant = dy > 0 ? (dx > 0 ? 65536.0f : 131072.0f) : (dx > 0 ? 0.0f : -65536.0f);
    if (dy < 0) {
        dy = -dy;
        dx = -dx;
    }
    if (dx < 0) {
        float tmp = dx;
        dx = dy;
        dy = -tmp;
    }
    return quadrant + dy / dx;
}

typedef struct {
    float slope;
    uint32_t idx;
} ValPt;

// Verbatim replica of apriltag_quad_thresh.c's ptsort, including its
// tie-order behaviour (sorting networks below six elements, then a
// right-biased merge), carrying point indices through the sort. The GPU
// sort implements the same network, so its output must match exactly.
static void refPtsort(ValPt *pts, int sz)
{
#define MAYBE_SWAP(arr, apos, bpos) \
    if (arr[apos].slope - arr[bpos].slope > 0) { \
        tmp = arr[apos]; arr[apos] = arr[bpos]; arr[bpos] = tmp; \
    };

    if (sz <= 1)
        return;
    if (sz == 2) {
        ValPt tmp;
        MAYBE_SWAP(pts, 0, 1);
        return;
    }
    if (sz == 3) {
        ValPt tmp;
        MAYBE_SWAP(pts, 0, 1);
        MAYBE_SWAP(pts, 1, 2);
        MAYBE_SWAP(pts, 0, 1);
        return;
    }
    if (sz == 4) {
        ValPt tmp;
        MAYBE_SWAP(pts, 0, 1);
        MAYBE_SWAP(pts, 2, 3);
        MAYBE_SWAP(pts, 0, 2);
        MAYBE_SWAP(pts, 1, 3);
        MAYBE_SWAP(pts, 1, 2);
        return;
    }
    if (sz == 5) {
        ValPt tmp;
        MAYBE_SWAP(pts, 0, 1);
        MAYBE_SWAP(pts, 3, 4);
        MAYBE_SWAP(pts, 1, 2);
        MAYBE_SWAP(pts, 0, 1);
        MAYBE_SWAP(pts, 0, 3);
        MAYBE_SWAP(pts, 2, 4);
        MAYBE_SWAP(pts, 1, 2);
        MAYBE_SWAP(pts, 2, 3);
        MAYBE_SWAP(pts, 1, 2);
        return;
    }
#undef MAYBE_SWAP

    ValPt stackBuffer[256];
    ValPt *tmp = (sz > 256) ? malloc(sizeof(ValPt) * (size_t)sz) : stackBuffer;
    memcpy(tmp, pts, sizeof(ValPt) * (size_t)sz);

    int asz = sz / 2;
    int bsz = sz - asz;
    ValPt *as = &tmp[0];
    ValPt *bs = &tmp[asz];
    refPtsort(as, asz);
    refPtsort(bs, bsz);

    int apos = 0, bpos = 0, outpos = 0;
    while (apos < asz && bpos < bsz) {
        if (as[apos].slope - bs[bpos].slope < 0)
            pts[outpos++] = as[apos++];
        else
            pts[outpos++] = bs[bpos++];
    }
    if (apos < asz)
        memcpy(&pts[outpos], &as[apos], (size_t)(asz - apos) * sizeof(ValPt));
    if (bpos < bsz)
        memcpy(&pts[outpos], &bs[bpos], (size_t)(bsz - bpos) * sizeof(ValPt));
    if (sz > 256)
        free(tmp);
}

typedef struct {
    int skipped, sortedSlm, sortedGlobal, tooBig, tieClusters;
    long tiePoints;
} FitValidateStats;

// Returns NULL when the cluster's GPU outputs replicate the CPU pre-sort
// semantics exactly, else a short description of the first mismatch.
static const char *checkFitCluster(const FitParams *params, const OclPt *pts, uint32_t n,
                                   const uint32_t *m, const uint64_t *gpuKeys, FitValidateStats *stats)
{
    uint32_t expect = FIT_PROCESSED;
    if ((int)n < params->minClusterPixels)
        expect |= FIT_SKIP_MINPIX;
    else if ((int)n > params->perimCap)
        expect |= FIT_SKIP_PERIM;
    if (expect != FIT_PROCESSED) {
        stats->skipped++;
        return (m[0] == expect && m[6] == n) ? NULL : "size filter flags";
    }

    uint16_t xmin = pts[0].x, xmax = pts[0].x, ymin = pts[0].y, ymax = pts[0].y;
    for (uint32_t i = 1; i < n; i++) {
        if (pts[i].x > xmax) xmax = pts[i].x; else if (pts[i].x < xmin) xmin = pts[i].x;
        if (pts[i].y > ymax) ymax = pts[i].y; else if (pts[i].y < ymin) ymin = pts[i].y;
    }
    const uint32_t bboxA = (uint32_t)xmin | ((uint32_t)xmax << 16);
    const uint32_t bboxB = (uint32_t)ymin | ((uint32_t)ymax << 16);
    if ((xmax - xmin) * (ymax - ymin) < params->tagWidth) {
        stats->skipped++;
        if (m[0] != (expect | FIT_SKIP_AREA))
            return "area filter flags";
        return (m[4] == bboxA && m[5] == bboxB && m[6] == n) ? NULL : "bbox";
    }

    float cx = (xmin + xmax) * 0.5 + 0.05118;
    float cy = (ymin + ymax) * 0.5 + -0.028581;
    float dot = 0;
    float *slopes = malloc(sizeof(float) * n);
    if (slopes == NULL)
        return "out of memory";
    for (uint32_t i = 0; i < n; i++) {
        float dx = pts[i].x - cx;
        float dy = pts[i].y - cy;
        dot += dx * pts[i].gx + dy * pts[i].gy;
        slopes[i] = hostSlope(pts[i].x, pts[i].y, cx, cy);
    }

    const int rev = dot < 0;
    uint32_t expectFlags = expect | (rev ? FIT_REVERSED : 0u);
    if (rev ? !params->reversedAllowed : !params->normalAllowed)
        expectFlags |= FIT_SKIP_BORDER;
    else if (n <= FIT_BIG_CAP)
        expectFlags |= FIT_SORTED;
    else
        expectFlags |= FIT_SKIP_TOOBIG;

    uint32_t cxBits, cyBits, dotBits;
    memcpy(&cxBits, &cx, sizeof(cxBits));
    memcpy(&cyBits, &cy, sizeof(cyBits));
    memcpy(&dotBits, &dot, sizeof(dotBits));
    const char *fail = NULL;
    if (m[0] != expectFlags)
        fail = "flags";
    else if (m[1] != cxBits || m[2] != cyBits)
        fail = "center bits";
    else if (m[3] != dotBits)
        fail = "dot bits";
    else if (m[4] != bboxA || m[5] != bboxB || m[6] != n)
        fail = "bbox";

    if (fail == NULL && (expectFlags & FIT_SKIP_BORDER) != 0)
        stats->skipped++;
    if (fail == NULL && (expectFlags & FIT_SKIP_TOOBIG) != 0)
        stats->tooBig++;

    if (fail == NULL && (expectFlags & FIT_SORTED) != 0) {
        if (n <= FIT_SLM_CAP)
            stats->sortedSlm++;
        else
            stats->sortedGlobal++;
        ValPt *ref = malloc(sizeof(ValPt) * n);
        if (ref == NULL) {
            free(slopes);
            return "out of memory";
        }
        for (uint32_t i = 0; i < n; i++) {
            ref[i].slope = slopes[i];
            ref[i].idx = i;
        }
        refPtsort(ref, (int)n);
        long ties = 0;
        for (uint32_t i = 0; i < n && fail == NULL; i++) {
            const uint64_t expectKey = ((uint64_t)hostOrderedFloatBits(ref[i].slope) << 32) | ref[i].idx;
            if (gpuKeys[i] != expectKey)
                fail = "sorted order vs ptsort";
            if (i > 0 && ref[i].slope == ref[i - 1].slope)
                ties++;
        }
        if (fail == NULL && ties > 0) {
            stats->tieClusters++;
            stats->tiePoints += ties;
        }
        free(ref);
    }
    free(slopes);
    return fail;
}

// Development gate for the fit preparation: reads the meta and sorted-key
// buffers back and checks every cluster against a host replication of the
// CPU's pre-sort semantics. Reports on stderr.
static void validateFitPrep(apriltag_detector_t *td, zarray_t *clusters, uint32_t recordCount,
                            cl_int cw, cl_int ch)
{
    FitParams params;
    computeFitParams(td, cw, ch, &params);
    const uint32_t clusterCount = (uint32_t)zarray_size(clusters);

    cl_int err = CL_SUCCESS;
    const uint64_t *keys = clEnqueueMapBuffer(oclQueue, cache.bufSortKeys, CL_TRUE, CL_MAP_READ, 0,
                                              (size_t)recordCount * 8, 0, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "apriltag opencl: fit validate: keys map failed\n");
        return;
    }
    const uint32_t *meta = clEnqueueMapBuffer(oclQueue, cache.bufFitMeta, CL_TRUE, CL_MAP_READ, 0,
                                              (size_t)clusterCount * 32, 0, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        clEnqueueUnmapMemObject(oclQueue, cache.bufSortKeys, (void *)keys, 0, NULL, NULL);
        fprintf(stderr, "apriltag opencl: fit validate: meta map failed\n");
        return;
    }

    FitValidateStats stats = { 0, 0, 0, 0, 0, 0 };
    uint64_t failures = 0;
    uint32_t off = 0;
    for (uint32_t c = 0; c < clusterCount; c++) {
        zarray_t *cluster;
        zarray_get(clusters, (int)c, &cluster);
        const uint32_t n = (uint32_t)cluster->size;
        const char *fail = checkFitCluster(&params, (const OclPt *)cluster->data, n,
                                           meta + 8u * c, keys + off, &stats);
        off += n;
        if (fail != NULL) {
            if (failures < 5)
                fprintf(stderr, "apriltag opencl: fit validate: cluster %u (n=%u): %s\n", c, n, fail);
            failures++;
        }
    }

    clEnqueueUnmapMemObject(oclQueue, cache.bufSortKeys, (void *)keys, 0, NULL, NULL);
    clEnqueueUnmapMemObject(oclQueue, cache.bufFitMeta, (void *)meta, 0, NULL, NULL);

    if (failures == 0)
        fprintf(stderr,
                "apriltag opencl: fit validate: PASS (%u clusters: %d skipped, %d slm-sorted, "
                "%d global-sorted, %d too-big; ties %d clusters / %ld pts, order == ptsort)\n",
                clusterCount, stats.skipped, stats.sortedSlm, stats.sortedGlobal, stats.tooBig,
                stats.tieClusters, stats.tiePoints);
    else
        fprintf(stderr, "apriltag opencl: fit validate: FAIL (%llu clusters mismatched)\n",
                (unsigned long long)failures);
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
static zarray_t *runClusterChain(apriltag_detector_t *td, cl_mem inputBuffer, cl_int cw, cl_int ch, cl_int cs, int labelsReady, int grayOnDevice)
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
    // preserved by sort stability). Superseded by the gather path below;
    // kept as validation scaffolding.
    int useSorted = getenv("APRILTAG_OPENCL_SORTED") != NULL;
    // P2: per-cluster fit preparation and slope sort over the gathered
    // records (implies the gather path).
    int useFit = !useSorted && getenv("APRILTAG_OPENCL_FIT") != NULL;
    // P1b: the hash build walk emits a permutation so one GPU gather
    // materializes cluster-contiguous records on-device for the fit stages.
    // Gated until the GPU fit lands; mutually exclusive with the sorted
    // path, whose cluster order differs from the walk's encounter order.
    int useGather = !useSorted && (useFit || getenv("APRILTAG_OPENCL_GATHER") != NULL);
    // P4: when the GPU fit will consume this frame (and no validation pass
    // needs host-side point data), the walk runs slim — it discovers the
    // grouping and counts but copies no points; the GPU fit reads the
    // gathered records and any cluster left to the CPU is materialized
    // from them. Conditions mirror oclFitQuads' acceptance checks so a
    // slim handoff cannot be turned down for a knowable reason.
    int slimWalk = useFit && grayOnDevice && oclFitReady != 0 &&
                   td->qtp.max_nmaxima >= 0 && td->qtp.max_nmaxima <= FIT_MAX_K &&
                   getenv("APRILTAG_OPENCL_FIT_VALIDATE") == NULL &&
                   getenv("APRILTAG_OPENCL_GATHER_VALIDATE") == NULL;
    if (useSorted && !sortRecords(recordCount))
        goto done;

    double t = hostNowUs();
    void *mapped = clEnqueueMapBuffer(oclQueue, cache.bufRecords, CL_TRUE, CL_MAP_READ, 0,
                                      (size_t)recordCount * 16, 0, NULL, profSlot("mapRecords"), &err);
    if (err != CL_SUCCESS)
        goto done;
    profHost("mapWait", t);
    t = hostNowUs();
    GatherPlan plan = { 0 };
    if (useSorted)
        clusters = buildClustersSorted(td, (const uint64_t *)mapped, recordCount);
    else
        clusters = buildClusters(td, (const uint64_t *)mapped, recordCount, segsPerRow, ch,
                                 useGather ? &plan : NULL, slimWalk);
    profHost("buildWalk", t);
    t = hostNowUs();
    clEnqueueUnmapMemObject(oclQueue, cache.bufRecords, mapped, 0, NULL, NULL);
    profHost("unmapRecords", t);

    int fitArmed = 0;
    if (useGather && clusters != NULL && plan.taskCount > 0) {
        if (gatherClusterRecords(td, clusters, &plan, recordCount)) {
            if (getenv("APRILTAG_OPENCL_GATHER_VALIDATE") != NULL)
                validateGather(clusters, recordCount);
            if (useFit) {
                t = hostNowUs();
                if (fitPrepSort(td, clusters, cw, ch)) {
                    profHost("fitEnqueue", t);
                    // The frontend profPrint needs the prep/sort events
                    // complete; outside the fitEnqueue stamp so the stamp
                    // reflects the real (non-profiled) enqueue cost.
                    if (profEnabled)
                        clFinish(oclQueue);
                    if (getenv("APRILTAG_OPENCL_FIT_VALIDATE") != NULL)
                        validateFitPrep(td, clusters, recordCount, cw, ch);
                    // The P3 chain (oclFitQuads) samples the decimated
                    // grayscale for lfps weights; hand off only when bufIm
                    // still holds it.
                    if (grayOnDevice) {
                        pendingFit.valid = 1;
                        pendingFit.slim = slimWalk;
                        pendingFit.clusters = clusters;
                        pendingFit.cw = cw;
                        pendingFit.ch = ch;
                        pendingFit.cs = cs;
                        fitArmed = 1;
                    }
                } else {
                    oclDebugLog("fit prep failed");
                }
            }
        } else {
            oclDebugLog("gather failed");
        }
    }
    destroyGatherPlan(&plan);

    // A slim walk whose fit handoff did not arm leaves shells nothing can
    // fill: rebuild the clusters in full from the still-intact record
    // buffer (failure path; never taken in steady state).
    if (slimWalk && !fitArmed && clusters != NULL) {
        oclDebugLog("slim walk discarded: rebuilding full clusters");
        destroyClusterList(clusters);
        clusters = NULL;
        mapped = clEnqueueMapBuffer(oclQueue, cache.bufRecords, CL_TRUE, CL_MAP_READ, 0,
                                    (size_t)recordCount * 16, 0, NULL, NULL, &err);
        if (err == CL_SUCCESS) {
            clusters = buildClusters(td, (const uint64_t *)mapped, recordCount, segsPerRow, ch,
                                     NULL, 0);
            clEnqueueUnmapMemObject(oclQueue, cache.bufRecords, mapped, 0, NULL, NULL);
        }
    }

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
    flushPendingFit();
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
        clusters = runClusterChain(td, inputBuffer, w, h, ts, labelsReady, 0);
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
    flushPendingFit();
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

    clusters = runClusterChain(td, cache.bufOut, w, h, s, 1, 1);

done:
    profPrint();
    pthread_mutex_unlock(&oclMutex);
    if (clusters == NULL)
        oclDebugLog("GPU frontend failed, falling back to CPU");
    return clusters;
}

// CPU reference for the P3 validation gate (apriltag_quad_thresh.c).
int fit_quad(apriltag_detector_t *td, image_u8_t *im, zarray_t *cluster, struct quad *quad,
             int tag_width, bool normal_border, bool reversed_border);

typedef struct {
    uint32_t clusterIdx;
    uint32_t lfpsOffset;
    uint8_t reversed;
} FitSlot;

// Development gate for the GPU quad fit: every GPU-fitted cluster is re-fit
// with the production CPU fit_quad (on a copy — fit_quad sorts its input)
// and the verdict plus the corner bits must match exactly.
static void validateFitQuads(apriltag_detector_t *td, zarray_t *clusters, image_u8_t *im,
                             const FitSlot *slots, uint32_t fitCount, const uint32_t *outBuf)
{
    FitParams params;
    computeFitParams(td, (cl_int)im->width, (cl_int)im->height, &params);

    uint32_t failures = 0, accepted = 0, rejected = 0;
    for (uint32_t i = 0; i < fitCount; i++) {
        zarray_t *cluster;
        zarray_get(clusters, (int)slots[i].clusterIdx, &cluster);
        zarray_t *copy = zarray_create(cluster->el_sz);
        zarray_ensure_capacity(copy, cluster->size);
        memcpy(copy->data, cluster->data, (size_t)cluster->size * cluster->el_sz);
        copy->size = cluster->size;
        struct quad ref;
        memset(&ref, 0, sizeof(ref));
        const int res = fit_quad(td, im, copy, &ref, params.tagWidth,
                                 params.normalAllowed != 0, params.reversedAllowed != 0);
        zarray_destroy(copy);

        const uint32_t *o = outBuf + (size_t)i * FIT_OUT_STRIDE;
        const char *fail = NULL;
        if (o[0] == FIT_QUAD_ACCEPT) {
            accepted++;
            if (res != 1)
                fail = "GPU accepted, CPU rejected";
            else if (memcmp(ref.p, o + 1, sizeof(float) * 8) != 0)
                fail = "corner bits differ";
            else if ((slots[i].reversed != 0) != ref.reversed_border)
                fail = "reversed flag differs";
        } else if (o[0] == FIT_QUAD_REJECT) {
            rejected++;
            if (res != 0)
                fail = "GPU rejected, CPU accepted";
        } else {
            fail = "no GPU verdict";
        }
        if (fail != NULL) {
            if (failures < 5)
                fprintf(stderr, "apriltag opencl: fit quads validate: cluster %u (n=%d): %s\n",
                        slots[i].clusterIdx, cluster->size, fail);
            failures++;
        }
    }
    if (failures == 0)
        fprintf(stderr,
                "apriltag opencl: fit quads validate: PASS (%u fits: %u quads, %u rejected, corner bits exact)\n",
                fitCount, accepted, rejected);
    else
        fprintf(stderr, "apriltag opencl: fit quads validate: FAIL (%u of %u mismatched)\n",
                failures, fitCount);
}

uint8_t *oclFitQuads(apriltag_detector_t *td, zarray_t *clusters, image_u8_t *im, zarray_t *quads)
{
    if (clusters == NULL)
        return NULL;

    // Every check happens under the lock so a rejected (or mismatched)
    // handoff is flushed rather than left pending — a slim handoff owes
    // its shells point data before anyone returns to the CPU path.
    pthread_mutex_lock(&oclMutex);
    const uint32_t clusterCount = (uint32_t)zarray_size(clusters);
    if (!pendingFit.valid || pendingFit.clusters != clusters || oclFitReady == 0 ||
        clusterCount == 0 || getenv("APRILTAG_OPENCL") == NULL ||
        td->qtp.max_nmaxima < 0 || td->qtp.max_nmaxima > FIT_MAX_K ||
        pendingFit.cw != im->width || pendingFit.ch != im->height ||
        pendingFit.cs != im->stride) {
        flushPendingFit();
        pthread_mutex_unlock(&oclMutex);
        return NULL;
    }
    pendingFit.valid = 0;
    const int slim = pendingFit.slim;
    const cl_int imW = pendingFit.cw, imH = pendingFit.ch, imS = pendingFit.cs;
    profReset();

    uint8_t *handled = NULL;
    FitSlot *slots = NULL;

    // Block on the P2 preparation and sorts, then split the clusters into
    // GPU-fit slots and the outcomes the fitPrep flags already decide: the
    // pre-fit filters (size, perimeter, bbox area, border direction) reject
    // exactly the clusters the CPU path would reject before/inside fit_quad.
    double t = hostNowUs();
    cl_int err = CL_SUCCESS;
    const uint32_t *meta = clEnqueueMapBuffer(oclQueue, cache.bufFitMeta, CL_TRUE, CL_MAP_READ, 0,
                                              (size_t)clusterCount * 32, 0, NULL, NULL, &err);
    if (err != CL_SUCCESS)
        goto fail;
    profHost("metaWait", t);

    t = hostNowUs();
    handled = calloc(clusterCount, 1);
    slots = malloc(sizeof(FitSlot) * (size_t)clusterCount);
    int ok = handled != NULL && slots != NULL;
    uint32_t fitCount = 0;
    uint32_t fitPoints = 0;
    const uint32_t skipMask = FIT_SKIP_MINPIX | FIT_SKIP_PERIM | FIT_SKIP_AREA | FIT_SKIP_BORDER;
    for (uint32_t c = 0; ok && c < clusterCount; c++) {
        const uint32_t flags = meta[8u * c];
        if ((flags & FIT_PROCESSED) == 0)
            continue;
        if ((flags & skipMask) != 0) {
            handled[c] = 1;
            continue;
        }
        if ((flags & FIT_SORTED) == 0)
            continue;
        zarray_t *cluster;
        zarray_get(clusters, (int)c, &cluster);
        const uint32_t n = (uint32_t)cluster->size;
        if (fitPoints + n > FIT_POINT_CAP)
            continue;
        slots[fitCount].clusterIdx = c;
        slots[fitCount].lfpsOffset = fitPoints;
        slots[fitCount].reversed = (flags & FIT_REVERSED) != 0;
        fitCount++;
        fitPoints += n;
    }
    err = clEnqueueUnmapMemObject(oclQueue, cache.bufFitMeta, (void *)meta, 0, NULL, NULL);
    profHost("fitSplit", t);
    if (!ok || err != CL_SUCCESS)
        goto fail;
    if (fitCount == 0)
        goto finish;

    if (!ensureChainBuffer(&cache.bufFitList, &cache.fitListCap,
                           CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, (size_t)fitCount * 8) ||
        !ensureChainBuffer(&cache.bufLfps, &cache.lfpsCap, CL_MEM_READ_WRITE, (size_t)fitPoints * 48) ||
        !ensureChainBuffer(&cache.bufErrsRaw, &cache.errsRawCap, CL_MEM_READ_WRITE, (size_t)fitPoints * 8) ||
        !ensureChainBuffer(&cache.bufErrsSmooth, &cache.errsSmoothCap, CL_MEM_READ_WRITE, (size_t)fitPoints * 8) ||
        !ensureChainBuffer(&cache.bufMaxima, &cache.maximaCap, CL_MEM_READ_WRITE,
                           (size_t)fitCount * FIT_MAXIMA_STRIDE * 4) ||
        !ensureChainBuffer(&cache.bufFitOut, &cache.fitOutCap,
                           CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, (size_t)fitCount * FIT_OUT_STRIDE * 4))
        goto fail;

    t = hostNowUs();
    {
        uint32_t *list = clEnqueueMapBuffer(oclQueue, cache.bufFitList, CL_TRUE,
                                            CL_MAP_WRITE_INVALIDATE_REGION, 0,
                                            (size_t)fitCount * 8, 0, NULL, NULL, &err);
        if (err != CL_SUCCESS)
            goto fail;
        for (uint32_t i = 0; i < fitCount; i++) {
            list[2 * i] = slots[i].clusterIdx;
            list[2 * i + 1] = slots[i].lfpsOffset;
        }
        err = clEnqueueUnmapMemObject(oclQueue, cache.bufFitList, list, 0, NULL, NULL);
    }
    const uint32_t zero = 0;
    err |= clEnqueueFillBuffer(oclQueue, cache.bufFitOut, &zero, 4, 0,
                               (size_t)fitCount * FIT_OUT_STRIDE * 4, 0, NULL, NULL);
    if (err != CL_SUCCESS)
        goto fail;
    profHost("fitListUpload", t);

    t = hostNowUs();
    {
        FitParams params;
        computeFitParams(td, imW, imH, &params);
        const cl_uint count = fitCount;
        const cl_int maxNmaxima = td->qtp.max_nmaxima;
        const cl_double maxDot = (cl_double)td->qtp.cos_critical_rad;
        const cl_double maxMse = (cl_double)td->qtp.max_line_fit_mse;

        const cl_uint lfStride = fitPoints;
        err |= clSetKernelArg(oclKernelFitLfps, 0, sizeof(cl_mem), &cache.bufRecordsAlt);
        err |= clSetKernelArg(oclKernelFitLfps, 1, sizeof(cl_mem), &cache.bufSortKeys);
        err |= clSetKernelArg(oclKernelFitLfps, 2, sizeof(cl_mem), &cache.bufClusterDesc);
        err |= clSetKernelArg(oclKernelFitLfps, 3, sizeof(cl_mem), &cache.bufFitList);
        err |= clSetKernelArg(oclKernelFitLfps, 4, sizeof(cl_uint), &count);
        err |= clSetKernelArg(oclKernelFitLfps, 5, sizeof(cl_mem), &cache.bufIm);
        err |= clSetKernelArg(oclKernelFitLfps, 6, sizeof(cl_int), &imW);
        err |= clSetKernelArg(oclKernelFitLfps, 7, sizeof(cl_int), &imH);
        err |= clSetKernelArg(oclKernelFitLfps, 8, sizeof(cl_int), &imS);
        err |= clSetKernelArg(oclKernelFitLfps, 9, sizeof(cl_uint), &lfStride);
        err |= clSetKernelArg(oclKernelFitLfps, 10, sizeof(cl_mem), &cache.bufLfps);

        err |= clSetKernelArg(oclKernelFitErrs, 0, sizeof(cl_mem), &cache.bufLfps);
        err |= clSetKernelArg(oclKernelFitErrs, 1, sizeof(cl_uint), &lfStride);
        err |= clSetKernelArg(oclKernelFitErrs, 2, sizeof(cl_mem), &cache.bufClusterDesc);
        err |= clSetKernelArg(oclKernelFitErrs, 3, sizeof(cl_mem), &cache.bufFitList);
        err |= clSetKernelArg(oclKernelFitErrs, 4, sizeof(cl_uint), &count);
        err |= clSetKernelArg(oclKernelFitErrs, 5, sizeof(cl_int), &maxNmaxima);
        err |= clSetKernelArg(oclKernelFitErrs, 6, sizeof(cl_mem), &cache.bufErrsRaw);
        err |= clSetKernelArg(oclKernelFitErrs, 7, sizeof(cl_mem), &cache.bufErrsSmooth);
        err |= clSetKernelArg(oclKernelFitErrs, 8, sizeof(cl_mem), &cache.bufMaximaScratch);
        err |= clSetKernelArg(oclKernelFitErrs, 9, sizeof(cl_mem), &cache.bufMaxima);
        err |= clSetKernelArg(oclKernelFitErrs, 10, sizeof(cl_mem), &cache.bufFitOut);

        err |= clSetKernelArg(oclKernelFitCombos, 0, sizeof(cl_mem), &cache.bufLfps);
        err |= clSetKernelArg(oclKernelFitCombos, 1, sizeof(cl_uint), &lfStride);
        err |= clSetKernelArg(oclKernelFitCombos, 2, sizeof(cl_mem), &cache.bufClusterDesc);
        err |= clSetKernelArg(oclKernelFitCombos, 3, sizeof(cl_mem), &cache.bufFitList);
        err |= clSetKernelArg(oclKernelFitCombos, 4, sizeof(cl_uint), &count);
        err |= clSetKernelArg(oclKernelFitCombos, 5, sizeof(cl_mem), &cache.bufMaxima);
        err |= clSetKernelArg(oclKernelFitCombos, 6, sizeof(cl_double), &maxDot);
        err |= clSetKernelArg(oclKernelFitCombos, 7, sizeof(cl_double), &maxMse);
        err |= clSetKernelArg(oclKernelFitCombos, 8, sizeof(cl_int), &params.tagWidth);
        err |= clSetKernelArg(oclKernelFitCombos, 9, sizeof(cl_mem), &cache.bufFitOut);
        if (err != CL_SUCCESS)
            goto fail;

        const size_t fitGlobal[1] = { (size_t)fitCount * 256 };
        const size_t fitLocal[1] = { 256 };
        err |= clEnqueueNDRangeKernel(oclQueue, oclKernelFitLfps, 1, NULL, fitGlobal, fitLocal, 0, NULL, profSlot("fitLfps"));
        err |= clEnqueueNDRangeKernel(oclQueue, oclKernelFitErrs, 1, NULL, fitGlobal, fitLocal, 0, NULL, profSlot("fitErrs"));
        err |= clEnqueueNDRangeKernel(oclQueue, oclKernelFitCombos, 1, NULL, fitGlobal, fitLocal, 0, NULL, profSlot("fitCombos"));
        if (err != CL_SUCCESS)
            goto fail;
    }
    profHost("fitChainEnqueue", t);

    t = hostNowUs();
    {
        const uint32_t *outBuf = clEnqueueMapBuffer(oclQueue, cache.bufFitOut, CL_TRUE, CL_MAP_READ, 0,
                                                    (size_t)fitCount * FIT_OUT_STRIDE * 4, 0, NULL, NULL, &err);
        if (err != CL_SUCCESS)
            goto fail;
        profHost("quadWait", t);

        t = hostNowUs();
        for (uint32_t i = 0; i < fitCount; i++) {
            const uint32_t *o = outBuf + (size_t)i * FIT_OUT_STRIDE;
            if (o[0] == FIT_QUAD_ACCEPT) {
                struct quad quad;
                memset(&quad, 0, sizeof(quad));
                memcpy(quad.p, o + 1, sizeof(float) * 8);
                quad.reversed_border = slots[i].reversed != 0;
                zarray_add(quads, &quad);
                handled[slots[i].clusterIdx] = 1;
            } else if (o[0] == FIT_QUAD_REJECT) {
                handled[slots[i].clusterIdx] = 1;
            }
            // FIT_QUAD_PENDING: the chain skipped it — CPU fallback.
        }
        profHost("quadBuild", t);
        if (!slim && getenv("APRILTAG_OPENCL_FIT_VALIDATE") != NULL)
            validateFitQuads(td, clusters, im, slots, fitCount, outBuf);
        clEnqueueUnmapMemObject(oclQueue, cache.bufFitOut, (void *)outBuf, 0, NULL, NULL);
    }

finish:
    // Slim shells the GPU did not decide are about to meet the CPU fit:
    // give them their points back from the gathered records.
    if (slim) {
        uint32_t fallbacks = 0;
        for (uint32_t c = 0; c < clusterCount; c++)
            fallbacks += handled[c] == 0;
        if (fallbacks > 0) {
            t = hostNowUs();
            materializeShells(clusters, handled);
            profHost("materialize", t);
        }
    }
    free(slots);
    profPrint();
    pthread_mutex_unlock(&oclMutex);
    return handled;

fail:
    oclDebugLog("GPU fit quads failed, falling back to CPU");
    if (slim)
        materializeShells(clusters, NULL);
    free(slots);
    free(handled);
    profPrint();
    pthread_mutex_unlock(&oclMutex);
    return NULL;
}
