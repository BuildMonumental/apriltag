/* OpenCL (Intel iGPU) acceleration for the AprilTag detector front-end.
 * See apriltag_opencl.h for the contract.
 *
 * Memory discipline: every buffer shared between CPU and GPU is an
 * Intel USM host allocation (clHostMemAllocINTEL). On integrated
 * graphics these are plain DRAM pages mapped into both address spaces,
 * so kernels read/write them directly and the CPU consumes the results
 * with no copies and no map/unmap round trips.
 */
#define CL_TARGET_OPENCL_VERSION 300

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <CL/cl.h>
#include <CL/cl_ext.h>

#include "apriltag_opencl.h"

// struct row_run is mirrored in the kernel source; both sides must be
// {u16,u16,u8} padded to 6 bytes
_Static_assert(sizeof(struct row_run) == 6, "row_run layout drifted");

struct at_ocl
{
    cl_platform_id platform;
    cl_device_id dev;
    cl_context ctx;
    cl_command_queue q;
    cl_program prog;

    cl_kernel k_minmax;
    cl_kernel k_blur;
    cl_kernel k_thresh_rle;
    cl_kernel k_rle_scan;
    cl_kernel k_row_prefix;
    cl_kernel k_rle_emit;

    clHostMemAllocINTEL_fn usm_alloc;
    clDeviceMemAllocINTEL_fn usm_dev_alloc;
    clEnqueueMemcpyINTEL_fn usm_memcpy;
    clMemFreeINTEL_fn usm_free;
    clSetKernelArgMemPointerINTEL_fn set_arg_ptr;

    bool prof; // APRILTAG_OCL_PROF=1: print per-kernel GPU times

    // USM host buffers, grown on demand
    uint8_t *img;        size_t img_cap;
    uint8_t *threshim;   size_t threshim_cap;
    uint8_t *tiles;      size_t tiles_cap;   // im_max, im_min, blurred max, blurred min
    uint32_t *row_off;   size_t row_off_cap; // h+1 entries
    uint32_t *win_cnt;   size_t win_cnt_cap; // nw*h run-start counts/offsets
    uint32_t *first_chg; size_t first_chg_cap; // nw*h first-change positions
    struct row_run *runs; size_t runs_cap;   // bytes

    image_u8_t thim; // returned threshold image; buf points into threshim
};

static const char *KSRC =
"typedef struct { ushort start; ushort end; uchar v; } run_t;\n"
"\n"
"kernel void tile_minmax(global const uchar *im, int s, int tw, int th,\n"
"                        global uchar *im_max, global uchar *im_min)\n"
"{\n"
"    int tx = get_global_id(0), ty = get_global_id(1);\n"
"    if (tx >= tw || ty >= th) return;\n"
"    uchar4 r0 = vload4(0, im + (ty*4+0)*s + tx*4);\n"
"    uchar4 r1 = vload4(0, im + (ty*4+1)*s + tx*4);\n"
"    uchar4 r2 = vload4(0, im + (ty*4+2)*s + tx*4);\n"
"    uchar4 r3 = vload4(0, im + (ty*4+3)*s + tx*4);\n"
"    uchar4 mx4 = max(max(r0, r1), max(r2, r3));\n"
"    uchar4 mn4 = min(min(r0, r1), min(r2, r3));\n"
"    im_max[ty*tw+tx] = max(max(mx4.s0, mx4.s1), max(mx4.s2, mx4.s3));\n"
"    im_min[ty*tw+tx] = min(min(mn4.s0, mn4.s1), min(mn4.s2, mn4.s3));\n"
"}\n"
"\n"
"kernel void tile_blur(global const uchar *im_max, global const uchar *im_min,\n"
"                      int tw, int th,\n"
"                      global uchar *bmax, global uchar *bmin)\n"
"{\n"
"    int tx = get_global_id(0), ty = get_global_id(1);\n"
"    if (tx >= tw || ty >= th) return;\n"
"    uchar mx = 0, mn = 255;\n"
"    for (int dy = -1; dy <= 1; dy++) {\n"
"        int y = ty + dy;\n"
"        if (y < 0 || y >= th) continue;\n"
"        for (int dx = -1; dx <= 1; dx++) {\n"
"            int x = tx + dx;\n"
"            if (x < 0 || x >= tw) continue;\n"
"            mx = max(mx, im_max[y*tw+x]);\n"
"            mn = min(mn, im_min[y*tw+x]);\n"
"        }\n"
"    }\n"
"    bmax[ty*tw+tx] = mx;\n"
"    bmin[ty*tw+tx] = mn;\n"
"}\n"
"\n"
"/* The threshold value of one pixel: full tiles compare against their\n"
" * blurred tile min/max and mark low-contrast tiles 127; right-edge\n"
" * columns and bottom-tail rows clamp to the last full tile and never\n"
" * produce 127, exactly like the CPU fixups. */\n"
"inline uchar thresh_px(global const uchar *im, int s, int x, int y,\n"
"                       int tw, int th, int min_wbd,\n"
"                       global const uchar *bmax, global const uchar *bmin)\n"
"{\n"
"    int tx = x >> 2, ty = y >> 2;\n"
"    int full = (tx < tw) & (ty < th);\n"
"    int ctx = min(tx, tw-1), cty = min(ty, th-1);\n"
"    int mn = bmin[cty*tw+ctx], mx = bmax[cty*tw+ctx];\n"
"    if (full && (mx - mn < min_wbd))\n"
"        return 127;\n"
"    int t = mn + (mx - mn) / 2;\n"
"    return (im[y*s+x] > t) ? (uchar)255 : (uchar)0;\n"
"}\n"
"\n"
"/* Threshold + run-length encoding, window-parallel: each work-item\n"
" * owns WSZ pixels of one row. It writes the thresholded bytes and, in\n"
" * the same pass, counts run starts (a run starts at x if\n"
" * (x == 0 || row[x] != row[x-1]) && row[x] != 127, x in [0, w-2]) and\n"
" * records the first value-change position in the window, which lets\n"
" * the emit pass find run ends without long forward scans. */\n"
"#define WSZ 32\n"
"#define NOCHG 0xffffffffu\n"
"\n"
"/* movemask: bit i = MSB of byte i (comparison results are 0xff/0x00) */\n"
"inline uint mask16_bits(char16 m)\n"
"{\n"
"    ulong2 u = as_ulong2(m);\n"
"    uint lo = (uint)(((u.s0 & 0x8080808080808080UL) * 0x0002040810204081UL) >> 56);\n"
"    uint hi = (uint)(((u.s1 & 0x8080808080808080UL) * 0x0002040810204081UL) >> 56);\n"
"    return lo | (hi << 8);\n"
"}\n"
"\n"
"/* shift a 32-byte window right by one, injecting prev at lane 0 */\n"
"inline void shift1(uchar16 blo, uchar16 bhi, uchar prev,\n"
"                   uchar16 *shlo, uchar16 *shhi)\n"
"{\n"
"    const uchar16 m = (uchar16)(16,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14);\n"
"    *shlo = shuffle2(blo, (uchar16)(prev), m);\n"
"    *shhi = shuffle2(bhi, (uchar16)(blo.sf), m);\n"
"}\n"
"\n"
"kernel void thresh_rle(global const uchar *im, global uchar *out,\n"
"                       int w, int h, int s, int tw, int th, int min_wbd,\n"
"                       global const uchar *bmax, global const uchar *bmin,\n"
"                       int nw, global uint *win_cnt, global uint *first_chg)\n"
"{\n"
"    int wi = get_global_id(0), y = get_global_id(1);\n"
"    if (wi >= nw || y >= h) return;\n"
"    int x0 = wi * WSZ;\n"
"    int x1 = min(x0 + WSZ - 1, w - 1);\n"
"    int xmax = w - 2;\n"
"    uchar prev = 0;\n"
"    if (x0 > 0)\n"
"        prev = thresh_px(im, s, x0-1, y, tw, th, min_wbd, bmax, bmin);\n"
"    uint count = 0;\n"
"    uint fc = NOCHG;\n"
"    int ty = y >> 2;\n"
"    if (x1 == x0 + WSZ - 1 && ty < th && (x0 >> 2) + 8 <= tw) {\n"
"        // fast path: full window over full tiles = exactly 8 tiles.\n"
"        // Block loads + vector compare; same integer math as thresh_px.\n"
"        int tbase = ty*tw + (x0 >> 2);\n"
"        uchar8 mn8 = vload8(0, bmin + tbase);\n"
"        uchar8 mx8 = vload8(0, bmax + tbase);\n"
"        uchar8 diff8 = mx8 - mn8;\n"
"        uchar8 t8 = mn8 + (diff8 >> (uchar8)1);\n"
"        const uchar16 elo = (uchar16)(0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3);\n"
"        const uchar16 ehi = (uchar16)(4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7);\n"
"        uchar16 ilo = vload16(0, im + y*s + x0);\n"
"        uchar16 ihi = vload16(0, im + y*s + x0 + 16);\n"
"        uchar16 lc_lo = as_uchar16(shuffle(diff8, elo) < (uchar16)(min_wbd));\n"
"        uchar16 lc_hi = as_uchar16(shuffle(diff8, ehi) < (uchar16)(min_wbd));\n"
"        uchar16 vlo = select(as_uchar16(ilo > shuffle(t8, elo)) & (uchar16)255,\n"
"                             (uchar16)127, lc_lo);\n"
"        uchar16 vhi = select(as_uchar16(ihi > shuffle(t8, ehi)) & (uchar16)255,\n"
"                             (uchar16)127, lc_hi);\n"
"        vstore16(vlo, 0, out + y*s + x0);\n"
"        vstore16(vhi, 0, out + y*s + x0 + 16);\n"
"        // change/127 masks; bit i = pixel x0+i\n"
"        uchar16 shlo, shhi;\n"
"        shift1(vlo, vhi, prev, &shlo, &shhi);\n"
"        uint chg = mask16_bits(vlo != shlo) | (mask16_bits(vhi != shhi) << 16);\n"
"        uint is127 = mask16_bits(vlo == (uchar16)127) |\n"
"                     (mask16_bits(vhi == (uchar16)127) << 16);\n"
"        // pixels past xmax don't participate in runs\n"
"        uint valid = (x0 + WSZ - 1 <= xmax) ? 0xffffffffu : 0x7fffffffu;\n"
"        chg &= valid;\n"
"        uint starts = chg & ~is127;\n"
"        if (x0 == 0) {\n"
"            chg &= ~1u; // a change is only defined for x >= 1\n"
"            starts = (starts & ~1u) | (~is127 & 1u);\n"
"        }\n"
"        count = popcount(starts);\n"
"        fc = chg ? (uint)(x0 + ctz(chg)) : NOCHG;\n"
"    } else {\n"
"        for (int x = x0; x <= x1; x++) {\n"
"            uchar val = thresh_px(im, s, x, y, tw, th, min_wbd, bmax, bmin);\n"
"            out[y*s+x] = val;\n"
"            if (x == 0) {\n"
"                count += val != 127;\n"
"            } else if (x <= xmax && val != prev) {\n"
"                if (fc == NOCHG) fc = x;\n"
"                count += val != 127;\n"
"            }\n"
"            prev = val;\n"
"        }\n"
"    }\n"
"    win_cnt[y*nw + wi] = count;\n"
"    first_chg[y*nw + wi] = fc;\n"
"}\n"
"\n"
"/* In-place exclusive scan of each row's window counts; the row total\n"
" * lands in row_off[y+1] (row_off[0] is zeroed; row_prefix turns the\n"
" * totals into global offsets). */\n"
"kernel void rle_scan_rows(global uint *win_cnt, int h, int nw,\n"
"                          global uint *row_off)\n"
"{\n"
"    int y = get_global_id(0);\n"
"    if (y >= h) return;\n"
"    global uint *c = win_cnt + y*nw;\n"
"    uint acc = 0;\n"
"    for (int i = 0; i < nw; i++) {\n"
"        uint v = c[i];\n"
"        c[i] = acc;\n"
"        acc += v;\n"
"    }\n"
"    row_off[y+1] = acc;\n"
"    if (y == 0) row_off[0] = 0;\n"
"}\n"
"\n"
"/* Single-workgroup inclusive scan of row_off[1..h]: per-row run counts\n"
" * become global run offsets. */\n"
"kernel void row_prefix(global uint *row_off, int h)\n"
"{\n"
"    local uint buf[2][256];\n"
"    int lid = get_local_id(0);\n"
"    uint carry = 0;\n"
"    for (int base = 1; base <= h; base += 256) {\n"
"        int i = base + lid;\n"
"        buf[0][lid] = (i <= h) ? row_off[i] : 0;\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"        int src = 0;\n"
"        for (int d = 1; d < 256; d <<= 1) {\n"
"            uint t = buf[src][lid];\n"
"            if (lid >= d) t += buf[src][lid - d];\n"
"            buf[1-src][lid] = t;\n"
"            barrier(CLK_LOCAL_MEM_FENCE);\n"
"            src = 1 - src;\n"
"        }\n"
"        if (i <= h) row_off[i] = buf[src][lid] + carry;\n"
"        carry += buf[src][255];\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"    }\n"
"}\n"
"\n"
"/* Emits each window's runs at runs[row_off[y] + win_cnt[y][wi]]. A\n"
" * run's end is the position before the next value change; changes\n"
" * past the window come from first_chg, so nobody walks a long run\n"
" * byte by byte. Output is identical to the CPU rle_row. */\n"
"kernel void rle_emit(global const uchar *im, int w, int h, int s, int nw,\n"
"                     global const uint *win_cnt, global const uint *first_chg,\n"
"                     global const uint *row_off, global run_t *runs)\n"
"{\n"
"    int wi = get_global_id(0), y = get_global_id(1);\n"
"    if (wi >= nw || y >= h) return;\n"
"    int xmax = w - 2;\n"
"    int x0 = wi * WSZ;\n"
"    if (x0 > xmax) return;\n"
"    global const uchar *row = im + y*s;\n"
"    global run_t *out = runs + row_off[y] + win_cnt[y*nw + wi];\n"
"    int x1 = min(x0 + WSZ - 1, xmax);\n"
"    uchar prev = (x0 > 0) ? row[x0 - 1] : (uchar)0;\n"
"    // window bytes as bitmasks: two block loads, register-only logic\n"
"    uint chg, is127, white, valid;\n"
"    if (x0 + WSZ - 1 <= xmax + 1) {\n"
"        uchar16 blo = vload16(0, row + x0);\n"
"        uchar16 bhi = vload16(0, row + x0 + 16);\n"
"        uchar16 shlo, shhi;\n"
"        shift1(blo, bhi, prev, &shlo, &shhi);\n"
"        chg = mask16_bits(blo != shlo) | (mask16_bits(bhi != shhi) << 16);\n"
"        is127 = mask16_bits(blo == (uchar16)127) |\n"
"                (mask16_bits(bhi == (uchar16)127) << 16);\n"
"        white = mask16_bits(blo == (uchar16)255) |\n"
"                (mask16_bits(bhi == (uchar16)255) << 16);\n"
"        valid = (x0 + WSZ - 1 <= xmax) ? 0xffffffffu : 0x7fffffffu;\n"
"    } else {\n"
"        chg = is127 = white = valid = 0;\n"
"        for (int x = x0; x <= x1; x++) {\n"
"            uchar c = row[x];\n"
"            uint bit = 1u << (x - x0);\n"
"            valid |= bit;\n"
"            if (c != prev) chg |= bit;\n"
"            if (c == 127) is127 |= bit;\n"
"            if (c == 255) white |= bit;\n"
"            prev = c;\n"
"        }\n"
"    }\n"
"    chg &= valid;\n"
"    uint starts = chg & ~is127;\n"
"    if (x0 == 0) {\n"
"        chg &= ~1u;\n"
"        starts = (starts & ~1u) | (~is127 & 1u);\n"
"    }\n"
"    int n = 0;\n"
"    while (starts) {\n"
"        int i = ctz(starts);\n"
"        starts &= starts - 1;\n"
"        // the run ends just before the next value change\n"
"        uint later = (i < 31) ? (chg >> (i + 1)) : 0u;\n"
"        int e;\n"
"        if (later) {\n"
"            e = x0 + i + 1 + ctz(later) - 1;\n"
"        } else {\n"
"            e = xmax;\n"
"            for (int wj = wi + 1; wj < nw; wj++) {\n"
"                uint fcj = first_chg[y*nw + wj];\n"
"                if (fcj != NOCHG) { e = (int)fcj - 1; break; }\n"
"            }\n"
"        }\n"
"        out[n].start = (ushort)(x0 + i);\n"
"        out[n].end = (ushort)e;\n"
"        out[n].v = (white >> i) & 1 ? (uchar)255 : (uchar)0;\n"
"        n++;\n"
"    }\n"
"}\n";

static void *usm_grow(at_ocl_t *o, void *cur, size_t *cap, size_t need)
{
    if (*cap >= need && cur)
        return cur;
    if (cur)
        o->usm_free(o->ctx, cur);
    void *p = o->usm_alloc(o->ctx, NULL, need, 4096, NULL);
    *cap = p ? need : 0;
    return p;
}

// device USM: GPU-cached, no CPU coherency snoops; for buffers only
// kernels touch
static void *usm_grow_dev(at_ocl_t *o, void *cur, size_t *cap, size_t need)
{
    if (*cap >= need && cur)
        return cur;
    if (cur)
        o->usm_free(o->ctx, cur);
    void *p = o->usm_dev_alloc(o->ctx, o->dev, NULL, need, 4096, NULL);
    *cap = p ? need : 0;
    return p;
}

void at_ocl_destroy(at_ocl_t *o)
{
    if (!o)
        return;
    if (o->q)
        clFinish(o->q);
    if (o->usm_free) {
        if (o->img) o->usm_free(o->ctx, o->img);
        if (o->threshim) o->usm_free(o->ctx, o->threshim);
        if (o->tiles) o->usm_free(o->ctx, o->tiles);
        if (o->row_off) o->usm_free(o->ctx, o->row_off);
        if (o->win_cnt) o->usm_free(o->ctx, o->win_cnt);
        if (o->first_chg) o->usm_free(o->ctx, o->first_chg);
        if (o->runs) o->usm_free(o->ctx, o->runs);
    }
    if (o->k_minmax) clReleaseKernel(o->k_minmax);
    if (o->k_blur) clReleaseKernel(o->k_blur);
    if (o->k_thresh_rle) clReleaseKernel(o->k_thresh_rle);
    if (o->k_rle_scan) clReleaseKernel(o->k_rle_scan);
    if (o->k_row_prefix) clReleaseKernel(o->k_row_prefix);
    if (o->k_rle_emit) clReleaseKernel(o->k_rle_emit);
    if (o->prog) clReleaseProgram(o->prog);
    if (o->q) clReleaseCommandQueue(o->q);
    if (o->ctx) clReleaseContext(o->ctx);
    free(o);
}

static at_ocl_t *at_ocl_create(void)
{
    at_ocl_t *o = calloc(1, sizeof(*o));
    if (!o)
        return NULL;

    cl_uint nplat = 0;
    if (clGetPlatformIDs(0, NULL, &nplat) != CL_SUCCESS || nplat == 0)
        goto fail;
    cl_platform_id plats[8];
    if (nplat > 8) nplat = 8;
    clGetPlatformIDs(nplat, plats, NULL);

    // first platform with a GPU device wins
    for (cl_uint i = 0; i < nplat && !o->dev; i++) {
        cl_device_id dev;
        if (clGetDeviceIDs(plats[i], CL_DEVICE_TYPE_GPU, 1, &dev, NULL) == CL_SUCCESS) {
            o->platform = plats[i];
            o->dev = dev;
        }
    }
    if (!o->dev)
        goto fail;

    char ext[8192] = {0};
    clGetDeviceInfo(o->dev, CL_DEVICE_EXTENSIONS, sizeof(ext)-1, ext, NULL);
    if (!strstr(ext, "cl_intel_unified_shared_memory"))
        goto fail;

    o->usm_alloc = (clHostMemAllocINTEL_fn)
        clGetExtensionFunctionAddressForPlatform(o->platform, "clHostMemAllocINTEL");
    o->usm_dev_alloc = (clDeviceMemAllocINTEL_fn)
        clGetExtensionFunctionAddressForPlatform(o->platform, "clDeviceMemAllocINTEL");
    o->usm_memcpy = (clEnqueueMemcpyINTEL_fn)
        clGetExtensionFunctionAddressForPlatform(o->platform, "clEnqueueMemcpyINTEL");
    o->usm_free = (clMemFreeINTEL_fn)
        clGetExtensionFunctionAddressForPlatform(o->platform, "clMemFreeINTEL");
    o->set_arg_ptr = (clSetKernelArgMemPointerINTEL_fn)
        clGetExtensionFunctionAddressForPlatform(o->platform, "clSetKernelArgMemPointerINTEL");
    if (!o->usm_alloc || !o->usm_dev_alloc || !o->usm_memcpy || !o->usm_free || !o->set_arg_ptr)
        goto fail;

    cl_int err;
    o->ctx = clCreateContext(NULL, 1, &o->dev, NULL, NULL, &err);
    if (err != CL_SUCCESS)
        goto fail;
    const char *prof = getenv("APRILTAG_OCL_PROF");
    o->prof = prof && prof[0] == '1';
    cl_queue_properties qprops[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    o->q = clCreateCommandQueueWithProperties(o->ctx, o->dev, o->prof ? qprops : NULL, &err);
    if (err != CL_SUCCESS)
        goto fail;

    o->prog = clCreateProgramWithSource(o->ctx, 1, &KSRC, NULL, &err);
    if (err != CL_SUCCESS)
        goto fail;
    if (clBuildProgram(o->prog, 1, &o->dev, "", NULL, NULL) != CL_SUCCESS) {
        char log[4096] = {0};
        clGetProgramBuildInfo(o->prog, o->dev, CL_PROGRAM_BUILD_LOG, sizeof(log)-1, log, NULL);
        fprintf(stderr, "apriltag: OpenCL build failed:\n%s\n", log);
        goto fail;
    }

    if (!(o->k_minmax = clCreateKernel(o->prog, "tile_minmax", &err)) ||
        !(o->k_blur = clCreateKernel(o->prog, "tile_blur", &err)) ||
        !(o->k_thresh_rle = clCreateKernel(o->prog, "thresh_rle", &err)) ||
        !(o->k_rle_scan = clCreateKernel(o->prog, "rle_scan_rows", &err)) ||
        !(o->k_row_prefix = clCreateKernel(o->prog, "row_prefix", &err)) ||
        !(o->k_rle_emit = clCreateKernel(o->prog, "rle_emit", &err)))
        goto fail;

    return o;

  fail:
    at_ocl_destroy(o);
    return NULL;
}

at_ocl_t *at_ocl_get(apriltag_detector_t *td)
{
    if (td->ocl_state == 2)
        return NULL;
    if (td->ocl_state == 1)
        return td->ocl;

    const char *env = getenv("APRILTAG_OPENCL");
    if (env && env[0] == '0') {
        td->ocl_state = 2;
        return NULL;
    }

    td->ocl = at_ocl_create();
    td->ocl_state = td->ocl ? 1 : 2;
    if (!td->ocl)
        fprintf(stderr, "apriltag: no usable OpenCL GPU; staying on the CPU path\n");
    return td->ocl;
}

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

static void prof_event(at_ocl_t *o, const char *name, cl_event ev)
{
    if (!o->prof || !ev)
        return;
    cl_ulong t0 = 0, t1 = 0;
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, NULL);
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, NULL);
    fprintf(stderr, "  ocl %-12s %8.3f ms\n", name, (t1 - t0) * 1e-6);
    clReleaseEvent(ev);
}

image_u8_t *at_ocl_threshold(at_ocl_t *o, apriltag_detector_t *td, image_u8_t *im,
                             struct row_run **runs_out, uint32_t **row_off_out)
{
    int w = im->width, h = im->height, s = im->stride;
    const int tilesz = 4;
    int tw = w / tilesz, th = h / tilesz;

    if (tw < 2 || th < 2 || w < 2)
        return NULL; // degenerate frame; let the CPU handle it

    size_t imsz = (size_t)h * s;
    o->img = usm_grow_dev(o, o->img, &o->img_cap, imsz);
    o->threshim = usm_grow(o, o->threshim, &o->threshim_cap, imsz);
    o->tiles = usm_grow_dev(o, o->tiles, &o->tiles_cap, (size_t)4 * tw * th);
    int nw = (w + 31) / 32; // 32-pixel RLE/threshold windows per row
    o->row_off = usm_grow(o, (void*)o->row_off, &o->row_off_cap, (size_t)(h + 1) * sizeof(uint32_t));
    o->win_cnt = usm_grow_dev(o, (void*)o->win_cnt, &o->win_cnt_cap, (size_t)h * nw * sizeof(uint32_t));
    o->first_chg = usm_grow_dev(o, (void*)o->first_chg, &o->first_chg_cap, (size_t)h * nw * sizeof(uint32_t));
    // worst case is alternating colors: w-1 single-pixel runs per row
    o->runs = usm_grow(o, (void*)o->runs, &o->runs_cap, (size_t)h * (w - 1) * sizeof(struct row_run));
    if (!o->img || !o->threshim || !o->tiles || !o->row_off || !o->win_cnt || !o->first_chg || !o->runs)
        return NULL;

    if (td->qtp.min_white_black_diff < 0 || td->qtp.min_white_black_diff > 255)
        return NULL; // the vector compare in thresh_rle is uchar-wide

    cl_event evs[7] = {0};
    cl_event *pev = o->prof ? evs : NULL;
    double tenq0 = o->prof ? now_ms() : 0;

    // async GPU-side copy of the frame into device-local USM; queued
    // ahead of the kernels, so no separate sync is paid
    if (o->usm_memcpy(o->q, CL_FALSE, o->img, im->buf, imsz,
                      0, NULL, pev ? &evs[6] : NULL) != CL_SUCCESS)
        return NULL;
    if (o->prof)
        fprintf(stderr, "  ocl %-12s %8.3f ms\n", "enq memcpy", now_ms() - tenq0);

    uint8_t *im_max = o->tiles;
    uint8_t *im_min = o->tiles + (size_t)tw * th;
    uint8_t *bmax = o->tiles + (size_t)2 * tw * th;
    uint8_t *bmin = o->tiles + (size_t)3 * tw * th;

    cl_int err = CL_SUCCESS;
    int min_wbd = td->qtp.min_white_black_diff;

    err |= o->set_arg_ptr(o->k_minmax, 0, o->img);
    err |= clSetKernelArg(o->k_minmax, 1, sizeof(s), &s);
    err |= clSetKernelArg(o->k_minmax, 2, sizeof(tw), &tw);
    err |= clSetKernelArg(o->k_minmax, 3, sizeof(th), &th);
    err |= o->set_arg_ptr(o->k_minmax, 4, im_max);
    err |= o->set_arg_ptr(o->k_minmax, 5, im_min);
    size_t g2[2] = { ((size_t)tw + 15) / 16 * 16, (size_t)th };
    err |= clEnqueueNDRangeKernel(o->q, o->k_minmax, 2, NULL, g2, NULL, 0, NULL, pev ? &evs[0] : NULL);

    err |= o->set_arg_ptr(o->k_blur, 0, im_max);
    err |= o->set_arg_ptr(o->k_blur, 1, im_min);
    err |= clSetKernelArg(o->k_blur, 2, sizeof(tw), &tw);
    err |= clSetKernelArg(o->k_blur, 3, sizeof(th), &th);
    err |= o->set_arg_ptr(o->k_blur, 4, bmax);
    err |= o->set_arg_ptr(o->k_blur, 5, bmin);
    err |= clEnqueueNDRangeKernel(o->q, o->k_blur, 2, NULL, g2, NULL, 0, NULL, pev ? &evs[1] : NULL);

    err |= o->set_arg_ptr(o->k_thresh_rle, 0, o->img);
    err |= o->set_arg_ptr(o->k_thresh_rle, 1, o->threshim);
    err |= clSetKernelArg(o->k_thresh_rle, 2, sizeof(w), &w);
    err |= clSetKernelArg(o->k_thresh_rle, 3, sizeof(h), &h);
    err |= clSetKernelArg(o->k_thresh_rle, 4, sizeof(s), &s);
    err |= clSetKernelArg(o->k_thresh_rle, 5, sizeof(tw), &tw);
    err |= clSetKernelArg(o->k_thresh_rle, 6, sizeof(th), &th);
    err |= clSetKernelArg(o->k_thresh_rle, 7, sizeof(min_wbd), &min_wbd);
    err |= o->set_arg_ptr(o->k_thresh_rle, 8, bmax);
    err |= o->set_arg_ptr(o->k_thresh_rle, 9, bmin);
    err |= clSetKernelArg(o->k_thresh_rle, 10, sizeof(nw), &nw);
    err |= o->set_arg_ptr(o->k_thresh_rle, 11, o->win_cnt);
    err |= o->set_arg_ptr(o->k_thresh_rle, 12, o->first_chg);
    size_t gwin[2] = { ((size_t)nw + 15) / 16 * 16, (size_t)h };
    err |= clEnqueueNDRangeKernel(o->q, o->k_thresh_rle, 2, NULL, gwin, NULL, 0, NULL, pev ? &evs[2] : NULL);

    err |= o->set_arg_ptr(o->k_rle_scan, 0, o->win_cnt);
    err |= clSetKernelArg(o->k_rle_scan, 1, sizeof(h), &h);
    err |= clSetKernelArg(o->k_rle_scan, 2, sizeof(nw), &nw);
    err |= o->set_arg_ptr(o->k_rle_scan, 3, o->row_off);
    size_t g1 = ((size_t)h + 63) / 64 * 64;
    err |= clEnqueueNDRangeKernel(o->q, o->k_rle_scan, 1, NULL, &g1, NULL, 0, NULL, pev ? &evs[3] : NULL);

    err |= o->set_arg_ptr(o->k_row_prefix, 0, o->row_off);
    err |= clSetKernelArg(o->k_row_prefix, 1, sizeof(h), &h);
    size_t gp = 256, lp = 256;
    err |= clEnqueueNDRangeKernel(o->q, o->k_row_prefix, 1, NULL, &gp, &lp, 0, NULL, pev ? &evs[4] : NULL);

    err |= o->set_arg_ptr(o->k_rle_emit, 0, o->threshim);
    err |= clSetKernelArg(o->k_rle_emit, 1, sizeof(w), &w);
    err |= clSetKernelArg(o->k_rle_emit, 2, sizeof(h), &h);
    err |= clSetKernelArg(o->k_rle_emit, 3, sizeof(s), &s);
    err |= clSetKernelArg(o->k_rle_emit, 4, sizeof(nw), &nw);
    err |= o->set_arg_ptr(o->k_rle_emit, 5, o->win_cnt);
    err |= o->set_arg_ptr(o->k_rle_emit, 6, o->first_chg);
    err |= o->set_arg_ptr(o->k_rle_emit, 7, o->row_off);
    err |= o->set_arg_ptr(o->k_rle_emit, 8, o->runs);
    err |= clEnqueueNDRangeKernel(o->q, o->k_rle_emit, 2, NULL, gwin, NULL, 0, NULL, pev ? &evs[5] : NULL);

    double tsync0 = o->prof ? now_ms() : 0;
    if (o->prof)
        fprintf(stderr, "  ocl %-12s %8.3f ms\n", "enqueues", tsync0 - tenq0);
    if (err != CL_SUCCESS || clFinish(o->q) != CL_SUCCESS)
        return NULL;
    if (o->prof) {
        fprintf(stderr, "  ocl %-12s %8.3f ms\n", "finish", now_ms() - tsync0);
        prof_event(o, "img copy", evs[6]);
        prof_event(o, "minmax", evs[0]);
        prof_event(o, "blur", evs[1]);
        prof_event(o, "thresh_rle", evs[2]);
        prof_event(o, "rle_scan", evs[3]);
        prof_event(o, "row_prefix", evs[4]);
        prof_event(o, "rle_emit", evs[5]);
    }

    // width/height/stride are const in image_u8_t; build and copy in,
    // like image_u8_create_stride does
    image_u8_t tmp = { .width = w, .height = h, .stride = s, .buf = o->threshim };
    memcpy(&o->thim, &tmp, sizeof(tmp));

    *runs_out = o->runs;
    *row_off_out = o->row_off;
    return &o->thim;
}
