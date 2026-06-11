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
#include <math.h>

#include <CL/cl.h>
#include <CL/cl_ext.h>

#include "apriltag_opencl.h"
#include "apriltag_opencl_quad.h"

// struct row_run is mirrored in the kernel source; both sides must be
// {u16,u16,u8} padded to 6 bytes
_Static_assert(sizeof(struct row_run) == 6, "row_run layout drifted");
_Static_assert(sizeof(struct pt) == 8, "pt layout drifted");
_Static_assert(sizeof(struct at_quad_out) == 40, "quad_out layout drifted");

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
    cl_kernel k_ccl_init;
    cl_kernel k_ccl_edges;
    cl_kernel k_ccl_union;
    cl_kernel k_ccl_compress;
    cl_kernel k_ccl_sizes;
    cl_kernel k_ccl_publish;
    cl_kernel k_fill_run_y;
    cl_kernel k_fit_quads;
    cl_kernel k_ht_clear;
    cl_kernel k_gc_sweep;
    cl_kernel k_scan_blocks;
    cl_kernel k_scan_tops;
    cl_kernel k_scan_add;
    cl_kernel k_gc_compact;

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

    // CCL buffers: device-side working set, host-USM published results
    uint32_t *d_parent;  size_t d_parent_cap;
    uint32_t *d_acc;     size_t d_acc_cap;
    void     *d_edges;   size_t d_edges_cap;  // uint2 pairs
    uint32_t *uf_parent; size_t uf_parent_cap;
    uint32_t *uf_size;   size_t uf_size_cap;
    uint32_t *flags;     size_t flags_cap;    // [0]=edge count, [1]=changed

    unionfind_t uf_pub; // returned union-find; arrays point into USM

    // gradient-cluster buffers
    uint16_t *d_run_y;    size_t d_run_y_cap;   // run index -> row (device)
    uint32_t *d_cnt;      size_t d_cnt_cap;     // per-run counts (device)
    uint32_t *d_rec_off;  size_t d_rec_off_cap; // per-run offsets (device)
    uint32_t *d_bsum;     size_t d_bsum_cap;    // scan block sums (device)
    uint64_t *d_htkeys;   size_t d_htkeys_cap;  // cluster hash keys (device)
    void     *gc_recs;    size_t gc_recs_cap;   // host: emitted records
    uint32_t *gc_pd;      size_t gc_pd_cap;     // host: probe -> dense
    uint64_t *gc_dir;     size_t gc_dir_cap;    // host: dense -> clusterid

    // quad-fitting buffers (host USM unless noted)
    struct pt *q_pts;    size_t q_pts_cap;
    uint32_t *q_coff;    size_t q_coff_cap;
    uint32_t *q_csz;     size_t q_csz_cap;
    uint32_t *q_eligscr; size_t q_eligscr_cap;
    uint32_t *q_eligord; size_t q_eligord_cap;
    double   *q_qsm;     size_t q_qsm_cap;     // 7-tap filter, host libm values
    void     *q_out;     size_t q_out_cap;
    void     *q_keys;    size_t q_keys_cap;    // device
    void     *q_tmp;     size_t q_tmp_cap;     // device
    void     *q_lf;      size_t q_lf_cap;      // device
    void     *q_errs;    size_t q_errs_cap;    // device
    void     *q_yfilt;   size_t q_yfilt_cap;   // device
    void     *q_maxima;  size_t q_maxima_cap;  // device
    void     *q_maxerrs; size_t q_maxerrs_cap; // device
    void     *q_memo;    size_t q_memo_cap;    // device

    // which stages ran on the GPU this frame, and for what geometry
    int frame_state; // bit 0: threshold, bit 1: CCL
    int frame_w, frame_h, frame_s;
    bool uf_published; // this frame's labels copied to shared memory

    image_u8_t thim; // returned threshold image; buf points into threshim
};

#define AT_HT_SLOTS (1u << 19)
#define AT_DIR_CAP  (1u << 17)

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
"}\n"
"\n"
"/* ---- connected components over runs --------------------------------\n"
" * Nodes are global run indices plus one virtual node per row for the\n"
" * run-less last column (vcol_base + y), exactly like the CPU\n"
" * union-find. Edges reproduce connect_runs_to_prev: vertical contact\n"
" * for both colors, diagonal contact for white (8-connected), and the\n"
" * white-run-into-last-column special case. Labels converge to each\n"
" * component's minimum node id via atomic-min hooking + compression,\n"
" * matching the canonicalized CPU labels bit for bit. */\n"
"\n"
"kernel void ccl_init(global uint *P, global uint *acc, uint n)\n"
"{\n"
"    uint i = get_global_id(0);\n"
"    if (i >= n) return;\n"
"    P[i] = i;\n"
"    acc[i] = 0;\n"
"}\n"
"\n"
"inline int row_of(global const uint *row_off, int h, uint i)\n"
"{\n"
"    int lo = 0, hi = h - 1; // y with row_off[y] <= i < row_off[y+1]\n"
"    while (lo < hi) {\n"
"        int mid = (lo + hi + 1) >> 1;\n"
"        if (row_off[mid] <= i) lo = mid; else hi = mid - 1;\n"
"    }\n"
"    return lo;\n"
"}\n"
"\n"
"kernel void ccl_edges(global const run_t *runs, global const uint *row_off,\n"
"                      global const uchar *im, int w, int h, int s,\n"
"                      uint vcol_base, uint cap,\n"
"                      global uint2 *edges, volatile global uint *ecount,\n"
"                      global const ushort *run_y)\n"
"{\n"
"    uint i = get_global_id(0);\n"
"    if (i >= vcol_base) return;\n"
"    int y = run_y[i];\n"
"    if (y == 0) return;\n"
"    run_t cur = runs[i];\n"
"    int a0 = cur.start, a1 = cur.end;\n"
"    uchar v = cur.v;\n"
"    uint pe = row_off[y];\n"
"    // first prev-row run with end >= a0-1 (ends are ascending)\n"
"    uint lo = row_off[y-1], hi = pe;\n"
"    while (lo < hi) {\n"
"        uint mid = (lo + hi) >> 1;\n"
"        if ((int)runs[mid].end < a0 - 1) lo = mid + 1; else hi = mid;\n"
"    }\n"
"    for (uint k = lo; k < pe; k++) {\n"
"        run_t p = runs[k];\n"
"        if ((int)p.start > a1 + 1) break;\n"
"        if (p.v != v) continue;\n"
"        int b0 = p.start, b1 = p.end;\n"
"        int conn = 0;\n"
"        int l = max(max(a0, b0), 1), r = min(a1, b1);\n"
"        if (l <= r) {\n"
"            conn = 1;\n"
"        } else if (v == 255) {\n"
"            int xl = max(max(a0, b0 + 1), 1);\n"
"            if (xl <= min(a1, b1 + 1)) conn = 1;\n"
"            else {\n"
"                int xr = max(max(a0, b0 - 1), 1);\n"
"                if (xr <= min(a1, b1 - 1)) conn = 1;\n"
"            }\n"
"        }\n"
"        if (conn) {\n"
"            uint e = atomic_inc(ecount);\n"
"            if (e < cap) edges[e] = (uint2)(i, k);\n"
"        }\n"
"    }\n"
"    if (v == 255 && a1 == w-2 && im[(y-1)*s + (w-1)] == 255 &&\n"
"        im[(y-1)*s + (w-2)] != 255) {\n"
"        uint e = atomic_inc(ecount);\n"
"        if (e < cap) edges[e] = (uint2)(i, vcol_base + (uint)(y-1));\n"
"    }\n"
"}\n"
"\n"
"/* parent values only ever decrease, so chains strictly descend and\n"
" * concurrent walks terminate; the path-halving write is monotone too */\n"
"inline uint ccl_find(volatile global uint *P, uint i)\n"
"{\n"
"    uint p = P[i];\n"
"    while (p != i) {\n"
"        uint gp = P[p];\n"
"        P[i] = gp;\n"
"        i = gp;\n"
"        p = P[i];\n"
"    }\n"
"    return i;\n"
"}\n"
"\n"
"/* ECL-CC-style lock-free union: only roots are hooked (CAS expects\n"
" * P[hi] == hi), so links are never lost and one pass over the edges\n"
" * establishes full connectivity. */\n"
"kernel void ccl_union(global const uint2 *edges, global const uint *ecount,\n"
"                      uint cap, volatile global uint *P)\n"
"{\n"
"    uint t = get_global_id(0);\n"
"    if (t >= min(*ecount, cap)) return;\n"
"    uint2 e = edges[t];\n"
"    uint a = e.x, b = e.y;\n"
"    for (;;) {\n"
"        a = ccl_find(P, a);\n"
"        b = ccl_find(P, b);\n"
"        if (a == b) break;\n"
"        uint hi = max(a, b), lo = min(a, b);\n"
"        uint old = atomic_cmpxchg(&P[hi], hi, lo);\n"
"        if (old == hi || old == lo) break;\n"
"        // someone re-rooted hi first; union its new root with ours\n"
"        a = lo;\n"
"        b = old;\n"
"    }\n"
"}\n"
"\n"
"kernel void ccl_compress(volatile global uint *P, uint n)\n"
"{\n"
"    uint i = get_global_id(0);\n"
"    if (i >= n) return;\n"
"    uint p = P[i];\n"
"    while (P[p] != p) p = P[p];\n"
"    P[i] = p;\n"
"}\n"
"\n"
"kernel void ccl_sizes(global const run_t *runs, global const uint *P,\n"
"                      volatile global uint *acc, uint vcol_base, uint n)\n"
"{\n"
"    uint i = get_global_id(0);\n"
"    if (i >= n) return;\n"
"    uint px = (i < vcol_base) ? (uint)(runs[i].end - runs[i].start + 1) : 1u;\n"
"    atomic_add(&acc[P[i]], px);\n"
"}\n"
"\n"
"kernel void ccl_publish(global const uint *P, global const uint *acc,\n"
"                        global uint *hp, global uint *hs, uint n)\n"
"{\n"
"    uint i = get_global_id(0);\n"
"    if (i >= n) return;\n"
"    uint p = P[i];\n"
"    hp[i] = p;\n"
"    hs[i] = (p == i) ? acc[i] - 1 : 0;\n"
"}\n"
"\n"
"/* ---- gradient clusters -------------------------------------------\n"
" * One work-item per run: replays the CPU run-driven sweep for that\n"
" * run exactly (same appends, same order), emitting (cluster, point)\n"
" * records at per-run offsets so the flat record stream equals the\n"
" * CPU emission order. Cluster identity is resolved through a global\n"
" * open-addressing hash table keyed by the canonical label pair. */\n"
"\n"
"typedef struct { uint slot; ushort x, y; short gx, gy; } rec_t;\n"
"#define EMPTY64 0xffffffffffffffffUL\n"
"\n"
"/* run containing x in [b,e), or -1 (ends ascending and disjoint) */\n"
"inline int run_at(global const run_t *runs, uint b, uint e, int x)\n"
"{\n"
"    uint lo = b, hi = e;\n"
"    while (lo < hi) {\n"
"        uint m = (lo + hi) >> 1;\n"
"        if ((int)runs[m].end < x) lo = m + 1; else hi = m;\n"
"    }\n"
"    return (lo < e && (int)runs[lo].start <= x) ? (int)lo : -1;\n"
"}\n"
"\n"
"/* did run j (row y) emit a (1,1) point at its last pixel? Equivalent\n"
" * to: run j usable, and the pixel at (end+1, y+1) has the opposite\n"
" * value with a usable component. Feeds the next run's cl_entry. */\n"
"inline int gc_fired11(global const run_t *runs, global const uint *row_off,\n"
"                      global const uchar *im, int w, int ts,\n"
"                      global const uint *P, global const uint *acc, uint min_px,\n"
"                      uint vcol_base, int y, uint j)\n"
"{\n"
"    run_t Q = runs[j];\n"
"    if ((int)Q.end < max((int)Q.start, 1)) return 0;\n"
"    if (acc[P[j]] < min_px) return 0;\n"
"    int x2 = (int)Q.end + 1;\n"
"    uchar uopp = 255 - Q.v;\n"
"    if (x2 <= w - 2) {\n"
"        int k = run_at(runs, row_off[y+1], row_off[y+2], x2);\n"
"        return k >= 0 && runs[k].v == uopp && acc[P[k]] >= min_px;\n"
"    }\n"
"    return im[(y+1)*ts + x2] == uopp && acc[P[vcol_base + (y+1)]] >= min_px;\n"
"}\n"
"\n"
"inline uint ht_insert(volatile global ulong *keys, uint mask, ulong id,\n"
"                      volatile global uint *overflow)\n"
"{\n"
"    uint p = (uint)((id * 0x9E3779B97F4A7C15UL) >> 40) & mask;\n"
"    for (uint probes = 0; probes <= mask; probes++, p = (p + 1) & mask) {\n"
"        ulong old = atom_cmpxchg(&keys[p], EMPTY64, id);\n"
"        if (old == EMPTY64 || old == id)\n"
"            return p;\n"
"    }\n"
"    *overflow = 1;\n"
"    return 0;\n"
"}\n"
"\n"
"/* run index -> row table; replaces per-run binary searches */\n"
"kernel void fill_run_y(global const uint *row_off, int h, global ushort *run_y)\n"
"{\n"
"    int y = get_global_id(0);\n"
"    if (y >= h) return;\n"
"    for (uint i = row_off[y]; i < row_off[y+1]; i++)\n"
"        run_y[i] = (ushort)y;\n"
"}\n"
"\n"
"kernel void ht_clear(global ulong *keys, uint n)\n"
"{\n"
"    uint i = get_global_id(0);\n"
"    if (i < n) keys[i] = EMPTY64;\n"
"}\n"
"\n"
"__attribute__((intel_reqd_sub_group_size(16)))\n"
"kernel void gc_sweep(global const run_t *runs, global const uint *row_off,\n"
"                     global const uchar *im, int w, int h, int ts,\n"
"                     global const uint *P, global const uint *acc,\n"
"                     uint min_px, uint vcol_base, int do_emit,\n"
"                     global uint *cnt, global const uint *rec_off,\n"
"                     global rec_t *recs,\n"
"                     volatile global ulong *ht_keys, uint ht_mask,\n"
"                     volatile global uint *overflow,\n"
"                     global const ushort *run_y,\n"
"                     uint reccap, volatile global uint *rec_overflow)\n"
"{\n"
"    uint i = get_global_id(0);\n"
"    if (i >= vcol_base) return;\n"
"    int y = run_y[i];\n"
"    uint n = 0;\n"
"    if (y < 1 || y > h - 2) { if (!do_emit) cnt[i] = 0; return; }\n"
"    run_t A = runs[i];\n"
"    int a0 = A.start, a1 = A.end;\n"
"    uchar v0 = A.v;\n"
"    uchar vopp = 255 - v0;\n"
"    int vdiff = (int)vopp - (int)v0;\n"
"    int ax0 = max(a0, 1);\n"
"    uint ia = i - row_off[y];\n"
"    uint na = row_off[y+1] - row_off[y];\n"
"    uint rep0 = P[i];\n"
"    int rep0_usable = acc[rep0] >= min_px;\n"
"    int cl_entry;\n"
"    if (!do_emit) {\n"
"        cl_entry = 0;\n"
"        if (ia > 0 && (int)runs[i-1].end + 1 == a0)\n"
"            cl_entry = gc_fired11(runs, row_off, im, w, ts, P, acc, min_px, vcol_base, y, i-1);\n"
"    } else {\n"
"        cl_entry = (cnt[i] >> 31) & 1; // carried from the count pass\n"
"    }\n"
"    int done10 = 0;\n"
"    uint off = do_emit ? rec_off[i] : 0;\n"
"    ulong last_id = EMPTY64;\n"
"    uint last_slot = 0;\n"
"\n"
"#define GC_APPEND(R1, X, Y, GX, GY) do { \\\n"
"    if (!do_emit) { n++; } else { \\\n"
"        ulong _id = (rep0 < (R1)) ? (((ulong)(R1) << 32) + rep0) \\\n"
"                                  : (((ulong)rep0 << 32) + (R1)); \\\n"
"        if (_id != last_id) { \\\n"
"            last_id = _id; \\\n"
"            last_slot = ht_insert(ht_keys, ht_mask, _id, overflow); \\\n"
"        } \\\n"
"        if (off < reccap) { \\\n"
"            rec_t _r = { last_slot, (ushort)(X), (ushort)(Y), (short)(GX), (short)(GY) }; \\\n"
"            recs[off] = _r; \\\n"
"        } else { *rec_overflow = 1; } \\\n"
"        off++; \\\n"
"    } } while (0)\n"
"\n"
"#define GC_EMIT10() do { \\\n"
"    done10 = 1; \\\n"
"    uchar _v1 = im[y*ts + a1 + 1]; \\\n"
"    if ((int)v0 + (int)_v1 == 255 && rep0_usable) { \\\n"
"        uint _rep1; \\\n"
"        if (ia + 1 < na && (int)runs[i+1].start == a1 + 1) _rep1 = P[i+1]; \\\n"
"        else _rep1 = P[vcol_base + y]; \\\n"
"        if (acc[_rep1] >= min_px) \\\n"
"            GC_APPEND(_rep1, 2*a1 + 1, 2*y, vdiff, 0); \\\n"
"    } } while (0)\n"
"\n"
"    int fired11 = 0;\n"
"    uint pe2 = row_off[y+2];\n"
"    // first next-row run with end >= ax0-1\n"
"    uint lo = row_off[y+1], hi = pe2;\n"
"    while (lo < hi) {\n"
"        uint m = (lo + hi) >> 1;\n"
"        if ((int)runs[m].end < ax0 - 1) lo = m + 1; else hi = m;\n"
"    }\n"
"    for (uint k = lo; k < pe2 && (int)runs[k].start <= a1 + 1; k++) {\n"
"        run_t B = runs[k];\n"
"        if (B.v != vopp) continue;\n"
"        if (!rep0_usable) break;\n"
"        uint rep1 = P[k];\n"
"        if (acc[rep1] < min_px) continue;\n"
"        int b0 = B.start, b1 = B.end;\n"
"        int p1 = b0 - 1;\n"
"        if (p1 >= ax0 && p1 <= a1) {\n"
"            if (p1 == a1 && !done10) GC_EMIT10();\n"
"            GC_APPEND(rep1, 2*p1 + 1, 2*y + 1, vdiff, vdiff);\n"
"            if (p1 == a1) fired11 = 1;\n"
"        }\n"
"        int xs = max(b0, ax0), xe = min(b1 - 1, a1);\n"
"        if (xs <= xe) {\n"
"            if (xs == a1 && !done10) GC_EMIT10();\n"
"            GC_APPEND(rep1, 2*xs, 2*y + 1, 0, vdiff);\n"
"            if (xs == ax0 && b0 < ax0 && !cl_entry)\n"
"                GC_APPEND(rep1, 2*xs - 1, 2*y + 1, -vdiff, vdiff);\n"
"            GC_APPEND(rep1, 2*xs + 1, 2*y + 1, vdiff, vdiff);\n"
"            if (xs == a1) fired11 = 1;\n"
"            int bend = min(xe, a1 - 1);\n"
"            if (!do_emit) {\n"
"                if (bend > xs) n += 2 * (uint)(bend - xs); // interior pairs, O(1)\n"
"            } else {\n"
"                for (int x = xs + 1; x <= bend; x++) {\n"
"                    GC_APPEND(rep1, 2*x, 2*y + 1, 0, vdiff);\n"
"                    GC_APPEND(rep1, 2*x + 1, 2*y + 1, vdiff, vdiff);\n"
"                }\n"
"            }\n"
"            if (xe == a1 && xe > xs) {\n"
"                if (!done10) GC_EMIT10();\n"
"                GC_APPEND(rep1, 2*xe, 2*y + 1, 0, vdiff);\n"
"                GC_APPEND(rep1, 2*xe + 1, 2*y + 1, vdiff, vdiff);\n"
"                fired11 = 1;\n"
"            }\n"
"        }\n"
"        if (b1 >= ax0 && b1 <= a1 && b1 > xe) {\n"
"            if (b1 == a1 && !done10) GC_EMIT10();\n"
"            GC_APPEND(rep1, 2*b1, 2*y + 1, 0, vdiff);\n"
"            if (b1 == ax0 && b1 > b0 && !cl_entry)\n"
"                GC_APPEND(rep1, 2*b1 - 1, 2*y + 1, -vdiff, vdiff);\n"
"        }\n"
"        int p3 = b1 + 1;\n"
"        if (p3 >= ax0 && p3 <= a1 && !(p3 == ax0 && cl_entry)) {\n"
"            if (p3 == a1 && !done10) GC_EMIT10();\n"
"            GC_APPEND(rep1, 2*p3 - 1, 2*y + 1, -vdiff, vdiff);\n"
"        }\n"
"    }\n"
"    if (!done10) GC_EMIT10();\n"
"    if (a1 == w - 2 && rep0_usable) {\n"
"        uchar v1 = im[(y+1)*ts + a1 + 1];\n"
"        if ((int)v0 + (int)v1 == 255) {\n"
"            uint rep1 = P[vcol_base + (y+1)];\n"
"            if (acc[rep1] >= min_px)\n"
"                GC_APPEND(rep1, 2*a1 + 1, 2*y + 1, vdiff, vdiff);\n"
"        }\n"
"    }\n"
"    (void)fired11;\n"
"    if (!do_emit) cnt[i] = n | ((uint)cl_entry << 31);\n"
"#undef GC_APPEND\n"
"#undef GC_EMIT10\n"
"}\n"
"\n"
"/* hierarchical exclusive scan of cnt[n] into off[n] */\n"
"kernel void scan_blocks(global const uint *cnt, uint n,\n"
"                        global uint *off, global uint *bsum)\n"
"{\n"
"    local uint buf[2][256];\n"
"    uint lid = get_local_id(0);\n"
"    uint gid = get_global_id(0);\n"
"    uint v = (gid < n) ? (cnt[gid] & 0x7fffffffu) : 0; // high bit carries cl_entry\n"
"    buf[0][lid] = v;\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    int src = 0;\n"
"    for (int d = 1; d < 256; d <<= 1) {\n"
"        uint t = buf[src][lid];\n"
"        if (lid >= (uint)d) t += buf[src][lid - d];\n"
"        buf[1-src][lid] = t;\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"        src = 1 - src;\n"
"    }\n"
"    if (gid < n) off[gid] = buf[src][lid] - v; // exclusive\n"
"    if (lid == 255) bsum[get_group_id(0)] = buf[src][255];\n"
"}\n"
"\n"
"kernel void scan_tops(global uint *bsum, uint nb, global uint *total)\n"
"{\n"
"    local uint buf[2][256];\n"
"    uint lid = get_local_id(0);\n"
"    uint carry = 0;\n"
"    for (uint base = 0; base < nb; base += 256) {\n"
"        uint idx = base + lid;\n"
"        uint v = (idx < nb) ? bsum[idx] : 0;\n"
"        buf[0][lid] = v;\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"        int src = 0;\n"
"        for (int d = 1; d < 256; d <<= 1) {\n"
"            uint t = buf[src][lid];\n"
"            if (lid >= (uint)d) t += buf[src][lid - d];\n"
"            buf[1-src][lid] = t;\n"
"            barrier(CLK_LOCAL_MEM_FENCE);\n"
"            src = 1 - src;\n"
"        }\n"
"        if (idx < nb) bsum[idx] = buf[src][lid] - v + carry; // exclusive\n"
"        carry += buf[src][255];\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"    }\n"
"    if (lid == 0) *total = carry;\n"
"}\n"
"\n"
"kernel void scan_add(global uint *off, uint n, global const uint *bsum)\n"
"{\n"
"    uint gid = get_global_id(0);\n"
"    if (gid < n) off[gid] += bsum[get_group_id(0)];\n"
"}\n"
"\n"
"/* assign dense ids to occupied hash slots; publish probe->dense and\n"
" * dense->clusterid for the host */\n"
"kernel void gc_compact(global const ulong *keys, uint nslots,\n"
"                       volatile global uint *ncl, uint dir_cap,\n"
"                       global uint *probe_dense, global ulong *dir_ids,\n"
"                       volatile global uint *overflow)\n"
"{\n"
"    uint p = get_global_id(0);\n"
"    if (p >= nslots) return;\n"
"    ulong k = keys[p];\n"
"    if (k == EMPTY64) return;\n"
"    uint d = atomic_inc(ncl);\n"
"    if (d >= dir_cap) { *overflow = 1; return; }\n"
"    probe_dense[p] = d;\n"
"    dir_ids[d] = k;\n"
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
        if (o->d_parent) o->usm_free(o->ctx, o->d_parent);
        if (o->d_acc) o->usm_free(o->ctx, o->d_acc);
        if (o->d_edges) o->usm_free(o->ctx, o->d_edges);
        if (o->uf_parent) o->usm_free(o->ctx, o->uf_parent);
        if (o->uf_size) o->usm_free(o->ctx, o->uf_size);
        if (o->flags) o->usm_free(o->ctx, o->flags);
        if (o->d_run_y) o->usm_free(o->ctx, o->d_run_y);
        if (o->d_cnt) o->usm_free(o->ctx, o->d_cnt);
        if (o->d_rec_off) o->usm_free(o->ctx, o->d_rec_off);
        if (o->d_bsum) o->usm_free(o->ctx, o->d_bsum);
        if (o->d_htkeys) o->usm_free(o->ctx, o->d_htkeys);
        if (o->gc_recs) o->usm_free(o->ctx, o->gc_recs);
        if (o->gc_pd) o->usm_free(o->ctx, o->gc_pd);
        if (o->gc_dir) o->usm_free(o->ctx, o->gc_dir);
        if (o->q_pts) o->usm_free(o->ctx, o->q_pts);
        if (o->q_coff) o->usm_free(o->ctx, o->q_coff);
        if (o->q_csz) o->usm_free(o->ctx, o->q_csz);
        if (o->q_eligscr) o->usm_free(o->ctx, o->q_eligscr);
        if (o->q_eligord) o->usm_free(o->ctx, o->q_eligord);
        if (o->q_qsm) o->usm_free(o->ctx, o->q_qsm);
        if (o->q_out) o->usm_free(o->ctx, o->q_out);
        if (o->q_keys) o->usm_free(o->ctx, o->q_keys);
        if (o->q_tmp) o->usm_free(o->ctx, o->q_tmp);
        if (o->q_lf) o->usm_free(o->ctx, o->q_lf);
        if (o->q_errs) o->usm_free(o->ctx, o->q_errs);
        if (o->q_yfilt) o->usm_free(o->ctx, o->q_yfilt);
        if (o->q_maxima) o->usm_free(o->ctx, o->q_maxima);
        if (o->q_maxerrs) o->usm_free(o->ctx, o->q_maxerrs);
        if (o->q_memo) o->usm_free(o->ctx, o->q_memo);
    }
    if (o->k_minmax) clReleaseKernel(o->k_minmax);
    if (o->k_blur) clReleaseKernel(o->k_blur);
    if (o->k_thresh_rle) clReleaseKernel(o->k_thresh_rle);
    if (o->k_rle_scan) clReleaseKernel(o->k_rle_scan);
    if (o->k_row_prefix) clReleaseKernel(o->k_row_prefix);
    if (o->k_rle_emit) clReleaseKernel(o->k_rle_emit);
    if (o->k_ccl_init) clReleaseKernel(o->k_ccl_init);
    if (o->k_ccl_edges) clReleaseKernel(o->k_ccl_edges);
    if (o->k_ccl_union) clReleaseKernel(o->k_ccl_union);
    if (o->k_ccl_compress) clReleaseKernel(o->k_ccl_compress);
    if (o->k_ccl_sizes) clReleaseKernel(o->k_ccl_sizes);
    if (o->k_ccl_publish) clReleaseKernel(o->k_ccl_publish);
    if (o->k_fill_run_y) clReleaseKernel(o->k_fill_run_y);
    if (o->k_fit_quads) clReleaseKernel(o->k_fit_quads);
    if (o->k_ht_clear) clReleaseKernel(o->k_ht_clear);
    if (o->k_gc_sweep) clReleaseKernel(o->k_gc_sweep);
    if (o->k_scan_blocks) clReleaseKernel(o->k_scan_blocks);
    if (o->k_scan_tops) clReleaseKernel(o->k_scan_tops);
    if (o->k_scan_add) clReleaseKernel(o->k_scan_add);
    if (o->k_gc_compact) clReleaseKernel(o->k_gc_compact);
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

    const char *sources[2] = { KSRC, KSRC_QUAD };
    o->prog = clCreateProgramWithSource(o->ctx, 2, sources, NULL, &err);
    if (err != CL_SUCCESS)
        goto fail;
    // the quad kernel's float divide/sqrt must round exactly like the CPU
    if (clBuildProgram(o->prog, 1, &o->dev, "-cl-fp32-correctly-rounded-divide-sqrt", NULL, NULL) != CL_SUCCESS) {
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
        !(o->k_rle_emit = clCreateKernel(o->prog, "rle_emit", &err)) ||
        !(o->k_ccl_init = clCreateKernel(o->prog, "ccl_init", &err)) ||
        !(o->k_ccl_edges = clCreateKernel(o->prog, "ccl_edges", &err)) ||
        !(o->k_ccl_union = clCreateKernel(o->prog, "ccl_union", &err)) ||
        !(o->k_ccl_compress = clCreateKernel(o->prog, "ccl_compress", &err)) ||
        !(o->k_ccl_sizes = clCreateKernel(o->prog, "ccl_sizes", &err)) ||
        !(o->k_ccl_publish = clCreateKernel(o->prog, "ccl_publish", &err)) ||
        !(o->k_fill_run_y = clCreateKernel(o->prog, "fill_run_y", &err)) ||
        !(o->k_fit_quads = clCreateKernel(o->prog, "fit_quads_k", &err)) ||
        !(o->k_ht_clear = clCreateKernel(o->prog, "ht_clear", &err)) ||
        !(o->k_gc_sweep = clCreateKernel(o->prog, "gc_sweep", &err)) ||
        !(o->k_scan_blocks = clCreateKernel(o->prog, "scan_blocks", &err)) ||
        !(o->k_scan_tops = clCreateKernel(o->prog, "scan_tops", &err)) ||
        !(o->k_scan_add = clCreateKernel(o->prog, "scan_add", &err)) ||
        !(o->k_gc_compact = clCreateKernel(o->prog, "gc_compact", &err)))
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

    o->frame_state = 1;
    o->frame_w = w;
    o->frame_h = h;
    o->frame_s = s;

    *runs_out = o->runs;
    *row_off_out = o->row_off;
    return &o->thim;
}

unionfind_t *at_ocl_connected_components(at_ocl_t *o, apriltag_detector_t *td,
                                         image_u8_t *threshim, int w, int h, int ts,
                                         struct row_run *runs, uint32_t *row_off)
{
    (void)td;
    // only valid when this frame's threshold ran on the GPU (the run
    // tables must already live in shared memory)
    if (runs != o->runs || row_off != o->row_off || threshim->buf != o->threshim)
        return NULL;

    uint32_t vcol_base = row_off[h];
    uint32_t maxid = vcol_base + h;
    // edge count is empirically ~1.0x the run count; 2x + slack is ample
    uint32_t cap = 2 * vcol_base + h + 64;

    o->d_run_y = usm_grow_dev(o, o->d_run_y, &o->d_run_y_cap, (size_t)(vcol_base ? vcol_base : 1) * 2);
    o->d_parent = usm_grow_dev(o, o->d_parent, &o->d_parent_cap, (size_t)(maxid + 1) * 4);
    o->d_acc = usm_grow_dev(o, o->d_acc, &o->d_acc_cap, (size_t)(maxid + 1) * 4);
    o->d_edges = usm_grow_dev(o, o->d_edges, &o->d_edges_cap, (size_t)cap * 8);
    o->uf_parent = usm_grow(o, o->uf_parent, &o->uf_parent_cap, (size_t)(maxid + 1) * 4);
    o->uf_size = usm_grow(o, o->uf_size, &o->uf_size_cap, (size_t)(maxid + 1) * 4);
    o->flags = usm_grow(o, o->flags, &o->flags_cap, 8 * sizeof(uint32_t));
    if (!o->d_run_y || !o->d_parent || !o->d_acc || !o->d_edges || !o->uf_parent || !o->uf_size || !o->flags)
        return NULL;

    o->flags[0] = 0; // edge count

    cl_int err = CL_SUCCESS;
    size_t gnodes = ((size_t)maxid + 63) / 64 * 64;
    size_t gruns = ((size_t)vcol_base + 63) / 64 * 64;

    err |= o->set_arg_ptr(o->k_fill_run_y, 0, o->row_off);
    err |= clSetKernelArg(o->k_fill_run_y, 1, sizeof(h), &h);
    err |= o->set_arg_ptr(o->k_fill_run_y, 2, o->d_run_y);
    size_t grows = ((size_t)h + 63) / 64 * 64;
    err |= clEnqueueNDRangeKernel(o->q, o->k_fill_run_y, 1, NULL, &grows, NULL, 0, NULL, NULL);

    err |= o->set_arg_ptr(o->k_ccl_init, 0, o->d_parent);
    err |= o->set_arg_ptr(o->k_ccl_init, 1, o->d_acc);
    err |= clSetKernelArg(o->k_ccl_init, 2, sizeof(maxid), &maxid);
    err |= clEnqueueNDRangeKernel(o->q, o->k_ccl_init, 1, NULL, &gnodes, NULL, 0, NULL, NULL);

    err |= o->set_arg_ptr(o->k_ccl_edges, 0, o->runs);
    err |= o->set_arg_ptr(o->k_ccl_edges, 1, o->row_off);
    err |= o->set_arg_ptr(o->k_ccl_edges, 2, o->threshim);
    err |= clSetKernelArg(o->k_ccl_edges, 3, sizeof(w), &w);
    err |= clSetKernelArg(o->k_ccl_edges, 4, sizeof(h), &h);
    err |= clSetKernelArg(o->k_ccl_edges, 5, sizeof(ts), &ts);
    err |= clSetKernelArg(o->k_ccl_edges, 6, sizeof(vcol_base), &vcol_base);
    err |= clSetKernelArg(o->k_ccl_edges, 7, sizeof(cap), &cap);
    err |= o->set_arg_ptr(o->k_ccl_edges, 8, o->d_edges);
    err |= o->set_arg_ptr(o->k_ccl_edges, 9, o->flags);
    err |= o->set_arg_ptr(o->k_ccl_edges, 10, o->d_run_y);
    err |= clEnqueueNDRangeKernel(o->q, o->k_ccl_edges, 1, NULL, &gruns, NULL, 0, NULL, NULL);

    // the union launch is sized by capacity (the kernel guards against
    // *ecount), so the whole CCL needs only one sync at the end
    size_t gedges = ((size_t)cap + 63) / 64 * 64;
    err |= o->set_arg_ptr(o->k_ccl_union, 0, o->d_edges);
    err |= o->set_arg_ptr(o->k_ccl_union, 1, o->flags);
    err |= clSetKernelArg(o->k_ccl_union, 2, sizeof(cap), &cap);
    err |= o->set_arg_ptr(o->k_ccl_union, 3, o->d_parent);
    err |= clEnqueueNDRangeKernel(o->q, o->k_ccl_union, 1, NULL, &gedges, NULL, 0, NULL, NULL);

    err |= o->set_arg_ptr(o->k_ccl_compress, 0, o->d_parent);
    err |= clSetKernelArg(o->k_ccl_compress, 1, sizeof(maxid), &maxid);
    err |= clEnqueueNDRangeKernel(o->q, o->k_ccl_compress, 1, NULL, &gnodes, NULL, 0, NULL, NULL);

    err |= o->set_arg_ptr(o->k_ccl_sizes, 0, o->runs);
    err |= o->set_arg_ptr(o->k_ccl_sizes, 1, o->d_parent);
    err |= o->set_arg_ptr(o->k_ccl_sizes, 2, o->d_acc);
    err |= clSetKernelArg(o->k_ccl_sizes, 3, sizeof(vcol_base), &vcol_base);
    err |= clSetKernelArg(o->k_ccl_sizes, 4, sizeof(maxid), &maxid);
    err |= clEnqueueNDRangeKernel(o->q, o->k_ccl_sizes, 1, NULL, &gnodes, NULL, 0, NULL, NULL);

    // publishing the union-find to shared memory costs ~8 MB of writes;
    // skip it unless the CPU will actually read it (debug, verification,
    // or a later fallback via at_ocl_ensure_uf)
    o->uf_published = false;
    if (td->debug || getenv("APRILTAG_CCL_VERIFY")) {
        err |= o->set_arg_ptr(o->k_ccl_publish, 0, o->d_parent);
        err |= o->set_arg_ptr(o->k_ccl_publish, 1, o->d_acc);
        err |= o->set_arg_ptr(o->k_ccl_publish, 2, o->uf_parent);
        err |= o->set_arg_ptr(o->k_ccl_publish, 3, o->uf_size);
        err |= clSetKernelArg(o->k_ccl_publish, 4, sizeof(maxid), &maxid);
        err |= clEnqueueNDRangeKernel(o->q, o->k_ccl_publish, 1, NULL, &gnodes, NULL, 0, NULL, NULL);
        o->uf_published = true;
    }

    if (err != CL_SUCCESS || clFinish(o->q) != CL_SUCCESS)
        return NULL;

    if (o->flags[0] > cap)
        return NULL; // edges were dropped; redo on the CPU path

    if (o->prof)
        fprintf(stderr, "  ocl ccl: %u nodes, %u edges\n", maxid, o->flags[0]);

    o->uf_pub.maxid = maxid;
    o->uf_pub.parent = o->uf_parent;
    o->uf_pub.size = o->uf_size;
    o->frame_state |= 2;
    return &o->uf_pub;
}

bool at_ocl_gradient_clusters(at_ocl_t *o, apriltag_detector_t *td,
                              int w, int h, int ts, int min_cluster_pixels,
                              struct at_gc_out *out)
{
    (void)td;
    if (o->frame_state != 3 || o->frame_w != w || o->frame_h != h || o->frame_s != ts)
        return false;
    if (h < 4 || min_cluster_pixels < 0)
        return false;

    uint32_t R = o->row_off[h]; // run count
    if (R == 0) {
        out->recs = (const struct at_gc_rec *)o->gc_recs;
        out->nrecs = 0;
        out->probe_dense = o->gc_pd;
        out->dir_ids = o->gc_dir;
        out->nclusters = 0;
        return true;
    }
    uint32_t vcol_base = R;
    uint32_t min_px = (uint32_t)min_cluster_pixels;
    uint32_t nblocks = (R + 255) / 256;

    o->d_cnt = usm_grow_dev(o, o->d_cnt, &o->d_cnt_cap, (size_t)R * 4);
    o->d_rec_off = usm_grow_dev(o, o->d_rec_off, &o->d_rec_off_cap, (size_t)R * 4);
    o->d_bsum = usm_grow_dev(o, o->d_bsum, &o->d_bsum_cap, (size_t)nblocks * 4);
    o->d_htkeys = usm_grow_dev(o, (void*)o->d_htkeys, &o->d_htkeys_cap, (size_t)AT_HT_SLOTS * 8);
    o->gc_pd = usm_grow(o, o->gc_pd, &o->gc_pd_cap, (size_t)AT_HT_SLOTS * 4);
    o->gc_dir = usm_grow(o, o->gc_dir, &o->gc_dir_cap, (size_t)AT_DIR_CAP * 8);
    if (!o->d_cnt || !o->d_rec_off || !o->d_bsum || !o->d_htkeys || !o->gc_pd || !o->gc_dir)
        return false;

    // generous record capacity so count+emit run in one batch; on
    // overflow the exact total is known and the emit half retries
    size_t want = (size_t)4 << 20;
    if (o->gc_recs_cap / sizeof(struct at_gc_rec) > want)
        want = o->gc_recs_cap / sizeof(struct at_gc_rec);
    o->gc_recs = usm_grow(o, o->gc_recs, &o->gc_recs_cap, want * sizeof(struct at_gc_rec));
    if (!o->gc_recs)
        return false;

    cl_int err = CL_SUCCESS;
    size_t gruns = ((size_t)R + 255) / 256 * 256, l256 = 256;
    size_t ght = AT_HT_SLOTS, g1 = 256;
    uint32_t nht = AT_HT_SLOTS;
    uint32_t dircap = AT_DIR_CAP;
    cl_uint htmask = AT_HT_SLOTS - 1;
    int do_emit;

    err |= o->set_arg_ptr(o->k_gc_sweep, 0, o->runs);
    err |= o->set_arg_ptr(o->k_gc_sweep, 1, o->row_off);
    err |= o->set_arg_ptr(o->k_gc_sweep, 2, o->threshim);
    err |= clSetKernelArg(o->k_gc_sweep, 3, sizeof(w), &w);
    err |= clSetKernelArg(o->k_gc_sweep, 4, sizeof(h), &h);
    err |= clSetKernelArg(o->k_gc_sweep, 5, sizeof(ts), &ts);
    err |= o->set_arg_ptr(o->k_gc_sweep, 6, o->d_parent);
    err |= o->set_arg_ptr(o->k_gc_sweep, 7, o->d_acc);
    err |= clSetKernelArg(o->k_gc_sweep, 8, sizeof(min_px), &min_px);
    err |= clSetKernelArg(o->k_gc_sweep, 9, sizeof(vcol_base), &vcol_base);
    err |= o->set_arg_ptr(o->k_gc_sweep, 11, o->d_cnt);
    err |= o->set_arg_ptr(o->k_gc_sweep, 12, o->d_rec_off);
    err |= o->set_arg_ptr(o->k_gc_sweep, 13, o->gc_recs);
    err |= o->set_arg_ptr(o->k_gc_sweep, 14, o->d_htkeys);
    err |= clSetKernelArg(o->k_gc_sweep, 15, sizeof(htmask), &htmask);
    err |= o->set_arg_ptr(o->k_gc_sweep, 16, o->flags + 4);
    err |= o->set_arg_ptr(o->k_gc_sweep, 17, o->d_run_y);
    err |= o->set_arg_ptr(o->k_gc_sweep, 19, o->flags + 5);

    err |= o->set_arg_ptr(o->k_scan_blocks, 0, o->d_cnt);
    err |= clSetKernelArg(o->k_scan_blocks, 1, sizeof(R), &R);
    err |= o->set_arg_ptr(o->k_scan_blocks, 2, o->d_rec_off);
    err |= o->set_arg_ptr(o->k_scan_blocks, 3, o->d_bsum);

    err |= o->set_arg_ptr(o->k_scan_tops, 0, o->d_bsum);
    err |= clSetKernelArg(o->k_scan_tops, 1, sizeof(nblocks), &nblocks);
    err |= o->set_arg_ptr(o->k_scan_tops, 2, o->flags + 2);

    err |= o->set_arg_ptr(o->k_scan_add, 0, o->d_rec_off);
    err |= clSetKernelArg(o->k_scan_add, 1, sizeof(R), &R);
    err |= o->set_arg_ptr(o->k_scan_add, 2, o->d_bsum);

    err |= o->set_arg_ptr(o->k_ht_clear, 0, o->d_htkeys);
    err |= clSetKernelArg(o->k_ht_clear, 1, sizeof(nht), &nht);

    err |= o->set_arg_ptr(o->k_gc_compact, 0, o->d_htkeys);
    err |= clSetKernelArg(o->k_gc_compact, 1, sizeof(nht), &nht);
    err |= o->set_arg_ptr(o->k_gc_compact, 2, o->flags + 3);
    err |= clSetKernelArg(o->k_gc_compact, 3, sizeof(dircap), &dircap);
    err |= o->set_arg_ptr(o->k_gc_compact, 4, o->gc_pd);
    err |= o->set_arg_ptr(o->k_gc_compact, 5, o->gc_dir);
    err |= o->set_arg_ptr(o->k_gc_compact, 6, o->flags + 4);

    int emit_only = 0;
    for (int attempt = 0; attempt < 2; attempt++) {
        if (!emit_only)
            o->flags[2] = 0; // total records (written by the count scan)
        o->flags[3] = 0; // cluster count
        o->flags[4] = 0; // hash/dir overflow
        o->flags[5] = 0; // record overflow
        uint32_t reccap = (uint32_t)(o->gc_recs_cap / sizeof(struct at_gc_rec));

        cl_event gevs[7] = {0};
        cl_event *gpev = o->prof ? gevs : NULL;
        if (!emit_only) {
            do_emit = 0;
            err |= clSetKernelArg(o->k_gc_sweep, 10, sizeof(do_emit), &do_emit);
            err |= clSetKernelArg(o->k_gc_sweep, 18, sizeof(reccap), &reccap);
            err |= clEnqueueNDRangeKernel(o->q, o->k_gc_sweep, 1, NULL, &gruns, &l256, 0, NULL, gpev ? &gevs[0] : NULL);
            err |= clEnqueueNDRangeKernel(o->q, o->k_scan_blocks, 1, NULL, &gruns, &l256, 0, NULL, gpev ? &gevs[1] : NULL);
            err |= clEnqueueNDRangeKernel(o->q, o->k_scan_tops, 1, NULL, &g1, &l256, 0, NULL, gpev ? &gevs[2] : NULL);
            err |= clEnqueueNDRangeKernel(o->q, o->k_scan_add, 1, NULL, &gruns, &l256, 0, NULL, gpev ? &gevs[3] : NULL);
        }
        err |= clEnqueueNDRangeKernel(o->q, o->k_ht_clear, 1, NULL, &ght, NULL, 0, NULL, gpev ? &gevs[4] : NULL);

        do_emit = 1;
        err |= clSetKernelArg(o->k_gc_sweep, 10, sizeof(do_emit), &do_emit);
        err |= o->set_arg_ptr(o->k_gc_sweep, 13, o->gc_recs);
        err |= clSetKernelArg(o->k_gc_sweep, 18, sizeof(reccap), &reccap);
        err |= clEnqueueNDRangeKernel(o->q, o->k_gc_sweep, 1, NULL, &gruns, &l256, 0, NULL, gpev ? &gevs[5] : NULL);
        err |= clEnqueueNDRangeKernel(o->q, o->k_gc_compact, 1, NULL, &ght, NULL, 0, NULL, gpev ? &gevs[6] : NULL);

        double tb = o->prof ? now_ms() : 0;
        if (err != CL_SUCCESS || clFinish(o->q) != CL_SUCCESS)
            return false;
        if (o->prof) {
            fprintf(stderr, "  ocl gc batch %.3f ms, %u recs, %u clusters%s\n",
                    now_ms() - tb, o->flags[2], o->flags[3],
                    o->flags[5] ? " (rec overflow, retrying)" : "");
            prof_event(o, "gc count", gevs[0]);
            prof_event(o, "scan_blocks", gevs[1]);
            prof_event(o, "scan_tops", gevs[2]);
            prof_event(o, "scan_add", gevs[3]);
            prof_event(o, "ht_clear", gevs[4]);
            prof_event(o, "gc emit", gevs[5]);
            prof_event(o, "gc compact", gevs[6]);
        }
        if (o->flags[4])
            return false; // hash table or directory overflow
        if (!o->flags[5])
            break;
        if (attempt == 1)
            return false;
        // grow to the exact total and redo the emit half
        o->gc_recs = usm_grow(o, o->gc_recs, &o->gc_recs_cap,
                              (size_t)o->flags[2] * sizeof(struct at_gc_rec));
        if (!o->gc_recs)
            return false;
        emit_only = 1;
    }

    out->recs = (const struct at_gc_rec *)o->gc_recs;
    out->nrecs = o->flags[2];
    out->probe_dense = o->gc_pd;
    out->dir_ids = o->gc_dir;
    out->nclusters = o->flags[3];
    return true;
}

bool at_ocl_ensure_uf(at_ocl_t *o)
{
    if (o->uf_published)
        return true;
    if (!(o->frame_state & 2))
        return false;
    uint32_t maxid = o->uf_pub.maxid;
    size_t gnodes = ((size_t)maxid + 63) / 64 * 64;
    cl_int err = CL_SUCCESS;
    err |= o->set_arg_ptr(o->k_ccl_publish, 0, o->d_parent);
    err |= o->set_arg_ptr(o->k_ccl_publish, 1, o->d_acc);
    err |= o->set_arg_ptr(o->k_ccl_publish, 2, o->uf_parent);
    err |= o->set_arg_ptr(o->k_ccl_publish, 3, o->uf_size);
    err |= clSetKernelArg(o->k_ccl_publish, 4, sizeof(maxid), &maxid);
    err |= clEnqueueNDRangeKernel(o->q, o->k_ccl_publish, 1, NULL, &gnodes, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS || clFinish(o->q) != CL_SUCCESS)
        return false;
    o->uf_published = true;
    return true;
}

bool at_ocl_quad_prepare(at_ocl_t *o, uint32_t ncl, uint32_t total_pts,
                         struct pt **pts_slab, uint32_t **coff, uint32_t **csz)
{
    size_t n1 = ncl ? ncl : 1, np = total_pts ? total_pts : 1;
    o->q_pts = usm_grow(o, o->q_pts, &o->q_pts_cap, np * sizeof(struct pt));
    o->q_coff = usm_grow(o, o->q_coff, &o->q_coff_cap, n1 * 4);
    o->q_csz = usm_grow(o, o->q_csz, &o->q_csz_cap, n1 * 4);
    if (!o->q_pts || !o->q_coff || !o->q_csz)
        return false;
    *pts_slab = o->q_pts;
    *coff = o->q_coff;
    *csz = o->q_csz;
    return true;
}

const struct at_quad_out *at_ocl_fit_quads(at_ocl_t *o, apriltag_detector_t *td,
                                           uint32_t ncl, int w, int h,
                                           int min_cluster_pixels, int tag_width,
                                           bool normal_border, bool reversed_border)
{
    if (!(o->frame_state & 1) || o->frame_w != w || o->frame_h != h)
        return NULL; // need this frame's image in device memory
    if (td->qtp.max_nmaxima > 16)
        return NULL; // memo slabs are sized for the default cap

    size_t n1 = ncl ? ncl : 1;
    o->q_eligscr = usm_grow(o, o->q_eligscr, &o->q_eligscr_cap, n1 * 4);
    o->q_eligord = usm_grow(o, o->q_eligord, &o->q_eligord_cap, n1 * 4);
    o->q_out = usm_grow(o, o->q_out, &o->q_out_cap, n1 * sizeof(struct at_quad_out));
    if (!o->q_eligscr || !o->q_eligord || !o->q_out)
        return NULL;

    // eligibility (the do_quad_task gates) and scratch layout
    int maxsz = 2 * (2*w + 2*h);
    const char *cutenv = getenv("APRILTAG_GPU_QUADS_CUT");
    int cut = cutenv ? atoi(cutenv) : 0; // size-split experiment
    if (cut && cut < maxsz)
        maxsz = cut;
    uint64_t E = 0;
    uint32_t nelig = 0;
    for (uint32_t ci = 0; ci < ncl; ci++) {
        uint32_t sz = o->q_csz[ci];
        if ((int)sz >= min_cluster_pixels && (int)sz <= maxsz) {
            o->q_eligscr[ci] = (uint32_t)E;
            o->q_eligord[ci] = nelig++;
            E += sz;
        } else {
            o->q_eligscr[ci] = 0xffffffff;
            o->q_eligord[ci] = 0;
        }
    }
    if (E == 0) {
        memset(o->q_out, 0, ncl * sizeof(struct at_quad_out));
        return (const struct at_quad_out *)o->q_out;
    }
    if (E > 0xffffffffu)
        return NULL;

    size_t Ep = E;
    o->q_keys = usm_grow_dev(o, o->q_keys, &o->q_keys_cap, Ep * 8);
    o->q_tmp = usm_grow_dev(o, o->q_tmp, &o->q_tmp_cap, Ep * 8);
    o->q_lf = usm_grow_dev(o, o->q_lf, &o->q_lf_cap, Ep * 6 * 8);
    o->q_errs = usm_grow_dev(o, o->q_errs, &o->q_errs_cap, Ep * 8);
    o->q_yfilt = usm_grow_dev(o, o->q_yfilt, &o->q_yfilt_cap, Ep * 8);
    o->q_maxima = usm_grow_dev(o, o->q_maxima, &o->q_maxima_cap, Ep * 4);
    o->q_maxerrs = usm_grow_dev(o, o->q_maxerrs, &o->q_maxerrs_cap, Ep * 8);
    o->q_memo = usm_grow_dev(o, o->q_memo, &o->q_memo_cap, (size_t)nelig * 16 * 16 * 64);
    o->q_qsm = usm_grow(o, o->q_qsm, &o->q_qsm_cap, 7 * sizeof(double));
    if (!o->q_keys || !o->q_tmp || !o->q_lf || !o->q_errs || !o->q_yfilt ||
        !o->q_maxima || !o->q_maxerrs || !o->q_memo || !o->q_qsm)
        return NULL;

    // the CPU stores the Gaussian taps as floats; match that rounding
    for (int i = 0; i < 7; i++) {
        int j = i - 3;
        o->q_qsm[i] = (double)(float)exp(-j*j/2.0);
    }

    // lf arrays are laid out as 6 slabs of lf_stride doubles
    cl_ulong lf_stride = (cl_ulong)(o->q_lf_cap / (6 * 8));
    cl_int nb = normal_border, rb = reversed_border;
    cl_int mn = td->qtp.max_nmaxima;
    cl_double mlfm = td->qtp.max_line_fit_mse;
    cl_double mdot = td->qtp.cos_critical_rad;

    cl_int err = CL_SUCCESS;
    err |= o->set_arg_ptr(o->k_fit_quads, 0, o->q_pts);
    err |= o->set_arg_ptr(o->k_fit_quads, 1, o->q_coff);
    err |= o->set_arg_ptr(o->k_fit_quads, 2, o->q_csz);
    err |= clSetKernelArg(o->k_fit_quads, 3, sizeof(ncl), &ncl);
    err |= o->set_arg_ptr(o->k_fit_quads, 4, o->q_eligscr);
    err |= o->set_arg_ptr(o->k_fit_quads, 5, o->q_eligord);
    err |= o->set_arg_ptr(o->k_fit_quads, 6, o->img);
    err |= clSetKernelArg(o->k_fit_quads, 7, sizeof(w), &w);
    err |= clSetKernelArg(o->k_fit_quads, 8, sizeof(h), &h);
    err |= clSetKernelArg(o->k_fit_quads, 9, sizeof(o->frame_s), &o->frame_s);
    err |= clSetKernelArg(o->k_fit_quads, 10, sizeof(tag_width), &tag_width);
    err |= clSetKernelArg(o->k_fit_quads, 11, sizeof(nb), &nb);
    err |= clSetKernelArg(o->k_fit_quads, 12, sizeof(rb), &rb);
    err |= clSetKernelArg(o->k_fit_quads, 13, sizeof(mn), &mn);
    err |= clSetKernelArg(o->k_fit_quads, 14, sizeof(mlfm), &mlfm);
    err |= clSetKernelArg(o->k_fit_quads, 15, sizeof(mdot), &mdot);
    err |= o->set_arg_ptr(o->k_fit_quads, 16, o->q_qsm);
    err |= o->set_arg_ptr(o->k_fit_quads, 17, o->q_keys);
    err |= o->set_arg_ptr(o->k_fit_quads, 18, o->q_tmp);
    err |= o->set_arg_ptr(o->k_fit_quads, 19, o->q_lf);
    err |= clSetKernelArg(o->k_fit_quads, 20, sizeof(lf_stride), &lf_stride);
    err |= o->set_arg_ptr(o->k_fit_quads, 21, o->q_errs);
    err |= o->set_arg_ptr(o->k_fit_quads, 22, o->q_yfilt);
    err |= o->set_arg_ptr(o->k_fit_quads, 23, o->q_maxima);
    err |= o->set_arg_ptr(o->k_fit_quads, 24, o->q_maxerrs);
    err |= o->set_arg_ptr(o->k_fit_quads, 25, o->q_memo);
    err |= o->set_arg_ptr(o->k_fit_quads, 26, o->q_out);

    size_t g = ((size_t)ncl + 63) / 64 * 64;
    err |= clEnqueueNDRangeKernel(o->q, o->k_fit_quads, 1, NULL, &g, NULL, 0, NULL, NULL);

    double t0 = o->prof ? now_ms() : 0;
    if (err != CL_SUCCESS || clFinish(o->q) != CL_SUCCESS)
        return NULL;
    if (o->prof)
        fprintf(stderr, "  ocl fit_quads %.3f ms (%u clusters, %u eligible, %llu pts)\n",
                now_ms() - t0, ncl, nelig, (unsigned long long)E);

    return (const struct at_quad_out *)o->q_out;
}
