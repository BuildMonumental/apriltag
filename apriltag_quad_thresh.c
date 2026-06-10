/* Copyright (C) 2013-2016, The Regents of The University of Michigan.
All rights reserved.
This software was developed in the APRIL Robotics Lab under the
direction of Edwin Olson, ebolson@umich.edu. This software may be
available under alternative licensing terms; contact the address above.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of the Regents of The University of Michigan.
*/

// limitation: image size must be <32768 in width and height. This is
// because we use a fixed-point 16 bit integer representation with one
// fractional bit.
#define _USE_MATH_DEFINES
#include <math.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>

#include "apriltag.h"
#include "common/image_u8x3.h"
#include "common/zarray.h"
#include "common/unionfind.h"
#include "common/timeprofile.h"
#include "common/zmaxheap.h"
#include "common/postscript_utils.h"
#include "common/math_util.h"

#ifdef _WIN32
static inline long int random(void)
{
        return rand();
}
#endif

static inline uint32_t u64hash_2(uint64_t x) {
    return (2654435761 * x) >> 32;
}

struct uint64_zarray_entry
{
    uint64_t id;
    struct gc_chunk *head, *tail;
    int npts;

    struct uint64_zarray_entry *next;
};

struct pt
{
    // Note: these represent 2*actual value.
    uint16_t x, y;
    int16_t gx, gy;

    float slope;
};

// Cluster points are accumulated in fixed-size chunks bump-allocated from
// a per-task pool: appending is a bounds check and a store, with none of
// the doubling reallocs a growing array needs (a frame can produce
// millions of points). Clusters are materialized into exact-size zarrays
// once their final length is known.
#define GC_CHUNK_PTS 32
struct gc_chunk
{
    struct gc_chunk *next;
    int count;
    struct pt pts[GC_CHUNK_PTS];
};

struct gc_chunk_pool
{
    struct gc_chunk **blocks;
    int nblocks;
    int cap_blocks;
    int used_in_block; // chunks handed out from the newest block
    int chunks_per_block;
};

static void gc_chunk_pool_init(struct gc_chunk_pool *pool)
{
    pool->cap_blocks = 16;
    pool->blocks = malloc(sizeof(struct gc_chunk *)*pool->cap_blocks);
    pool->chunks_per_block = 1024;
    pool->blocks[0] = malloc(sizeof(struct gc_chunk)*pool->chunks_per_block);
    pool->nblocks = 1;
    pool->used_in_block = 0;
}

static struct gc_chunk *gc_chunk_alloc(struct gc_chunk_pool *pool)
{
    if (pool->used_in_block == pool->chunks_per_block) {
        if (pool->nblocks == pool->cap_blocks) {
            pool->cap_blocks *= 2;
            pool->blocks = realloc(pool->blocks, sizeof(struct gc_chunk *)*pool->cap_blocks);
        }
        pool->blocks[pool->nblocks++] = malloc(sizeof(struct gc_chunk)*pool->chunks_per_block);
        pool->used_in_block = 0;
    }
    struct gc_chunk *c = &pool->blocks[pool->nblocks - 1][pool->used_in_block++];
    c->next = NULL;
    c->count = 0;
    return c;
}

static void gc_chunk_pool_free(struct gc_chunk_pool *pool)
{
    for (int i = 0; i < pool->nblocks; i++)
        free(pool->blocks[i]);
    free(pool->blocks);
}

struct unionfind_task
{
    int y0, y1;
    int w, h, s;
    unionfind_t *uf;
    image_u8_t *im;
};

struct quad_task
{
    zarray_t *clusters;
    int cidx0, cidx1; // [cidx0, cidx1)
    zarray_t *quads;
    apriltag_detector_t *td;
    int w, h;

    image_u8_t *im;
    int tag_width;
    bool normal_border;
    bool reversed_border;
};


struct cluster_task
{
    int y0;
    int y1;
    int w;
    int s;
    int nclustermap;
    int min_cluster_pixels;
    unionfind_t* uf;
    image_u8_t* im;
    zarray_t* clusters;
};

struct minmax_task {
    int ty;

    image_u8_t *im;
    uint8_t *im_max;
    uint8_t *im_min;
};

struct blur_task {
    int ty;

    image_u8_t *im;
    uint8_t *im_max;
    uint8_t *im_min;
    uint8_t *im_max_tmp;
    uint8_t *im_min_tmp;
};

struct threshold_task {
    int ty;

    apriltag_detector_t *td;
    image_u8_t *im;
    image_u8_t *threshim;
    uint8_t *im_max;
    uint8_t *im_min;
};

struct remove_vertex
{
    int i;           // which vertex to remove?
    int left, right; // left vertex, right vertex

    double err;
};

struct segment
{
    int is_vertex;

    // always greater than zero, but right can be > size, which denotes
    // a wrap around back to the beginning of the points. and left < right.
    int left, right;
};

struct line_fit_pt
{
    double Mx, My;
    double Mxx, Myy, Mxy;
    double W; // total weight
};

struct cluster_hash
{
    uint32_t hash;
    uint64_t id;
    zarray_t* data;
};

// scratch buffers reused across all clusters processed by one quad task,
// so fit_quad doesn't malloc/free per cluster.
struct quad_fit_scratch
{
    int capacity; // in points
    struct line_fit_pt *lfps;
    double *errs;
    double *yfilt;
    int *maxima;
    double *maxima_errs;
    struct pt *pt_tmp;
    uint64_t *sort_keys;
    uint64_t *sort_tmp;
};

static void quad_fit_scratch_ensure(struct quad_fit_scratch *scratch, int sz)
{
    if (sz <= scratch->capacity)
        return;
    int cap = scratch->capacity ? 2*scratch->capacity : 1024;
    if (cap < sz)
        cap = sz;
    free(scratch->lfps);
    free(scratch->errs);
    free(scratch->yfilt);
    free(scratch->maxima);
    free(scratch->maxima_errs);
    free(scratch->pt_tmp);
    free(scratch->sort_keys);
    free(scratch->sort_tmp);
    scratch->lfps = malloc(sizeof(struct line_fit_pt)*cap);
    scratch->errs = malloc(sizeof(double)*cap);
    scratch->yfilt = malloc(sizeof(double)*cap);
    scratch->maxima = malloc(sizeof(int)*cap);
    scratch->maxima_errs = malloc(sizeof(double)*cap);
    scratch->pt_tmp = malloc(sizeof(struct pt)*cap);
    scratch->sort_keys = malloc(sizeof(uint64_t)*cap);
    scratch->sort_tmp = malloc(sizeof(uint64_t)*cap);
    scratch->capacity = cap;
}

static void quad_fit_scratch_free(struct quad_fit_scratch *scratch)
{
    free(scratch->lfps);
    free(scratch->errs);
    free(scratch->yfilt);
    free(scratch->maxima);
    free(scratch->maxima_errs);
    free(scratch->pt_tmp);
    free(scratch->sort_keys);
    free(scratch->sort_tmp);
}


// lfps contains *cumulative* moments for N points, with
// index j reflecting points [0,j] (inclusive).
//
// fit a line to the points [i0, i1] (inclusive). i0, i1 are both [0,
// sz) if i1 < i0, we treat this as a wrap around.
void fit_line(struct line_fit_pt *lfps, int sz, int i0, int i1, double *lineparm, double *err, double *mse)
{
    assert(i0 != i1);
    assert(i0 >= 0 && i1 >= 0 && i0 < sz && i1 < sz);

    double Mx, My, Mxx, Myy, Mxy, W;
    int N; // how many points are included in the set?

    if (i0 < i1) {
        N = i1 - i0 + 1;

        Mx  = lfps[i1].Mx;
        My  = lfps[i1].My;
        Mxx = lfps[i1].Mxx;
        Mxy = lfps[i1].Mxy;
        Myy = lfps[i1].Myy;
        W   = lfps[i1].W;

        if (i0 > 0) {
            Mx  -= lfps[i0-1].Mx;
            My  -= lfps[i0-1].My;
            Mxx -= lfps[i0-1].Mxx;
            Mxy -= lfps[i0-1].Mxy;
            Myy -= lfps[i0-1].Myy;
            W   -= lfps[i0-1].W;
        }

    } else {
        // i0 > i1, e.g. [15, 2]. Wrap around.
        assert(i0 > 0);

        Mx  = lfps[sz-1].Mx   - lfps[i0-1].Mx;
        My  = lfps[sz-1].My   - lfps[i0-1].My;
        Mxx = lfps[sz-1].Mxx  - lfps[i0-1].Mxx;
        Mxy = lfps[sz-1].Mxy  - lfps[i0-1].Mxy;
        Myy = lfps[sz-1].Myy  - lfps[i0-1].Myy;
        W   = lfps[sz-1].W    - lfps[i0-1].W;

        Mx  += lfps[i1].Mx;
        My  += lfps[i1].My;
        Mxx += lfps[i1].Mxx;
        Mxy += lfps[i1].Mxy;
        Myy += lfps[i1].Myy;
        W   += lfps[i1].W;

        N = sz - i0 + i1 + 1;
    }

    assert(N >= 2);

    double Ex = Mx / W;
    double Ey = My / W;
    double Cxx = Mxx / W - Ex*Ex;
    double Cxy = Mxy / W - Ex*Ey;
    double Cyy = Myy / W - Ey*Ey;

    //if (1) {
    //    // on iOS about 5% of total CPU spent in these trig functions.
    //    // 85 ms per frame on 5S, example.pnm
    //    //
    //    // XXX this was using the double-precision atan2. Was there a case where
    //    // we needed that precision? Seems doubtful.
    //    double normal_theta = .5 * atan2f(-2*Cxy, (Cyy - Cxx));
    //    nx_old = cosf(normal_theta);
    //    ny_old = sinf(normal_theta);
    //}

    // Instead of using the above cos/sin method, pose it as an eigenvalue problem.
    double eig_small = 0.5*(Cxx + Cyy - sqrtf((Cxx - Cyy)*(Cxx - Cyy) + 4*Cxy*Cxy));

    if (lineparm) {
        lineparm[0] = Ex;
        lineparm[1] = Ey;

        double eig = 0.5*(Cxx + Cyy + sqrtf((Cxx - Cyy)*(Cxx - Cyy) + 4*Cxy*Cxy));
        double nx1 = Cxx - eig;
        double ny1 = Cxy;
        double M1 = nx1*nx1 + ny1*ny1;
        double nx2 = Cxy;
        double ny2 = Cyy - eig;
        double M2 = nx2*nx2 + ny2*ny2;

        double nx, ny, M;
        if (M1 > M2) {
            nx = nx1;
            ny = ny1;
            M = M1;
        } else {
            nx = nx2;
            ny = ny2;
            M = M2;
        }

        double length = sqrtf(M);
        if (fabs(length) < 1e-12) {
            lineparm[2] = lineparm[3] = 0;
        }
        else {
            lineparm[2] = nx/length;
            lineparm[3] = ny/length;
        }
    }

    // sum of squared errors =
    //
    // SUM_i ((p_x - ux)*nx + (p_y - uy)*ny)^2
    // SUM_i  nx*nx*(p_x - ux)^2 + 2nx*ny(p_x -ux)(p_y-uy) + ny*ny*(p_y-uy)*(p_y-uy)
    //  nx*nx*SUM_i((p_x -ux)^2) + 2nx*ny*SUM_i((p_x-ux)(p_y-uy)) + ny*ny*SUM_i((p_y-uy)^2)
    //
    //  nx*nx*N*Cxx + 2nx*ny*N*Cxy + ny*ny*N*Cyy

    // sum of squared errors
    if (err)
        *err = N*eig_small;

    // mean squared error
    if (mse)
        *mse = eig_small;
}

static inline float pt_compare_angle(struct pt *a, struct pt *b) {
    return a->slope - b->slope;
}

int err_compare_descending(const void *_a, const void *_b)
{
    const double *a =  _a;
    const double *b =  _b;

    return ((*a) < (*b)) ? 1 : -1;
}

/*

  1. Identify A) white points near a black point and B) black points near a white point.

  2. Find the connected components within each of the classes above,
  yielding clusters of "white-near-black" and
  "black-near-white". (These two classes are kept separate). Each
  segment has a unique id.

  3. For every pair of "white-near-black" and "black-near-white"
  clusters, find the set of points that are in one and adjacent to the
  other. In other words, a "boundary" layer between the two
  clusters. (This is actually performed by iterating over the pixels,
  rather than pairs of clusters.) Critically, this helps keep nearby
  edges from becoming connected.
*/
// memoized fit_line over pairs of maxima; the candidate-quad search asks
// for the same segment fit many times across its nested loops.
struct pair_fit
{
    double err, mse;
    double params[4];
    bool computed;
};

static inline struct pair_fit *pair_fit_get(struct line_fit_pt *lfps, int sz, int *maxima, int nmaxima,
                                            struct pair_fit *memo, int ma, int mb)
{
    struct pair_fit *pf = &memo[ma*nmaxima + mb];
    if (!pf->computed) {
        fit_line(lfps, sz, maxima[ma], maxima[mb], pf->params, &pf->err, &pf->mse);
        pf->computed = true;
    }
    return pf;
}

// Gaussian low-pass kernel for the per-point fit errors. sigma = 1,
// cutoff = 0.05 give a fixed size of 7; values match
// exp(-j*j/(2*sigma*sigma)) for j in [-3, 3].
#define QSM_FSZ 7
static __thread float qsm_kernel[QSM_FSZ];
static __thread bool qsm_kernel_init;

int quad_segment_maxima(apriltag_detector_t *td, zarray_t *cluster, struct line_fit_pt *lfps, int indices[4],
                        struct quad_fit_scratch *scratch)
{
    int sz = zarray_size(cluster);

    // ksz: when fitting points, how many points on either side do we consider?
    // (actual "kernel" width is 2ksz).
    //
    // This value should be about: 0.5 * (points along shortest edge).
    //
    // If all edges were equally-sized, that would give a value of
    // sz/8. We make it somewhat smaller to account for tags at high
    // aspects.

    // XXX Tunable. Maybe make a multiple of JPEG block size to increase robustness
    // to JPEG compression artifacts?
    int ksz = imin(20, sz / 12);

    // can't fit a quad if there are too few points.
    if (ksz < 2)
        return 0;

    double *errs = scratch->errs;

    // windows that wrap around the ends (or start exactly at 0) go through
    // the general fit_line; the bulk of the windows take the lean inline
    // path below, with identical arithmetic
    int mid_lo = ksz + 1;
    int mid_hi = sz - ksz - 1;

    for (int i = 0; i < mid_lo; i++) {
        int i0 = i - ksz;
        if (i0 < 0)
            i0 += sz;
        fit_line(lfps, sz, i0, i + ksz, NULL, &errs[i], NULL);
    }

    int N = 2*ksz + 1;
    for (int i = mid_lo; i <= mid_hi; i++) {
        const struct line_fit_pt *l1 = &lfps[i + ksz];
        const struct line_fit_pt *l0 = &lfps[i - ksz - 1];

        double Mx  = l1->Mx  - l0->Mx;
        double My  = l1->My  - l0->My;
        double Mxx = l1->Mxx - l0->Mxx;
        double Mxy = l1->Mxy - l0->Mxy;
        double Myy = l1->Myy - l0->Myy;
        double W   = l1->W   - l0->W;

        double Ex = Mx / W;
        double Ey = My / W;
        double Cxx = Mxx / W - Ex*Ex;
        double Cxy = Mxy / W - Ex*Ey;
        double Cyy = Myy / W - Ey*Ey;

        double eig_small = 0.5*(Cxx + Cyy - sqrtf((Cxx - Cyy)*(Cxx - Cyy) + 4*Cxy*Cxy));
        errs[i] = N*eig_small;
    }

    for (int i = mid_hi + 1; i < sz; i++) {
        int i1 = i + ksz;
        if (i1 >= sz)
            i1 -= sz;
        fit_line(lfps, sz, i - ksz, i1, NULL, &errs[i], NULL);
    }

    // apply a low-pass filter to errs
    if (1) {
        double *y = scratch->yfilt;

        if (!qsm_kernel_init) {
            double sigma = 1; // was 3
            double cutoff = 0.05;
            int fsz = sqrt(-log(cutoff)*2*sigma*sigma) + 1;
            fsz = 2*fsz + 1;
            assert(fsz == QSM_FSZ);
            for (int i = 0; i < fsz; i++) {
                int j = i - fsz / 2;
                qsm_kernel[i] = exp(-j*j/(2*sigma*sigma));
            }
            qsm_kernel_init = true;
        }

        // sz >= 12*ksz >= 24 > QSM_FSZ, so single wrap adjustments suffice
        for (int iy = 0; iy < sz; iy++) {
            double acc = 0;

            int j = iy - QSM_FSZ / 2;
            if (j < 0)
                j += sz;
            for (int i = 0; i < QSM_FSZ; i++) {
                acc += errs[j] * qsm_kernel[i];
                if (++j == sz)
                    j = 0;
            }
            y[iy] = acc;
        }

        memcpy(errs, y, sizeof(double)*sz);
    }

    int *maxima = scratch->maxima;
    double *maxima_errs = scratch->maxima_errs;
    int nmaxima = 0;

    for (int i = 0; i < sz; i++) {
        double e = errs[i];
        if (e > errs[i + 1 == sz ? 0 : i + 1] && e > errs[i == 0 ? sz - 1 : i - 1]) {
            maxima[nmaxima] = i;
            maxima_errs[nmaxima] = e;
            nmaxima++;
        }
    }

    // if we didn't get at least 4 maxima, we can't fit a quad.
    if (nmaxima < 4)
        return 0;

    // select only the best maxima if we have too many
    int max_nmaxima = td->qtp.max_nmaxima;

    if (nmaxima > max_nmaxima) {
        double *maxima_errs_copy = malloc(sizeof(double)*nmaxima);
        memcpy(maxima_errs_copy, maxima_errs, sizeof(double)*nmaxima);

        // throw out all but the best handful of maxima. Sorts descending.
        qsort(maxima_errs_copy, nmaxima, sizeof(double), err_compare_descending);

        double maxima_thresh = maxima_errs_copy[max_nmaxima];
        int out = 0;
        for (int in = 0; in < nmaxima; in++) {
            if (maxima_errs[in] <= maxima_thresh)
                continue;
            maxima[out++] = maxima[in];
        }
        nmaxima = out;
        free(maxima_errs_copy);
    }

    int best_indices[4];
    double best_error = HUGE_VALF;

    // disallow quads where the angle is less than a critical value.
    double max_dot = td->qtp.cos_critical_rad; //25*M_PI/180);

    double max_line_fit_mse = td->qtp.max_line_fit_mse;

    struct pair_fit memo_stack[16*16];
    struct pair_fit *memo = memo_stack;
    if (nmaxima > 16)
        memo = malloc(sizeof(struct pair_fit)*nmaxima*nmaxima);
    for (int i = 0; i < nmaxima*nmaxima; i++)
        memo[i].computed = false;

    for (int m0 = 0; m0 < nmaxima - 3; m0++) {
        int i0 = maxima[m0];

        for (int m1 = m0+1; m1 < nmaxima - 2; m1++) {
            struct pair_fit *pf01 = pair_fit_get(lfps, sz, maxima, nmaxima, memo, m0, m1);

            if (pf01->mse > max_line_fit_mse)
                continue;

            for (int m2 = m1+1; m2 < nmaxima - 1; m2++) {
                struct pair_fit *pf12 = pair_fit_get(lfps, sz, maxima, nmaxima, memo, m1, m2);
                if (pf12->mse > max_line_fit_mse)
                    continue;

                double dot = pf01->params[2]*pf12->params[2] + pf01->params[3]*pf12->params[3];
                if (fabs(dot) > max_dot)
                    continue;

                for (int m3 = m2+1; m3 < nmaxima; m3++) {
                    struct pair_fit *pf23 = pair_fit_get(lfps, sz, maxima, nmaxima, memo, m2, m3);
                    if (pf23->mse > max_line_fit_mse)
                        continue;

                    struct pair_fit *pf30 = pair_fit_get(lfps, sz, maxima, nmaxima, memo, m3, m0);
                    if (pf30->mse > max_line_fit_mse)
                        continue;

                    double err = pf01->err + pf12->err + pf23->err + pf30->err;
                    if (err < best_error) {
                        best_error = err;
                        best_indices[0] = i0;
                        best_indices[1] = maxima[m1];
                        best_indices[2] = maxima[m2];
                        best_indices[3] = maxima[m3];
                    }
                }
            }
        }
    }

    if (memo != memo_stack)
        free(memo);

    if (best_error == HUGE_VALF)
        return 0;

    for (int i = 0; i < 4; i++)
        indices[i] = best_indices[i];

    if (best_error / sz < td->qtp.max_line_fit_mse)
        return 1;
    return 0;
}

// returns 0 if the cluster looks bad.
int quad_segment_agg(zarray_t *cluster, struct line_fit_pt *lfps, int indices[4])
{
    int sz = zarray_size(cluster);

    zmaxheap_t *heap = zmaxheap_create(sizeof(struct remove_vertex*));

    // We will initially allocate sz rvs. We then have two types of
    // iterations: some iterations that are no-ops in terms of
    // allocations, and those that remove a vertex and allocate two
    // more children.  This will happen at most (sz-4) times.  Thus we
    // need: sz + 2*(sz-4) entries.

    int rvalloc_pos = 0;
    int rvalloc_size = 3*sz;
    struct remove_vertex *rvalloc = calloc(rvalloc_size, sizeof(struct remove_vertex));

    struct segment *segs = calloc(sz, sizeof(struct segment));

    // populate with initial entries
    for (int i = 0; i < sz; i++) {
        struct remove_vertex *rv = &rvalloc[rvalloc_pos++];
        rv->i = i;
        if (i == 0) {
            rv->left = sz-1;
            rv->right = 1;
        } else {
            rv->left  = i-1;
            rv->right = (i+1) % sz;
        }

        fit_line(lfps, sz, rv->left, rv->right, NULL, NULL, &rv->err);

        zmaxheap_add(heap, &rv, -rv->err);

        segs[i].left = rv->left;
        segs[i].right = rv->right;
        segs[i].is_vertex = 1;
    }

    int nvertices = sz;

    while (nvertices > 4) {
        assert(rvalloc_pos < rvalloc_size);

        struct remove_vertex *rv;
        float err;

        int res = zmaxheap_remove_max(heap, &rv, &err);
        if (!res)
            return 0;
        assert(res);

        // is this remove_vertex valid? (Or has one of the left/right
        // vertices changes since we last looked?)
        if (!segs[rv->i].is_vertex ||
            !segs[rv->left].is_vertex ||
            !segs[rv->right].is_vertex) {
            continue;
        }

        // we now merge.
        assert(segs[rv->i].is_vertex);

        segs[rv->i].is_vertex = 0;
        segs[rv->left].right = rv->right;
        segs[rv->right].left = rv->left;

        // create the join to the left
        if (1) {
            struct remove_vertex *child = &rvalloc[rvalloc_pos++];
            child->i = rv->left;
            child->left = segs[rv->left].left;
            child->right = rv->right;

            fit_line(lfps, sz, child->left, child->right, NULL, NULL, &child->err);

            zmaxheap_add(heap, &child, -child->err);
        }

        // create the join to the right
        if (1) {
            struct remove_vertex *child = &rvalloc[rvalloc_pos++];
            child->i = rv->right;
            child->left = rv->left;
            child->right = segs[rv->right].right;

            fit_line(lfps, sz, child->left, child->right, NULL, NULL, &child->err);

            zmaxheap_add(heap, &child, -child->err);
        }

        // we now have one less vertex
        nvertices--;
    }

    free(rvalloc);
    zmaxheap_destroy(heap);

    int idx = 0;
    for (int i = 0; i < sz; i++) {
        if (segs[i].is_vertex) {
            indices[idx++] = i;
        }
    }

    free(segs);

    return 1;
}

/**
 * Compute statistics that allow line fit queries to be
 * efficiently computed for any contiguous range of indices.
 */
void compute_lfps(int sz, zarray_t* cluster, image_u8_t* im, struct line_fit_pt *lfps) {
    double sum_Mx = 0, sum_My = 0, sum_Mxx = 0, sum_Myy = 0, sum_Mxy = 0, sum_W = 0;

    for (int i = 0; i < sz; i++) {
        struct pt *p;
        zarray_get_volatile(cluster, i, &p);

        // we now undo our fixed-point arithmetic.
        double delta = 0.5; // adjust for pixel center bias
        double x = p->x * .5 + delta;
        double y = p->y * .5 + delta;
        int ix = x, iy = y;
        double W = 1;

        if (ix > 0 && ix+1 < im->width && iy > 0 && iy+1 < im->height) {
            int grad_x = im->buf[iy * im->stride + ix + 1] -
                im->buf[iy * im->stride + ix - 1];

            int grad_y = im->buf[(iy+1) * im->stride + ix] -
                im->buf[(iy-1) * im->stride + ix];

            // XXX Tunable. How to shape the gradient magnitude?
            W = sqrt(grad_x*grad_x + grad_y*grad_y) + 1;
        }

        double fx = x, fy = y;
        sum_Mx  += W * fx;
        sum_My  += W * fy;
        sum_Mxx += W * fx * fx;
        sum_Mxy += W * fx * fy;
        sum_Myy += W * fy * fy;
        sum_W   += W;
        
        // Store cumulative sums
        lfps[i].Mx = sum_Mx;
        lfps[i].My = sum_My;
        lfps[i].Mxx = sum_Mxx;
        lfps[i].Mxy = sum_Mxy;
        lfps[i].Myy = sum_Myy;
        lfps[i].W = sum_W;
    }
}

// Sorting networks for <= 5 points, identical to the historical ptsort
// base cases (ties are NOT swapped).
static inline void pt_network_sort(struct pt *pts, int sz)
{
#define MAYBE_SWAP(arr,apos,bpos)                                   \
    if (pt_compare_angle(&(arr[apos]), &(arr[bpos])) > 0) {                        \
        tmp = arr[apos]; arr[apos] = arr[bpos]; arr[bpos] = tmp;    \
    };

    if (sz <= 1)
        return;

    if (sz == 2) {
        struct pt tmp;
        MAYBE_SWAP(pts, 0, 1);
        return;
    }

    // NB: Using less-branch-intensive sorting networks here on the
    // hunch that it's better for performance.
    if (sz == 3) { // 3 element bubble sort is optimal
        struct pt tmp;
        MAYBE_SWAP(pts, 0, 1);
        MAYBE_SWAP(pts, 1, 2);
        MAYBE_SWAP(pts, 0, 1);
        return;
    }

    if (sz == 4) { // 4 element optimal sorting network.
        struct pt tmp;
        MAYBE_SWAP(pts, 0, 1); // sort each half, like a merge sort
        MAYBE_SWAP(pts, 2, 3);
        MAYBE_SWAP(pts, 0, 2); // minimum value is now at 0.
        MAYBE_SWAP(pts, 1, 3); // maximum value is now at end.
        MAYBE_SWAP(pts, 1, 2); // that only leaves the middle two.
        return;
    }

    // sz == 5: this 9-step swap is optimal for a sorting network, but
    // two steps slower than a generic sort.
    struct pt tmp;
    MAYBE_SWAP(pts, 0, 1); // sort each half (3+2), like a merge sort
    MAYBE_SWAP(pts, 3, 4);
    MAYBE_SWAP(pts, 1, 2);
    MAYBE_SWAP(pts, 0, 1);
    MAYBE_SWAP(pts, 0, 3); // minimum element now at 0
    MAYBE_SWAP(pts, 2, 4); // maximum element now at end
    MAYBE_SWAP(pts, 1, 2); // now resort the three elements 1-3.
    MAYBE_SWAP(pts, 2, 3);
    MAYBE_SWAP(pts, 1, 2);

#undef MAYBE_SWAP
}

// The slope sort runs on packed 64-bit keys:
//
//   key64 = (order-preserving bits of slope) << 32 | ~original_index
//
// The float-to-bits map is strictly monotone for the finite slopes produced
// by fit_quad, so key comparisons order exactly like slope comparisons, and
// equal slopes mean equal high words. The complemented index makes a full
// 64-bit merge comparison reproduce the historical merge's tie rule (ties
// take from the right-hand run: left-run elements always carry smaller
// original indices, hence larger complements). Leaf networks compare the
// high word only, matching the historical networks' no-swap-on-tie rule.
// The result is bit-identical to the original ptsort, but the sort moves
// 8-byte keys instead of 12-byte structs and compares without calls.
static inline uint32_t slope_sort_key(float slope)
{
    union { float f; uint32_t u; } u;
    u.f = slope;
    return u.u ^ ((uint32_t)((int32_t)u.u >> 31) | 0x80000000u);
}

// Sorting networks for <= 5 keys; same shapes as pt_network_sort, ties
// (equal high words) are not swapped.
static inline void key_network_sort(uint64_t *k, int sz)
{
#define MAYBE_SWAP(apos,bpos)                                       \
    if ((k[apos] >> 32) > (k[bpos] >> 32)) {                        \
        uint64_t tmp = k[apos]; k[apos] = k[bpos]; k[bpos] = tmp;   \
    };

    if (sz <= 1)
        return;

    if (sz == 2) {
        MAYBE_SWAP(0, 1);
        return;
    }

    if (sz == 3) {
        MAYBE_SWAP(0, 1);
        MAYBE_SWAP(1, 2);
        MAYBE_SWAP(0, 1);
        return;
    }

    if (sz == 4) {
        MAYBE_SWAP(0, 1);
        MAYBE_SWAP(2, 3);
        MAYBE_SWAP(0, 2);
        MAYBE_SWAP(1, 3);
        MAYBE_SWAP(1, 2);
        return;
    }

    MAYBE_SWAP(0, 1);
    MAYBE_SWAP(3, 4);
    MAYBE_SWAP(1, 2);
    MAYBE_SWAP(0, 1);
    MAYBE_SWAP(0, 3);
    MAYBE_SWAP(2, 4);
    MAYBE_SWAP(1, 2);
    MAYBE_SWAP(2, 3);
    MAYBE_SWAP(1, 2);

#undef MAYBE_SWAP
}

static inline void key_merge(uint64_t *as, int asz, uint64_t *bs, int bsz, uint64_t *out)
{
    // branchless select: merge comparisons are data-dependent coin flips,
    // so conditional moves beat 50%-mispredicted branches
    #define MERGE(apos,bpos)                            \
    do {                                                \
        uint64_t av = as[apos], bv = bs[bpos];          \
        int take_a = av < bv;                           \
        out[outpos++] = take_a ? av : bv;               \
        apos += take_a;                                 \
        bpos += !take_a;                                \
    } while (0)

    int apos = 0, bpos = 0, outpos = 0;
    while (apos + 8 < asz && bpos + 8 < bsz) {
        MERGE(apos,bpos); MERGE(apos,bpos); MERGE(apos,bpos); MERGE(apos,bpos);
        MERGE(apos,bpos); MERGE(apos,bpos); MERGE(apos,bpos); MERGE(apos,bpos);
    }

    while (apos < asz && bpos < bsz) {
        MERGE(apos,bpos);
    }

    if (apos < asz)
        memcpy(&out[outpos], &as[apos], (asz-apos)*sizeof(uint64_t));
    if (bpos < bsz)
        memcpy(&out[outpos], &bs[bpos], (bsz-bpos)*sizeof(uint64_t));

#undef MERGE
}

// Ping-pong merge sort: same splits, leaf networks, and merge comparisons
// as the historical copy-per-level ptsort; data only copied at the leaves.
static void keysort_move(uint64_t *A, uint64_t *B, int sz);

// sort A in place, using tmp (>= sz entries) as scratch
static void keysort_in_place(uint64_t *A, uint64_t *tmp, int sz)
{
    if (sz <= 5) {
        key_network_sort(A, sz);
        return;
    }

    int asz = sz/2;
    int bsz = sz - asz;
    keysort_move(A, tmp, asz);
    keysort_move(A + asz, tmp + asz, bsz);
    key_merge(tmp, asz, tmp + asz, bsz, A);
}

// sort A's contents into B (A is clobbered)
static void keysort_move(uint64_t *A, uint64_t *B, int sz)
{
    if (sz <= 5) {
        key_network_sort(A, sz);
        memcpy(B, A, sz*sizeof(uint64_t));
        return;
    }

    int asz = sz/2;
    int bsz = sz - asz;
    keysort_in_place(A, B, asz);
    keysort_in_place(A + asz, B + asz, bsz);
    key_merge(A, asz, A + asz, bsz, B);
}

static void pt_slope_sort(struct pt *pts, int sz, struct quad_fit_scratch *scratch)
{
    if (sz <= 5) {
        pt_network_sort(pts, sz);
        return;
    }

    uint64_t *keys = scratch->sort_keys;
    for (int i = 0; i < sz; i++)
        keys[i] = ((uint64_t)slope_sort_key(pts[i].slope) << 32) | (uint32_t)~(uint32_t)i;

    keysort_in_place(keys, scratch->sort_tmp, sz);

    struct pt *tmp = scratch->pt_tmp;
    for (int i = 0; i < sz; i++)
        tmp[i] = pts[~(uint32_t)keys[i]];
    memcpy(pts, tmp, sizeof(struct pt)*sz);
}

// return 1 if the quad looks okay, 0 if it should be discarded
int fit_quad(
        apriltag_detector_t *td,
        image_u8_t *im,
        zarray_t *cluster,
        struct quad *quad,
        int tag_width,
        bool normal_border,
        bool reversed_border,
        struct quad_fit_scratch *scratch) {
    int res = 0;

    /////////////////////////////////////////////////////////////
    // Step 1. Sort points so they wrap around the center of the
    // quad. We will constrain our quad fit to simply partition this
    // ordered set into 4 groups.

    // compute a bounding box so that we can order the points
    // according to their angle WRT the center.
    struct pt *p1;
    zarray_get_volatile(cluster, 0, &p1);
    uint16_t xmax = p1->x;
    uint16_t xmin = p1->x;
    uint16_t ymax = p1->y;
    uint16_t ymin = p1->y;
    for (int pidx = 1; pidx < zarray_size(cluster); pidx++) {
        struct pt *p;
        zarray_get_volatile(cluster, pidx, &p);

        if (p->x > xmax) {
            xmax = p->x;
        } else if (p->x < xmin) {
            xmin = p->x;
        }

        if (p->y > ymax) {
            ymax = p->y;
        } else if (p->y < ymin) {
            ymin = p->y;
        }
    }

    if ((xmax - xmin)*(ymax - ymin) < tag_width) {
        return 0;
    }

    // add some noise to (cx,cy) so that pixels get a more diverse set
    // of theta estimates. This will help us remove more points.
    // (Only helps a small amount. The actual noise values here don't
    // matter much at all, but we want them [-1, 1]. (XXX with
    // fixed-point, should range be bigger?)
    float cx = (xmin + xmax) * 0.5 + 0.05118;
    float cy = (ymin + ymax) * 0.5 + -0.028581;

    float dot = 0;

    float quadrants[2][2] = {{-1*(2 << 15), 0}, {2*(2 << 15), 2 << 15}};

    for (int pidx = 0; pidx < zarray_size(cluster); pidx++) {
        struct pt *p;
        zarray_get_volatile(cluster, pidx, &p);

        float dx = p->x - cx;
        float dy = p->y - cy;

        dot += dx*p->gx + dy*p->gy;

        float quadrant = quadrants[dy > 0][dx > 0];
        if (dy < 0) {
            dy = -dy;
            dx = -dx;
        }

        if (dx < 0) {
            float tmp = dx;
            dx = dy;
            dy = -tmp;
        }
        p->slope = quadrant + dy/dx;
    }

    // Ensure that the black border is inside the white border.
    quad->reversed_border = dot < 0;
    if (!reversed_border && quad->reversed_border) {
        return 0;
    }
    if (!normal_border && !quad->reversed_border) {
        return 0;
    }

    int sz = zarray_size(cluster);
    quad_fit_scratch_ensure(scratch, sz);

    // we now sort the points according to theta. This is a prepatory
    // step for segmenting them into four lines.
    if (1) {
        pt_slope_sort((struct pt*) cluster->data, sz, scratch);
    }

    struct line_fit_pt *lfps = scratch->lfps;
    compute_lfps(sz, cluster, im, lfps);

    int indices[4];
    if (1) {
        if (!quad_segment_maxima(td, cluster, lfps, indices, scratch))
            goto finish;
    } else {
        if (!quad_segment_agg(cluster, lfps, indices))
            goto finish;
    }


    double lines[4][4];

    for (int i = 0; i < 4; i++) {
        int i0 = indices[i];
        int i1 = indices[(i+1)&3];

        double mse;
        fit_line(lfps, sz, i0, i1, lines[i], NULL, &mse);

        if (mse > td->qtp.max_line_fit_mse) {
            res = 0;
            goto finish;
        }
    }

    for (int i = 0; i < 4; i++) {
        // solve for the intersection of lines (i) and (i+1)&3.
        // p0 + lambda0*u0 = p1 + lambda1*u1, where u0 and u1
        // are the line directions.
        //
        // lambda0*u0 - lambda1*u1 = (p1 - p0)
        //
        // rearrange (solve for lambdas)
        //
        // [u0_x   -u1_x ] [lambda0] = [ p1_x - p0_x ]
        // [u0_y   -u1_y ] [lambda1]   [ p1_y - p0_y ]
        //
        // remember that lines[i][0,1] = p, lines[i][2,3] = NORMAL vector.
        // We want the unit vector, so we need the perpendiculars. Thus, below
        // we have swapped the x and y components and flipped the y components.

        double A00 =  lines[i][3],  A01 = -lines[(i+1)&3][3];
        double A10 =  -lines[i][2],  A11 = lines[(i+1)&3][2];
        double B0 = -lines[i][0] + lines[(i+1)&3][0];
        double B1 = -lines[i][1] + lines[(i+1)&3][1];

        double det = A00 * A11 - A10 * A01;

        // inverse.
        if (fabs(det) < 0.001) {
            res = 0;
            goto finish;
        }
        double W00 = A11 / det, W01 = -A01 / det;

        // solve
        double L0 = W00*B0 + W01*B1;

        // compute intersection
        quad->p[i][0] = lines[i][0] + L0*A00;
        quad->p[i][1] = lines[i][1] + L0*A10;

        res = 1;
    }

    // reject quads that are too small
    if (1) {
        double area = 0;

        // get area of triangle formed by points 0, 1, 2, 0
        double length[3], p;
        for (int i = 0; i < 3; i++) {
            int idxa = i; // 0, 1, 2,
            int idxb = (i+1) % 3; // 1, 2, 0
            length[i] = sqrt(sq(quad->p[idxb][0] - quad->p[idxa][0]) +
                             sq(quad->p[idxb][1] - quad->p[idxa][1]));
        }
        p = (length[0] + length[1] + length[2]) / 2;

        area += sqrt(p*(p-length[0])*(p-length[1])*(p-length[2]));

        // get area of triangle formed by points 2, 3, 0, 2
        for (int i = 0; i < 3; i++) {
            int idxs[] = { 2, 3, 0, 2 };
            int idxa = idxs[i];
            int idxb = idxs[i+1];
            length[i] = sqrt(sq(quad->p[idxb][0] - quad->p[idxa][0]) +
                             sq(quad->p[idxb][1] - quad->p[idxa][1]));
        }
        p = (length[0] + length[1] + length[2]) / 2;

        area += sqrt(p*(p-length[0])*(p-length[1])*(p-length[2]));

        if (area < 0.95*tag_width*tag_width) {
            res = 0;
            goto finish;
        }
    }

    // reject quads whose cumulative angle change isn't equal to 2PI
    if (1) {
        for (int i = 0; i < 4; i++) {
            int i0 = i, i1 = (i+1)&3, i2 = (i+2)&3;

            double dx1 = quad->p[i1][0] - quad->p[i0][0];
            double dy1 = quad->p[i1][1] - quad->p[i0][1];
            double dx2 = quad->p[i2][0] - quad->p[i1][0];
            double dy2 = quad->p[i2][1] - quad->p[i1][1];
            double denominator = sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2));
            if (denominator == 0) {
                res = 0;
                goto finish;
            }
            double cos_dtheta = (dx1*dx2 + dy1*dy2) / denominator;

            if ((cos_dtheta > td->qtp.cos_critical_rad || cos_dtheta < -td->qtp.cos_critical_rad) || dx1*dy2 < dy1*dx2) {
                res = 0;
                goto finish;
            }
        }
    }

  finish:

    return res;
}

// a maximal horizontal segment of equal non-127 pixels, x in [0, w-2]
// (the last column never participates in runs; it is only reachable as a
// diagonal neighbor of a white run ending at w-2)
struct row_run
{
    uint16_t start, end; // inclusive
    uint8_t v;
};

static int rle_row(const uint8_t *row, int w, struct row_run *runs)
{
    int n = 0;
    int x = 0;
    int xmax = w - 2;
    while (x <= xmax) {
        uint8_t v = row[x];
        if (v == 127) {
            x++;
            continue;
        }
        int start = x;
        x++;
        while (x <= xmax && row[x] == v)
            x++;
        runs[n].start = start;
        runs[n].end = x - 1;
        runs[n].v = v;
        n++;
    }
    return n;
}

// Initialize each run's head as a union-find node owning the whole run.
// Only run heads (and the lazily-initialized last column) ever enter the
// union-find: the cluster pass resolves representatives through run heads
// too, so per-pixel parent entries are never needed.
static void unionfind_init_run_heads(unionfind_t *uf, int w, int y, struct row_run *runs, int nruns)
{
    uint32_t base = (uint32_t)y*w;
    for (int i = 0; i < nruns; i++) {
        uint32_t head = base + runs[i].start;
        uf->parent[head] = head;
        uf->size[head] = runs[i].end - runs[i].start; // excludes the root
    }
}

// Union the runs of row y against the runs of row y-1: one union per pair
// of vertically (or, for white, diagonally) adjacent same-value runs. The
// per-pixel code's skip conditions already reduce its connects to exactly
// these pairs, so the resulting components and sizes are identical.
static void connect_runs_to_prev(unionfind_t *uf, const uint8_t *buf, int w, int s, int y,
                                 struct row_run *cur, int ncur, struct row_run *prev, int nprev)
{
    int j = 0;
    for (int i = 0; i < ncur; i++) {
        int a0 = cur[i].start, a1 = cur[i].end;
        uint8_t v = cur[i].v;
        uint32_t head_a = (uint32_t)y*w + a0;

        while (j < nprev && prev[j].end + 1 < a0)
            j++;

        for (int k = j; k < nprev && prev[k].start <= a1 + 1; k++) {
            if (prev[k].v != v)
                continue;
            int b0 = prev[k].start, b1 = prev[k].end;
            uint32_t head_b = (uint32_t)(y-1)*w + b0;

            // direct vertical contact (only at x >= 1; the per-pixel
            // code never connects column 0 upward)
            int lo = imax(imax(a0, b0), 1);
            int hi = imin(a1, b1);
            if (lo <= hi) {
                unionfind_connect(uf, head_a, head_b);
            } else if (v == 255) {
                // white is 8-connected: diagonal-only contact
                int xl = imax(imax(a0, b0 + 1), 1);
                if (xl <= imin(a1, b1 + 1)) {
                    unionfind_connect(uf, head_a, head_b);
                } else {
                    int xr = imax(imax(a0, b0 - 1), 1);
                    if (xr <= imin(a1, b1 - 1)) {
                        unionfind_connect(uf, head_a, head_b);
                    }
                }
            }
        }

        // The last column holds no runs, but a white run ending at w-2
        // reaches (w-1, y-1) diagonally. The per-pixel code only does
        // this connect when the pixel above the run end is not white
        // (otherwise its redundancy test skips it).
        if (v == 255 && a1 == w-2 && buf[(y-1)*s + (w-1)] == 255 && buf[(y-1)*s + (w-2)] != 255) {
            unionfind_connect(uf, head_a, (uint32_t)(y-1)*w + (w-1));
        }
    }
}

// Process rows [y0, y1) by runs. The row above the chunk (row 0 or a gap
// row owned by no task) has its run heads initialized here; the serial
// stitch pass later connects gap rows to the rows above them.
static void do_unionfind_task2(void *p)
{
    struct unionfind_task *task = (struct unionfind_task*) p;
    unionfind_t *uf = task->uf;
    int w = task->w, s = task->s;
    uint8_t *buf = task->im->buf;

    struct row_run *prev = malloc(sizeof(struct row_run)*(w+1));
    struct row_run *cur = malloc(sizeof(struct row_run)*(w+1));

    int nprev = rle_row(&buf[(task->y0 - 1)*s], w, prev);
    // no unions touch the prev row before this task runs, so initializing
    // its heads here is race-free (re-initialization before any union is
    // an identity for row 0 in the single-thread path)
    unionfind_init_run_heads(uf, w, task->y0 - 1, prev, nprev);

    for (int y = task->y0; y < task->y1; y++) {
        int ncur = rle_row(&buf[y*s], w, cur);
        unionfind_init_run_heads(uf, w, y, cur, ncur);
        connect_runs_to_prev(uf, buf, w, s, y, cur, ncur, prev, nprev);

        struct row_run *t = prev;
        prev = cur;
        cur = t;
        nprev = ncur;
    }

    free(prev);
    free(cur);
}

static void do_quad_task(void *p)
{
    struct quad_task *task = (struct quad_task*) p;

    zarray_t *clusters = task->clusters;
    zarray_t *quads = task->quads;
    apriltag_detector_t *td = task->td;
    int w = task->w, h = task->h;

    struct quad_fit_scratch scratch;
    memset(&scratch, 0, sizeof(scratch));

    for (int cidx = task->cidx0; cidx < task->cidx1; cidx++) {

        zarray_t **cluster;
        zarray_get_volatile(clusters, cidx, &cluster);

        // a cluster should contain only boundary points around the
        // tag. it cannot be bigger than the whole screen. (Reject
        // large connected blobs that will be prohibitively slow to
        // fit quads to.) A typical point along an edge is added two
        // times (because it has 2 unique neighbors). The maximum
        // perimeter is 2w+2h.
        if (zarray_size(*cluster) >= td->qtp.min_cluster_pixels &&
            zarray_size(*cluster) <= 2*(2*w+2*h)) {

            struct quad quad;
            memset(&quad, 0, sizeof(struct quad));

            if (fit_quad(td, task->im, *cluster, &quad, task->tag_width, task->normal_border, task->reversed_border, &scratch)) {
                pthread_mutex_lock(&td->mutex);
                zarray_add(quads, &quad);
                pthread_mutex_unlock(&td->mutex);
            }
        }

        // destroy here, in parallel and while cache-warm, rather than in
        // a serial loop after all quad tasks finish
        zarray_destroy(*cluster);
        *cluster = NULL;
    }

    quad_fit_scratch_free(&scratch);
}

void do_minmax_task(void *p)
{
    const int tilesz = 4;
    struct minmax_task* task = (struct minmax_task*) p;
    int s = task->im->stride;
    int ty = task->ty;
    int tw = task->im->width / tilesz;
    image_u8_t *im = task->im;

    for (int tx = 0; tx < tw; tx++) {
        uint8_t max = 0, min = 255;

        for (int dy = 0; dy < tilesz; dy++) {

            for (int dx = 0; dx < tilesz; dx++) {

                uint8_t v = im->buf[(ty*tilesz+dy)*s + tx*tilesz + dx];
                if (v < min)
                    min = v;
                if (v > max)
                    max = v;
            }
        }

        task->im_max[ty*tw+tx] = max;
        task->im_min[ty*tw+tx] = min;
    }
}

void do_blur_task(void *p)
{
    const int tilesz = 4;
    struct blur_task* task = (struct blur_task*) p;
    int ty = task->ty;
    int tw = task->im->width / tilesz;
    int th = task->im->height / tilesz;
    uint8_t *im_max = task->im_max;
    uint8_t *im_min = task->im_min;

    for (int tx = 0; tx < tw; tx++) {
        uint8_t max = 0, min = 255;

        for (int dy = -1; dy <= 1; dy++) {
            if (ty+dy < 0 || ty+dy >= th)
                continue;
            for (int dx = -1; dx <= 1; dx++) {
                if (tx+dx < 0 || tx+dx >= tw)
                    continue;

                uint8_t m = im_max[(ty+dy)*tw+tx+dx];
                if (m > max)
                    max = m;
                m = im_min[(ty+dy)*tw+tx+dx];
                if (m < min)
                    min = m;
            }
        }

        task->im_max_tmp[ty*tw + tx] = max;
        task->im_min_tmp[ty*tw + tx] = min;
    }
}

void do_threshold_task(void *p)
{
    const int tilesz = 4;
    struct threshold_task* task = (struct threshold_task*) p;
    int ty = task->ty;
    int tw = task->im->width / tilesz;
    int s = task->im->stride;
    uint8_t *im_max = task->im_max;
    uint8_t *im_min = task->im_min;
    image_u8_t *im = task->im;
    image_u8_t *threshim = task->threshim;
    int min_white_black_diff = task->td->qtp.min_white_black_diff;

    for (int tx = 0; tx < tw; tx++) {
        int min = im_min[ty*tw + tx];
        int max = im_max[ty*tw + tx];

        // low contrast region? (no edges)
        if (max - min < min_white_black_diff) {
            for (int dy = 0; dy < tilesz; dy++) {
                int y = ty*tilesz + dy;

                for (int dx = 0; dx < tilesz; dx++) {
                    int x = tx*tilesz + dx;

                    threshim->buf[y*s+x] = 127;
                }
            }
            continue;
        }

        // otherwise, actually threshold this tile.

        // argument for biasing towards dark; specular highlights
        // can be substantially brighter than white tag parts
        uint8_t thresh = min + (max - min) / 2;

        for (int dy = 0; dy < tilesz; dy++) {
            int y = ty*tilesz + dy;

            for (int dx = 0; dx < tilesz; dx++) {
                int x = tx*tilesz + dx;

                uint8_t v = im->buf[y*s+x];
                if (v > thresh)
                    threshim->buf[y*s+x] = 255;
                else
                    threshim->buf[y*s+x] = 0;
            }
        }
    }
}
 
image_u8_t *threshold(apriltag_detector_t *td, image_u8_t *im)
{
    int w = im->width, h = im->height, s = im->stride;
    assert(w < 32768);
    assert(h < 32768);

    if (td->cached_threshim && (td->cached_threshim->width != w ||
                                td->cached_threshim->height != h ||
                                td->cached_threshim->stride != s)) {
        image_u8_destroy(td->cached_threshim);
        td->cached_threshim = NULL;
    }
    if (!td->cached_threshim)
        td->cached_threshim = image_u8_create_alignment(w, h, s);
    image_u8_t *threshim = td->cached_threshim;
    assert(threshim->stride == s);

    // The idea is to find the maximum and minimum values in a
    // window around each pixel. If it's a contrast-free region
    // (max-min is small), don't try to binarize. Otherwise,
    // threshold according to (max+min)/2.
    //
    // Mark low-contrast regions with value 127 so that we can skip
    // future work on these areas too.

    // however, computing max/min around every pixel is needlessly
    // expensive. We compute max/min for tiles. To avoid artifacts
    // that arise when high-contrast features appear near a tile
    // edge (and thus moving from one tile to another results in a
    // large change in max/min value), the max/min values used for
    // any pixel are computed from all 3x3 surrounding tiles. Thus,
    // the max/min sampling area for nearby pixels overlap by at least
    // one tile.
    //
    // The important thing is that the windows be large enough to
    // capture edge transitions; the tag does not need to fit into
    // a tile.

    // XXX Tunable. Generally, small tile sizes--- so long as they're
    // large enough to span a single tag edge--- seem to be a winner.
    const int tilesz = 4;

    // the last (possibly partial) tiles along each row and column will
    // just use the min/max value from the last full tile.
    int tw = w / tilesz;
    int th = h / tilesz;

    // tile min/max scratch (4 arrays: min, max, and their blur outputs),
    // reused across detect calls; every entry is written before being read
    if (td->cached_tile_bufs_size < 4*tw*th) {
        free(td->cached_tile_bufs);
        td->cached_tile_bufs = malloc(4*tw*th);
        td->cached_tile_bufs_size = 4*tw*th;
    }
    uint8_t *im_max = td->cached_tile_bufs;
    uint8_t *im_min = td->cached_tile_bufs + tw*th;

    struct minmax_task *minmax_tasks = malloc(sizeof(struct minmax_task)*th);
    // first, collect min/max statistics for each tile
    for (int ty = 0; ty < th; ty++) {
        minmax_tasks[ty].im = im;
        minmax_tasks[ty].im_max = im_max;
        minmax_tasks[ty].im_min = im_min;
        minmax_tasks[ty].ty = ty;

        workerpool_add_task(td->wp, do_minmax_task, &minmax_tasks[ty]);
    }
    workerpool_run(td->wp);
    free(minmax_tasks);

    // second, apply 3x3 max/min convolution to "blur" these values
    // over larger areas. This reduces artifacts due to abrupt changes
    // in the threshold value.
    if (1) {
        uint8_t *im_max_tmp = td->cached_tile_bufs + 2*tw*th;
        uint8_t *im_min_tmp = td->cached_tile_bufs + 3*tw*th;

        struct blur_task *blur_tasks = malloc(sizeof(struct blur_task)*th);
        for (int ty = 0; ty < th; ty++) {
            blur_tasks[ty].im = im;
            blur_tasks[ty].im_max = im_max;
            blur_tasks[ty].im_min = im_min;
            blur_tasks[ty].im_max_tmp = im_max_tmp;
            blur_tasks[ty].im_min_tmp = im_min_tmp;
            blur_tasks[ty].ty = ty;

            workerpool_add_task(td->wp, do_blur_task, &blur_tasks[ty]);
        }
        workerpool_run(td->wp);
        free(blur_tasks);
        im_max = im_max_tmp;
        im_min = im_min_tmp;
    }

    struct threshold_task *threshold_tasks = malloc(sizeof(struct threshold_task)*th);
    for (int ty = 0; ty < th; ty++) {
        threshold_tasks[ty].im = im;
        threshold_tasks[ty].threshim = threshim;
        threshold_tasks[ty].im_max = im_max;
        threshold_tasks[ty].im_min = im_min;
        threshold_tasks[ty].ty = ty;
        threshold_tasks[ty].td = td;

        workerpool_add_task(td->wp, do_threshold_task, &threshold_tasks[ty]);
    }
    workerpool_run(td->wp);
    free(threshold_tasks);

    // we skipped over the non-full-sized tiles above. Fix those now.
    if (1) {
        for (int y = 0; y < h; y++) {

            // what is the first x coordinate we need to process in this row?

            int x0;

            if (y >= th*tilesz) {
                x0 = 0; // we're at the bottom; do the whole row.
            } else {
                x0 = tw*tilesz; // we only need to do the right most part.
            }

            // compute tile coordinates and clamp.
            int ty = y / tilesz;
            if (ty >= th)
                ty = th - 1;

            for (int x = x0; x < w; x++) {
                int tx = x / tilesz;
                if (tx >= tw)
                    tx = tw - 1;

                int max = im_max[ty*tw + tx];
                int min = im_min[ty*tw + tx];
                int thresh = min + (max - min) / 2;

                uint8_t v = im->buf[y*s+x];
                if (v > thresh)
                    threshim->buf[y*s+x] = 255;
                else
                    threshim->buf[y*s+x] = 0;
            }
        }
    }


    // this is a dilate/erode deglitching scheme that does not improve
    // anything as far as I can tell.
    if (td->qtp.deglitch) {
        image_u8_t *tmp = image_u8_create(w, h);

        for (int y = 1; y + 1 < h; y++) {
            for (int x = 1; x + 1 < w; x++) {
                uint8_t max = 0;
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        uint8_t v = threshim->buf[(y+dy)*s + x + dx];
                        if (v > max)
                            max = v;
                    }
                }
                tmp->buf[y*s+x] = max;
            }
        }

        for (int y = 1; y + 1 < h; y++) {
            for (int x = 1; x + 1 < w; x++) {
                uint8_t min = 255;
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        uint8_t v = tmp->buf[(y+dy)*s + x + dx];
                        if (v < min)
                            min = v;
                    }
                }
                threshim->buf[y*s+x] = min;
            }
        }

        image_u8_destroy(tmp);
    }

    timeprofile_stamp(td->tp, "threshold");

    return threshim;
}

// basically the same as threshold(), but assumes the input image is a
// bayer image. It collects statistics separately for each 2x2 block
// of pixels. NOT WELL TESTED.
image_u8_t *threshold_bayer(apriltag_detector_t *td, image_u8_t *im)
{
    int w = im->width, h = im->height, s = im->stride;

    image_u8_t *threshim = image_u8_create_alignment(w, h, s);
    assert(threshim->stride == s);

    int tilesz = 32;
    assert((tilesz & 1) == 0); // must be multiple of 2

    int tw = w/tilesz + 1;
    int th = h/tilesz + 1;

    uint8_t *im_max[4], *im_min[4];
    for (int i = 0; i < 4; i++) {
        im_max[i] = calloc(tw*th, sizeof(uint8_t));
        im_min[i] = calloc(tw*th, sizeof(uint8_t));
    }

    for (int ty = 0; ty < th; ty++) {
        for (int tx = 0; tx < tw; tx++) {

            uint8_t max[4] = { 0, 0, 0, 0};
            uint8_t min[4] = { 255, 255, 255, 255 };

            for (int dy = 0; dy < tilesz; dy++) {
                if (ty*tilesz+dy >= h)
                    continue;

                for (int dx = 0; dx < tilesz; dx++) {
                    if (tx*tilesz+dx >= w)
                        continue;

                    // which bayer element is this pixel?
                    int idx = (2*(dy&1) + (dx&1));

                    uint8_t v = im->buf[(ty*tilesz+dy)*s + tx*tilesz + dx];
                    if (v < min[idx])
                        min[idx] = v;
                    if (v > max[idx])
                        max[idx] = v;
                }
            }

            for (int i = 0; i < 4; i++) {
                im_max[i][ty*tw+tx] = max[i];
                im_min[i][ty*tw+tx] = min[i];
            }
        }
    }

    for (int ty = 0; ty < th; ty++) {
        for (int tx = 0; tx < tw; tx++) {

            uint8_t max[4] = { 0, 0, 0, 0};
            uint8_t min[4] = { 255, 255, 255, 255 };

            for (int dy = -1; dy <= 1; dy++) {
                if (ty+dy < 0 || ty+dy >= th)
                    continue;
                for (int dx = -1; dx <= 1; dx++) {
                    if (tx+dx < 0 || tx+dx >= tw)
                        continue;

                    for (int i = 0; i < 4; i++) {
                        uint8_t m = im_max[i][(ty+dy)*tw+tx+dx];
                        if (m > max[i])
                            max[i] = m;
                        m = im_min[i][(ty+dy)*tw+tx+dx];
                        if (m < min[i])
                            min[i] = m;
                    }
                }
            }

            // XXX CONSTANT
//            if (max - min < 30)
//                continue;

            // argument for biasing towards dark: specular highlights
            // can be substantially brighter than white tag parts
            uint8_t thresh[4];
            for (int i = 0; i < 4; i++) {
                thresh[i] = min[i] + (max[i] - min[i]) / 2;
            }

            for (int dy = 0; dy < tilesz; dy++) {
                int y = ty*tilesz + dy;
                if (y >= h)
                    continue;

                for (int dx = 0; dx < tilesz; dx++) {
                    int x = tx*tilesz + dx;
                    if (x >= w)
                        continue;

                    // which bayer element is this pixel?
                    int idx = (2*(y&1) + (x&1));

                    uint8_t v = im->buf[y*s+x];
                    threshim->buf[y*s+x] = v > thresh[idx];
                }
            }
        }
    }

    for (int i = 0; i < 4; i++) {
        free(im_min[i]);
        free(im_max[i]);
    }

    timeprofile_stamp(td->tp, "threshold");

    return threshim;
}

unionfind_t* connected_components(apriltag_detector_t *td, image_u8_t* threshim, int w, int h, int ts) {
    uint32_t maxid = w * h;
    if (td->cached_uf) {
        if (td->cached_uf->maxid < maxid) {
            unionfind_resize(td->cached_uf, maxid);
        }
    } else {
        td->cached_uf = unionfind_create(maxid);
    }
    unionfind_t *uf = td->cached_uf;

    // No full unionfind_reset between frames: only run heads enter the
    // union-find, and every head is (re)initialized by the run pass below.
    // The last column is the one set of pixels still initialized lazily
    // (reachable as a diagonal neighbor), so invalidate its stale parents.
    // Debug runs reset everything so the per-pixel debug queries see
    // lazily-initialized singletons for pixels outside any run.
    if (td->debug)
        unionfind_reset(uf);
    for (int y = 0; y < h; y++)
        uf->parent[(uint32_t)y*w + (w-1)] = 0xffffffff;

    if (td->nthreads <= 1) {
        struct unionfind_task task;
        task.y0 = 1;
        task.y1 = h;
        task.h = h;
        task.w = w;
        task.s = ts;
        task.uf = uf;
        task.im = threshim;
        do_unionfind_task2(&task);
    } else {
        int sz = h;
        int chunksize = 1 + sz / (APRILTAG_TASKS_PER_THREAD_TARGET * td->nthreads);
        // a chunk size below 2 would leave rows covered by no task
        if (chunksize < 2)
            chunksize = 2;
        struct unionfind_task *tasks = malloc(sizeof(struct unionfind_task)*(sz / chunksize + 1));

        int ntasks = 0;

        for (int i = 1; i < sz; i += chunksize) {
            // each task will process [y0, y1). Note that this attaches
            // each cell to the right and down, so row y1 *is* potentially modified.
            //
            // for parallelization, make sure that each task doesn't touch rows
            // used by another thread.
            tasks[ntasks].y0 = i;
            tasks[ntasks].y1 = imin(sz, i + chunksize - 1);
            tasks[ntasks].h = h;
            tasks[ntasks].w = w;
            tasks[ntasks].s = ts;
            tasks[ntasks].uf = uf;
            tasks[ntasks].im = threshim;

            ntasks++;
        }

        for (int i = 0; i < ntasks; i++) {
            workerpool_add_task(td->wp, do_unionfind_task2, &tasks[i]);
        }

        workerpool_run(td->wp);

        // stitch together the chunks: connect each gap row (whose heads the
        // task below initialized) to the row above it
        if (ntasks > 1) {
            struct row_run *gap = malloc(sizeof(struct row_run)*(w+1));
            struct row_run *above = malloc(sizeof(struct row_run)*(w+1));
            for (int i = 1; i < ntasks; i++) {
                int gy = tasks[i].y0 - 1;
                int ngap = rle_row(&threshim->buf[gy*ts], w, gap);
                int nabove = rle_row(&threshim->buf[(gy-1)*ts], w, above);
                connect_runs_to_prev(uf, threshim->buf, w, ts, gy, gap, ngap, above, nabove);
            }
            free(gap);
            free(above);
        }

        free(tasks);
    }

    // The debug segmentation image queries the representative of every
    // pixel; attach each run's pixels to its head so those queries resolve.
    if (td->debug) {
        struct row_run *runs = malloc(sizeof(struct row_run)*(w+1));
        for (int y = 0; y < h; y++) {
            int n = rle_row(&threshim->buf[y*ts], w, runs);
            for (int i = 0; i < n; i++) {
                uint32_t head = (uint32_t)y*w + runs[i].start;
                for (int x = runs[i].start + 1; x <= runs[i].end; x++)
                    uf->parent[(uint32_t)y*w + x] = head;
            }
        }
        free(runs);
    }

    return uf;
}

// per-cluster-task hash/pool state for gc_add_point
struct gc_ctx
{
    struct uint64_zarray_entry **clustermap;
    uint32_t bucket_mask;
    struct uint64_zarray_entry **mem_pools;
    int mem_chunk_size;
    int mem_pools_capacity;
    int mem_pool_idx;
    int mem_pool_loc;
    struct gc_chunk_pool chunk_pool;
    // consecutive boundary points usually belong to the same cluster, so
    // remember the last entry to skip the hash lookup
    struct uint64_zarray_entry *last_entry;
};

// Add the point half-way between two adjacent black/white pixels to the
// cluster keyed by the components' representative pair. (v1-v0) is +-255
// and points towards the white pixel.
static inline void gc_add_point(struct gc_ctx *ctx, uint64_t rep0, uint64_t rep1,
                                int px, int py, int gx, int gy)
{
    uint64_t clusterid;
    if (rep0 < rep1)
        clusterid = (rep1 << 32) + rep0;
    else
        clusterid = (rep0 << 32) + rep1;

    struct uint64_zarray_entry *entry;
    if (ctx->last_entry && ctx->last_entry->id == clusterid) {
        entry = ctx->last_entry;
    } else {
        /* XXX lousy hash function */
        uint32_t bucket = u64hash_2(clusterid) & ctx->bucket_mask;
        entry = ctx->clustermap[bucket];
        while (entry && entry->id != clusterid) {
            entry = entry->next;
        }

        if (!entry) {
            if (ctx->mem_pool_loc == ctx->mem_chunk_size) {
                ctx->mem_pool_loc = 0;
                ctx->mem_pool_idx++;
                if (ctx->mem_pool_idx == ctx->mem_pools_capacity) {
                    ctx->mem_pools_capacity *= 2;
                    ctx->mem_pools = realloc(ctx->mem_pools, sizeof(struct uint64_zarray_entry *)*ctx->mem_pools_capacity);
                }
                ctx->mem_pools[ctx->mem_pool_idx] = calloc(ctx->mem_chunk_size, sizeof(struct uint64_zarray_entry));
            }
            entry = ctx->mem_pools[ctx->mem_pool_idx] + ctx->mem_pool_loc;
            ctx->mem_pool_loc++;

            entry->id = clusterid;
            entry->head = entry->tail = gc_chunk_alloc(&ctx->chunk_pool);
            entry->npts = 0;
            entry->next = ctx->clustermap[bucket];
            ctx->clustermap[bucket] = entry;
        }
        ctx->last_entry = entry;
    }

    struct gc_chunk *t = entry->tail;
    if (t->count == GC_CHUNK_PTS) {
        struct gc_chunk *c = gc_chunk_alloc(&ctx->chunk_pool);
        t->next = c;
        entry->tail = c;
        t = c;
    }
    struct pt p = { .x = px, .y = py, .gx = gx, .gy = gy };
    t->pts[t->count++] = p;
    entry->npts++;
}

// append directly to an already-resolved entry
static inline void gc_entry_append(struct gc_ctx *ctx, struct uint64_zarray_entry *entry,
                                   int px, int py, int gx, int gy)
{
    struct gc_chunk *t = entry->tail;
    if (t->count == GC_CHUNK_PTS) {
        struct gc_chunk *c = gc_chunk_alloc(&ctx->chunk_pool);
        t->next = c;
        entry->tail = c;
        t = c;
    }
    struct pt p = { .x = px, .y = py, .gx = gx, .gy = gy };
    t->pts[t->count++] = p;
    entry->npts++;
}

// lazily computed representative + size gate for one row run
struct run_rep
{
    uint32_t rep;
    int8_t state; // 0 = unknown, 1 = usable, 2 = component too small
};

static inline int run_usable(unionfind_t *uf, int w, int y, const struct row_run *runs,
                             struct run_rep *cache, int idx, int min_cluster_pixels, uint32_t *rep_out)
{
    if (cache[idx].state == 0) {
        uint32_t rep = unionfind_get_representative(uf, (uint32_t)y*w + runs[idx].start);
        cache[idx].rep = rep;
        cache[idx].state = ((int)(uf->size[rep] + 1) >= min_cluster_pixels) ? 1 : 2;
    }
    *rep_out = cache[idx].rep;
    return cache[idx].state == 1;
}

// Run-driven cluster construction. This emits exactly the points the
// historical per-pixel scan emitted -- in the same (y, x, neighbor) order,
// with the same lazy component-size gates -- but derives boundaries from
// row runs: the component representative is resolved once per run instead
// of once per pixel, 127 spans are skipped wholesale, and the cluster
// entry is found once per run pair instead of once per point.
//
// nclustermap must be a power of two.
zarray_t* do_gradient_clusters(image_u8_t* threshim, int ts, int y0, int y1, int w, int nclustermap, int min_cluster_pixels, unionfind_t* uf, zarray_t* clusters) {
    struct gc_ctx ctx;
    ctx.clustermap = calloc(nclustermap, sizeof(struct uint64_zarray_entry*));
    ctx.bucket_mask = (uint32_t)nclustermap - 1;
    ctx.mem_chunk_size = 2048;
    ctx.mem_pools_capacity = 16;
    ctx.mem_pools = malloc(sizeof(struct uint64_zarray_entry *)*ctx.mem_pools_capacity);
    ctx.mem_pool_idx = 0;
    ctx.mem_pool_loc = 0;
    ctx.mem_pools[0] = calloc(ctx.mem_chunk_size, sizeof(struct uint64_zarray_entry));
    gc_chunk_pool_init(&ctx.chunk_pool);
    ctx.last_entry = NULL;

    uint8_t *buf = threshim->buf;

    struct row_run *runs_a = malloc(sizeof(struct row_run)*(w+1)); // row y
    struct row_run *runs_b = malloc(sizeof(struct row_run)*(w+1)); // row y+1
    struct run_rep *cache_a = malloc(sizeof(struct run_rep)*(w+1));
    struct run_rep *cache_b = malloc(sizeof(struct run_rep)*(w+1));

    int na = rle_row(&buf[y0*ts], w, runs_a);
    for (int i = 0; i < na; i++)
        cache_a[i].state = 0;

    for (int y = y0; y < y1; y++) {
        int nb = rle_row(&buf[(y+1)*ts], w, runs_b);
        for (int i = 0; i < nb; i++)
            cache_b[i].state = 0;

        // did the previous pixel add a point via its (1,1) neighbor?
        bool connected_last = false;

        // sweep pointers into runs_b for targets x, x-1, x+1
        int p0 = 0, pm = 0, pp = 0;

        for (int ia = 0; ia < na; ia++) {
            int a0 = runs_a[ia].start, a1 = runs_a[ia].end;
            uint8_t v0 = runs_a[ia].v;
            uint8_t vopp = 255 - v0;
            int vdiff = (int)vopp - (int)v0; // v1 - v0, +-255

            // a 127 gap before this run resets the (1,1) memory
            if (ia == 0 || runs_a[ia-1].end + 1 < a0)
                connected_last = false;

            uint32_t rep0 = 0;
            int rep0_state = 0; // lazy, as in the per-pixel code

            int ax0 = a0 < 1 ? 1 : a0;

            for (int x = ax0; x <= a1; x++) {
                bool connected = false;

                // (1, 0): right neighbor differs only at the run end
                if (x == a1) {
                    uint8_t v1 = buf[y*ts + x + 1];
                    if (v0 + v1 == 255) {
                        if (rep0_state == 0) {
                            rep0_state = run_usable(uf, w, y, runs_a, cache_a, ia, min_cluster_pixels, &rep0) ? 1 : 2;
                        }
                        if (rep0_state == 1) {
                            uint32_t rep1;
                            int ok;
                            if (ia + 1 < na && runs_a[ia+1].start == x + 1) {
                                ok = run_usable(uf, w, y, runs_a, cache_a, ia+1, min_cluster_pixels, &rep1);
                            } else {
                                // x+1 == w-1: the last column holds no runs
                                rep1 = unionfind_get_representative(uf, (uint32_t)y*w + x + 1);
                                ok = (int)(uf->size[rep1] + 1) >= min_cluster_pixels;
                            }
                            if (ok)
                                gc_add_point(&ctx, rep0, rep1, 2*x + 1, 2*y, vdiff, 0);
                        }
                    }
                }

                if (rep0_state != 2) {
                    // (0, 1)
                    while (p0 < nb && runs_b[p0].end < x)
                        p0++;
                    if (p0 < nb && runs_b[p0].start <= x && runs_b[p0].v == vopp) {
                        if (rep0_state == 0)
                            rep0_state = run_usable(uf, w, y, runs_a, cache_a, ia, min_cluster_pixels, &rep0) ? 1 : 2;
                        if (rep0_state == 1) {
                            uint32_t rep1;
                            if (run_usable(uf, w, y+1, runs_b, cache_b, p0, min_cluster_pixels, &rep1))
                                gc_add_point(&ctx, rep0, rep1, 2*x, 2*y + 1, 0, vdiff);
                        }
                    }
                }

                // (-1, 1): skipped when the previous pixel's (1,1) already
                // added this point
                if (rep0_state != 2 && !connected_last) {
                    while (pm < nb && runs_b[pm].end < x - 1)
                        pm++;
                    if (pm < nb && runs_b[pm].start <= x - 1 && runs_b[pm].v == vopp) {
                        if (rep0_state == 0)
                            rep0_state = run_usable(uf, w, y, runs_a, cache_a, ia, min_cluster_pixels, &rep0) ? 1 : 2;
                        if (rep0_state == 1) {
                            uint32_t rep1;
                            if (run_usable(uf, w, y+1, runs_b, cache_b, pm, min_cluster_pixels, &rep1))
                                gc_add_point(&ctx, rep0, rep1, 2*x - 1, 2*y + 1, -vdiff, vdiff);
                        }
                    }
                }

                // (1, 1)
                if (rep0_state != 2) {
                    if (x + 1 <= w - 2) {
                        while (pp < nb && runs_b[pp].end < x + 1)
                            pp++;
                        if (pp < nb && runs_b[pp].start <= x + 1 && runs_b[pp].v == vopp) {
                            if (rep0_state == 0)
                                rep0_state = run_usable(uf, w, y, runs_a, cache_a, ia, min_cluster_pixels, &rep0) ? 1 : 2;
                            if (rep0_state == 1) {
                                uint32_t rep1;
                                if (run_usable(uf, w, y+1, runs_b, cache_b, pp, min_cluster_pixels, &rep1)) {
                                    gc_add_point(&ctx, rep0, rep1, 2*x + 1, 2*y + 1, vdiff, vdiff);
                                    connected = true;
                                }
                            }
                        }
                    } else {
                        // x+1 == w-1: the last column holds no runs
                        uint8_t v1 = buf[(y+1)*ts + x + 1];
                        if (v0 + v1 == 255) {
                            if (rep0_state == 0)
                                rep0_state = run_usable(uf, w, y, runs_a, cache_a, ia, min_cluster_pixels, &rep0) ? 1 : 2;
                            if (rep0_state == 1) {
                                uint32_t rep1 = unionfind_get_representative(uf, (uint32_t)(y+1)*w + x + 1);
                                if ((int)(uf->size[rep1] + 1) >= min_cluster_pixels) {
                                    gc_add_point(&ctx, rep0, rep1, 2*x + 1, 2*y + 1, vdiff, vdiff);
                                    connected = true;
                                }
                            }
                        }
                    }
                }

                connected_last = connected;

                // Fast path: while this run and the (1,1)-connected run
                // below keep overlapping, every pixel emits exactly the
                // (0,1) and (1,1) points into the same cluster, (1,0)
                // cannot fire, and (-1,1) stays suppressed by the
                // previous pixel's (1,1).
                if (connected && x + 1 <= w - 2 && rep0_state == 1) {
                    int tend = imin(a1 - 1, runs_b[pp].end - 1);
                    if (tend > x) {
                        struct uint64_zarray_entry *entry = ctx.last_entry;
                        for (int tx = x + 1; tx <= tend; tx++) {
                            gc_entry_append(&ctx, entry, 2*tx, 2*y + 1, 0, vdiff);
                            gc_entry_append(&ctx, entry, 2*tx + 1, 2*y + 1, vdiff, vdiff);
                        }
                        // resume after the batch; connected_last stays
                        // true, since pixel tend fired its (1,1)
                        x = tend;
                    }
                }
            }
        }

        // row y+1's runs and resolved representatives become row y's
        struct row_run *rt = runs_a; runs_a = runs_b; runs_b = rt;
        struct run_rep *ct = cache_a; cache_a = cache_b; cache_b = ct;
        na = nb;
    }

    free(runs_a);
    free(runs_b);
    free(cache_a);
    free(cache_b);

    struct uint64_zarray_entry **clustermap = ctx.clustermap;
    struct uint64_zarray_entry **mem_pools = ctx.mem_pools;
    int mem_pool_idx = ctx.mem_pool_idx;

    for (int i = 0; i < nclustermap; i++) {
        int start = zarray_size(clusters);
        for (struct uint64_zarray_entry *entry = clustermap[i]; entry; entry = entry->next) {
            struct cluster_hash* cluster_hash = malloc(sizeof(struct cluster_hash));
            cluster_hash->hash = i; // == u64hash_2(entry->id) & bucket_mask
            cluster_hash->id = entry->id;

            // materialize the chunk list into an exact-size zarray
            zarray_t *cl = zarray_create(sizeof(struct pt));
            zarray_ensure_capacity(cl, entry->npts);
            struct pt *dst = (struct pt*)cl->data;
            for (struct gc_chunk *c = entry->head; c; c = c->next) {
                memcpy(dst, c->pts, c->count*sizeof(struct pt));
                dst += c->count;
            }
            cl->size = entry->npts;
            cluster_hash->data = cl;

            zarray_add(clusters, &cluster_hash);
        }
        int end = zarray_size(clusters);

        // Do a quick bubblesort on the secondary key.
        int n = end - start;
        for (int j = 0; j < n - 1; j++) {
            for (int k = 0; k < n - j - 1; k++) {
                struct cluster_hash** hash1;
                struct cluster_hash** hash2;
                zarray_get_volatile(clusters, start + k, &hash1);
                zarray_get_volatile(clusters, start + k + 1, &hash2);
                if ((*hash1)->id > (*hash2)->id) {
                    struct cluster_hash tmp = **hash2;
                    **hash2 = **hash1;
                    **hash1 = tmp;
                }
            }
        }
    }
    for (int i = 0; i <= mem_pool_idx; i++) {
        free(mem_pools[i]);
    }
    free(mem_pools);
    free(clustermap);
    gc_chunk_pool_free(&ctx.chunk_pool);

    return clusters;
}

static void do_cluster_task(void *p)
{
    struct cluster_task *task = (struct cluster_task*) p;

    do_gradient_clusters(task->im, task->s, task->y0, task->y1, task->w, task->nclustermap, task->min_cluster_pixels, task->uf, task->clusters);
}

zarray_t* merge_clusters(zarray_t* c1, zarray_t* c2) {
    zarray_t* ret = zarray_create(sizeof(struct cluster_hash*));
    zarray_ensure_capacity(ret, zarray_size(c1) + zarray_size(c2));

    int i1 = 0;
    int i2 = 0;
    int l1 = zarray_size(c1);
    int l2 = zarray_size(c2);

    while (i1 < l1 && i2 < l2) {
        struct cluster_hash** h1;
        struct cluster_hash** h2;
        zarray_get_volatile(c1, i1, &h1);
        zarray_get_volatile(c2, i2, &h2);

        if ((*h1)->hash == (*h2)->hash && (*h1)->id == (*h2)->id) {
            zarray_add_range((*h1)->data, (*h2)->data, 0, zarray_size((*h2)->data));
            zarray_add(ret, h1);
            i1++;
            i2++;
            zarray_destroy((*h2)->data);
            free(*h2);
        } else if ((*h2)->hash < (*h1)->hash || ((*h2)->hash == (*h1)->hash && (*h2)->id < (*h1)->id)) {
            zarray_add(ret, h2);
            i2++;
        } else {
            zarray_add(ret, h1);
            i1++;
        }
    }

    zarray_add_range(ret, c1, i1, l1);
    zarray_add_range(ret, c2, i2, l2);

    zarray_destroy(c1);
    zarray_destroy(c2);

    return ret;
}

struct cluster_merge_task
{
    zarray_t *c1;
    zarray_t *c2;
    zarray_t *out;
};

static void do_cluster_merge_task(void *p)
{
    struct cluster_merge_task *task = (struct cluster_merge_task*) p;
    task->out = merge_clusters(task->c1, task->c2);
}

zarray_t* gradient_clusters(apriltag_detector_t *td, image_u8_t* threshim, int w, int h, int ts, unionfind_t* uf) {
    zarray_t* clusters;

    int sz = h - 1;
    int chunksize = 1 + sz / (APRILTAG_TASKS_PER_THREAD_TARGET * td->nthreads);
    struct cluster_task *tasks = malloc(sizeof(struct cluster_task)*(sz / chunksize + 1));

    // per-task hash table: power of two so lookups can mask instead of
    // divide, and sized to the slab (entry counts run well below one
    // per 64 slab pixels) so it stays cache resident.
    int nclustermap = 1024;
    while (nclustermap < chunksize*w / 64 && nclustermap < 65536)
        nclustermap <<= 1;

    int ntasks = 0;

    for (int i = 1; i < sz; i += chunksize) {
        // each task will process [y0, y1). Note that this processes
        // each cell to the right and down.
        tasks[ntasks].y0 = i;
        tasks[ntasks].y1 = imin(sz, i + chunksize);
        tasks[ntasks].w = w;
        tasks[ntasks].s = ts;
        tasks[ntasks].uf = uf;
        tasks[ntasks].im = threshim;
        tasks[ntasks].nclustermap = nclustermap;
        tasks[ntasks].min_cluster_pixels = td->qtp.min_cluster_pixels;
        tasks[ntasks].clusters = zarray_create(sizeof(struct cluster_hash*));

        workerpool_add_task(td->wp, do_cluster_task, &tasks[ntasks]);
        ntasks++;
    }

    workerpool_run(td->wp);

    zarray_t** clusters_list = malloc(sizeof(zarray_t *)*ntasks);
    for (int i = 0; i < ntasks; i++) {
        clusters_list[i] = tasks[i].clusters;
    }

    struct cluster_merge_task *mtasks = malloc(sizeof(struct cluster_merge_task)*(ntasks/2 + 1));

    int length = ntasks;
    while (length > 1) {
        int npairs = length / 2;
        for (int i = 0; i < npairs; i++) {
            mtasks[i].c1 = clusters_list[2*i];
            mtasks[i].c2 = clusters_list[2*i + 1];
            workerpool_add_task(td->wp, do_cluster_merge_task, &mtasks[i]);
        }
        workerpool_run(td->wp);

        for (int i = 0; i < npairs; i++) {
            clusters_list[i] = mtasks[i].out;
        }
        if (length % 2) {
            clusters_list[npairs] = clusters_list[length - 1];
        }

        length = npairs + length % 2;
    }

    free(mtasks);

    clusters = zarray_create(sizeof(zarray_t*));
    zarray_ensure_capacity(clusters, zarray_size(clusters_list[0]));
    for (int i = 0; i < zarray_size(clusters_list[0]); i++) {
        struct cluster_hash** hash;
        zarray_get_volatile(clusters_list[0], i, &hash);
        zarray_add(clusters, &(*hash)->data);
        free(*hash);
    }
    zarray_destroy(clusters_list[0]);
    free(clusters_list);
    free(tasks);
    return clusters;
}

zarray_t* fit_quads(apriltag_detector_t *td, int w, int h, zarray_t* clusters, image_u8_t* im) {
    zarray_t *quads = zarray_create(sizeof(struct quad));

    bool normal_border = false;
    bool reversed_border = false;
    int min_tag_width = 1000000;
    for (int i = 0; i < zarray_size(td->tag_families); i++) {
        apriltag_family_t* family;
        zarray_get(td->tag_families, i, &family);
        if (family->width_at_border < min_tag_width) {
            min_tag_width = family->width_at_border;
        }
        normal_border |= !family->reversed_border;
        reversed_border |= family->reversed_border;
    }
    if (td->quad_decimate > 1)
        min_tag_width /= td->quad_decimate;
    if (min_tag_width < 3) {
        min_tag_width = 3;
    }

    int sz = zarray_size(clusters);
    int chunksize = 1 + sz / (APRILTAG_TASKS_PER_THREAD_TARGET * td->nthreads);
    struct quad_task *tasks = malloc(sizeof(struct quad_task)*(sz / chunksize + 1));

    int ntasks = 0;
    for (int i = 0; i < sz; i += chunksize) {
        tasks[ntasks].td = td;
        tasks[ntasks].cidx0 = i;
        tasks[ntasks].cidx1 = imin(sz, i + chunksize);
        tasks[ntasks].h = h;
        tasks[ntasks].w = w;
        tasks[ntasks].quads = quads;
        tasks[ntasks].clusters = clusters;
        tasks[ntasks].im = im;
        tasks[ntasks].tag_width = min_tag_width;
        tasks[ntasks].normal_border = normal_border;
        tasks[ntasks].reversed_border = reversed_border;

        workerpool_add_task(td->wp, do_quad_task, &tasks[ntasks]);
        ntasks++;
    }

    workerpool_run(td->wp);

    free(tasks);

    return quads;
}

zarray_t *apriltag_quad_thresh(apriltag_detector_t *td, image_u8_t *im)
{
    ////////////////////////////////////////////////////////
    // step 1. threshold the image, creating the edge image.

    int w = im->width, h = im->height;

    image_u8_t *threshim = threshold(td, im);
    int ts = threshim->stride;

    if (td->debug)
        image_u8_write_pnm(threshim, "debug_threshold.pnm");


    ////////////////////////////////////////////////////////
    // step 2. find connected components.
    unionfind_t* uf = connected_components(td, threshim, w, h, ts);

    // make segmentation image.
    if (td->debug) {
        image_u8x3_t *d = image_u8x3_create(w, h);

        uint32_t *colors = (uint32_t*) calloc(w*h, sizeof(*colors));

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                uint32_t v = unionfind_get_representative(uf, y*w+x);

                if ((int)unionfind_get_set_size(uf, v) < td->qtp.min_cluster_pixels)
                    continue;

                uint32_t color = colors[v];
                uint8_t r = color >> 16,
                    g = color >> 8,
                    b = color;

                if (color == 0) {
                    const int bias = 50;
                    r = bias + (random() % (200-bias));
                    g = bias + (random() % (200-bias));
                    b = bias + (random() % (200-bias));
                    colors[v] = (r << 16) | (g << 8) | b;
                }

                d->buf[y*d->stride + 3*x + 0] = r;
                d->buf[y*d->stride + 3*x + 1] = g;
                d->buf[y*d->stride + 3*x + 2] = b;
            }
        }

        free(colors);

        image_u8x3_write_pnm(d, "debug_segmentation.pnm");
        image_u8x3_destroy(d);
    }


    timeprofile_stamp(td->tp, "unionfind");

    zarray_t* clusters = gradient_clusters(td, threshim, w, h, ts, uf);

    if (td->debug) {
        image_u8x3_t *d = image_u8x3_create(w, h);

        for (int i = 0; i < zarray_size(clusters); i++) {
            zarray_t *cluster;
            zarray_get(clusters, i, &cluster);

            uint32_t r, g, b;

            if (1) {
                const int bias = 50;
                r = bias + (random() % (200-bias));
                g = bias + (random() % (200-bias));
                b = bias + (random() % (200-bias));
            }

            for (int j = 0; j < zarray_size(cluster); j++) {
                struct pt *p;
                zarray_get_volatile(cluster, j, &p);

                int x = p->x / 2;
                int y = p->y / 2;
                d->buf[y*d->stride + 3*x + 0] = r;
                d->buf[y*d->stride + 3*x + 1] = g;
                d->buf[y*d->stride + 3*x + 2] = b;
            }
        }

        image_u8x3_write_pnm(d, "debug_clusters.pnm");
        image_u8x3_destroy(d);
    }


    // threshim is cached on the detector and reused next frame
    timeprofile_stamp(td->tp, "make clusters");

    ////////////////////////////////////////////////////////
    // step 3. process each connected component.

    zarray_t* quads = fit_quads(td, w, h, clusters, im);

    if (td->debug) {
        FILE *f = fopen("debug_lines.ps", "w");
        fprintf(f, "%%!PS\n\n");

        image_u8_t *im2 = image_u8_copy(im);
        image_u8_darken(im2);
        image_u8_darken(im2);

        // assume letter, which is 612x792 points.
        double scale = fmin(612.0/im->width, 792.0/im2->height);
        fprintf(f, "%.15f %.15f scale\n", scale, scale);
        fprintf(f, "0 %d translate\n", im2->height);
        fprintf(f, "1 -1 scale\n");

        postscript_image(f, im2);

        image_u8_destroy(im2);

        for (int i = 0; i < zarray_size(quads); i++) {
            struct quad *q;
            zarray_get_volatile(quads, i, &q);

            float rgb[3];
            int bias = 100;

            for (int j = 0; j < 3; j++)
                rgb[j] = bias + (random() % (255-bias));

            fprintf(f, "%f %f %f setrgbcolor\n", rgb[0]/255.0f, rgb[1]/255.0f, rgb[2]/255.0f);
            fprintf(f, "%.15f %.15f moveto %.15f %.15f lineto %.15f %.15f lineto %.15f %.15f lineto %.15f %.15f lineto stroke\n",
                    q->p[0][0], q->p[0][1],
                    q->p[1][0], q->p[1][1],
                    q->p[2][0], q->p[2][1],
                    q->p[3][0], q->p[3][1],
                    q->p[0][0], q->p[0][1]);
        }

        fclose(f);
    }

    timeprofile_stamp(td->tp, "fit quads to clusters");

    // individual clusters were destroyed by the quad tasks
    zarray_destroy(clusters);

    return quads;
}
