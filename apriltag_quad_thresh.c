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

#ifdef __AVX2__
#include <immintrin.h>
#endif

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
};

// a finished cluster: header and points in one allocation
struct pt_list
{
    int size;
    int pad; // keep pts 8-byte aligned
    struct pt pts[];
};

// Cluster points are accumulated in fixed-size chunks bump-allocated from
// a per-task pool: appending is a bounds check and a store, with none of
// the doubling reallocs a growing array needs (a frame can produce
// millions of points). Clusters are materialized into exact-size zarrays
// once their final length is known.
#define GC_CHUNK_PTS 64
struct gc_chunk
{
    struct gc_chunk *next;
    int count;
    int pad; // keep pts 8-byte aligned: points are stored as packed u64s
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

struct row_run; // see below

struct unionfind_task
{
    int y0, y1;
    int w, h, s;
    unionfind_t *uf;
    image_u8_t *im;
    struct row_run *runs;
    uint32_t *row_off;
    uint32_t vcol_base;
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
    struct cluster_hash *ch_pool; // backing storage for clusters entries
    struct row_run *runs;
    uint32_t *row_off;
    uint32_t vcol_base;
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

// Cumulative line-fit moments in SoA layout: entry i of each array is the
// sum over points [0, i]. Contiguous per-field arrays let the window-error
// loop vectorize.
struct lfps_soa
{
    double *Mx, *My;
    double *Mxx, *Myy, *Mxy;
    double *W; // total weight
};

struct cluster_hash
{
    uint32_t hash;
    uint64_t id;
    struct pt_list *data;
};

// scratch buffers reused across all clusters processed by one quad task,
// so fit_quad doesn't malloc/free per cluster.
struct quad_fit_scratch
{
    int capacity; // in points
    double *lf_block; // 6 contiguous arrays backing lfps
    struct lfps_soa lfps;
    double *errs;
    double *yfilt;
    int *maxima;
    double *maxima_errs;
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
    free(scratch->lf_block);
    free(scratch->errs);
    free(scratch->yfilt);
    free(scratch->maxima);
    free(scratch->maxima_errs);
    free(scratch->sort_keys);
    free(scratch->sort_tmp);
    scratch->lf_block = malloc(sizeof(double)*6*cap);
    scratch->lfps.Mx  = scratch->lf_block;
    scratch->lfps.My  = scratch->lf_block + cap;
    scratch->lfps.Mxx = scratch->lf_block + 2*cap;
    scratch->lfps.Mxy = scratch->lf_block + 3*cap;
    scratch->lfps.Myy = scratch->lf_block + 4*cap;
    scratch->lfps.W   = scratch->lf_block + 5*cap;
    scratch->errs = malloc(sizeof(double)*cap);
    scratch->yfilt = malloc(sizeof(double)*cap);
    scratch->maxima = malloc(sizeof(int)*cap);
    scratch->maxima_errs = malloc(sizeof(double)*cap);
    scratch->sort_keys = malloc(sizeof(uint64_t)*cap);
    scratch->sort_tmp = malloc(sizeof(uint64_t)*cap);
    scratch->capacity = cap;
}

static void quad_fit_scratch_free(struct quad_fit_scratch *scratch)
{
    free(scratch->lf_block);
    free(scratch->errs);
    free(scratch->yfilt);
    free(scratch->maxima);
    free(scratch->maxima_errs);
    free(scratch->sort_keys);
    free(scratch->sort_tmp);
}


// lfps contains *cumulative* moments for N points, with
// index j reflecting points [0,j] (inclusive).
//
// fit a line to the points [i0, i1] (inclusive). i0, i1 are both [0,
// sz) if i1 < i0, we treat this as a wrap around.
void fit_line(const struct lfps_soa *L, int sz, int i0, int i1, double *lineparm, double *err, double *mse)
{
    assert(i0 != i1);
    assert(i0 >= 0 && i1 >= 0 && i0 < sz && i1 < sz);

    double Mx, My, Mxx, Myy, Mxy, W;
    int N; // how many points are included in the set?

    if (i0 < i1) {
        N = i1 - i0 + 1;

        Mx  = L->Mx[i1];
        My  = L->My[i1];
        Mxx = L->Mxx[i1];
        Mxy = L->Mxy[i1];
        Myy = L->Myy[i1];
        W   = L->W[i1];

        if (i0 > 0) {
            Mx  -= L->Mx[i0-1];
            My  -= L->My[i0-1];
            Mxx -= L->Mxx[i0-1];
            Mxy -= L->Mxy[i0-1];
            Myy -= L->Myy[i0-1];
            W   -= L->W[i0-1];
        }

    } else {
        // i0 > i1, e.g. [15, 2]. Wrap around.
        assert(i0 > 0);

        Mx  = L->Mx[sz-1]   - L->Mx[i0-1];
        My  = L->My[sz-1]   - L->My[i0-1];
        Mxx = L->Mxx[sz-1]  - L->Mxx[i0-1];
        Mxy = L->Mxy[sz-1]  - L->Mxy[i0-1];
        Myy = L->Myy[sz-1]  - L->Myy[i0-1];
        W   = L->W[sz-1]    - L->W[i0-1];

        Mx  += L->Mx[i1];
        My  += L->My[i1];
        Mxx += L->Mxx[i1];
        Mxy += L->Mxy[i1];
        Myy += L->Myy[i1];
        W   += L->W[i1];

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

static inline struct pair_fit *pair_fit_get(const struct lfps_soa *lfps, int sz, int *maxima, int nmaxima,
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

int quad_segment_maxima(apriltag_detector_t *td, int sz, const struct lfps_soa *lfps, int indices[4],
                        struct quad_fit_scratch *scratch)
{

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
    {
        const double *aMx = lfps->Mx, *aMy = lfps->My, *aMxx = lfps->Mxx;
        const double *aMxy = lfps->Mxy, *aMyy = lfps->Myy, *aW = lfps->W;
        int i = mid_lo;

#ifdef __AVX2__
        // 4 windows per iteration; the SoA loads are contiguous and every
        // step is a single exact-rounded operation, including the
        // double->float->sqrtf->double sequence of the scalar code
        const __m256d half = _mm256_set1_pd(0.5);
        const __m256d four = _mm256_set1_pd(4.0);
        const __m256d vN = _mm256_set1_pd((double)N);
        for (; i + 4 <= mid_hi + 1; i += 4) {
            int u = i + ksz, l = i - ksz - 1;

            __m256d Mx  = _mm256_sub_pd(_mm256_loadu_pd(&aMx[u]),  _mm256_loadu_pd(&aMx[l]));
            __m256d My  = _mm256_sub_pd(_mm256_loadu_pd(&aMy[u]),  _mm256_loadu_pd(&aMy[l]));
            __m256d Mxx = _mm256_sub_pd(_mm256_loadu_pd(&aMxx[u]), _mm256_loadu_pd(&aMxx[l]));
            __m256d Mxy = _mm256_sub_pd(_mm256_loadu_pd(&aMxy[u]), _mm256_loadu_pd(&aMxy[l]));
            __m256d Myy = _mm256_sub_pd(_mm256_loadu_pd(&aMyy[u]), _mm256_loadu_pd(&aMyy[l]));
            __m256d W   = _mm256_sub_pd(_mm256_loadu_pd(&aW[u]),   _mm256_loadu_pd(&aW[l]));

            __m256d Ex = _mm256_div_pd(Mx, W);
            __m256d Ey = _mm256_div_pd(My, W);
            __m256d Cxx = _mm256_sub_pd(_mm256_div_pd(Mxx, W), _mm256_mul_pd(Ex, Ex));
            __m256d Cxy = _mm256_sub_pd(_mm256_div_pd(Mxy, W), _mm256_mul_pd(Ex, Ey));
            __m256d Cyy = _mm256_sub_pd(_mm256_div_pd(Myy, W), _mm256_mul_pd(Ey, Ey));

            __m256d d = _mm256_sub_pd(Cxx, Cyy);
            __m256d rad = _mm256_add_pd(_mm256_mul_pd(d, d),
                                        _mm256_mul_pd(four, _mm256_mul_pd(Cxy, Cxy)));
            // sqrtf semantics: round to float, sqrt in float, widen back
            __m256d root = _mm256_cvtps_pd(_mm_sqrt_ps(_mm256_cvtpd_ps(rad)));

            __m256d eig = _mm256_mul_pd(half,
                          _mm256_sub_pd(_mm256_add_pd(Cxx, Cyy), root));
            _mm256_storeu_pd(&errs[i], _mm256_mul_pd(vN, eig));
        }
#endif

        for (; i <= mid_hi; i++) {
            int u = i + ksz, l = i - ksz - 1;

            double Mx  = aMx[u]  - aMx[l];
            double My  = aMy[u]  - aMy[l];
            double Mxx = aMxx[u] - aMxx[l];
            double Mxy = aMxy[u] - aMxy[l];
            double Myy = aMyy[u] - aMyy[l];
            double W   = aW[u]   - aW[l];

            double Ex = Mx / W;
            double Ey = My / W;
            double Cxx = Mxx / W - Ex*Ex;
            double Cxy = Mxy / W - Ex*Ey;
            double Cyy = Myy / W - Ey*Ey;

            double eig_small = 0.5*(Cxx + Cyy - sqrtf((Cxx - Cyy)*(Cxx - Cyy) + 4*Cxy*Cxy));
            errs[i] = N*eig_small;
        }
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
        int iy = 0;

#ifdef __AVX2__
        // middle outputs need no wrap handling; 4 outputs per iteration
        // with the same per-output tap order as the scalar code
        {
            __m256d k0 = _mm256_set1_pd(qsm_kernel[0]);
            __m256d k1 = _mm256_set1_pd(qsm_kernel[1]);
            __m256d k2 = _mm256_set1_pd(qsm_kernel[2]);
            __m256d k3 = _mm256_set1_pd(qsm_kernel[3]);
            __m256d k4 = _mm256_set1_pd(qsm_kernel[4]);
            __m256d k5 = _mm256_set1_pd(qsm_kernel[5]);
            __m256d k6 = _mm256_set1_pd(qsm_kernel[6]);

            for (iy = QSM_FSZ/2; iy + 4 <= sz - QSM_FSZ/2; iy += 4) {
                const double *base = &errs[iy - QSM_FSZ/2];
                __m256d acc = _mm256_mul_pd(_mm256_loadu_pd(base), k0);
                acc = _mm256_add_pd(acc, _mm256_mul_pd(_mm256_loadu_pd(base + 1), k1));
                acc = _mm256_add_pd(acc, _mm256_mul_pd(_mm256_loadu_pd(base + 2), k2));
                acc = _mm256_add_pd(acc, _mm256_mul_pd(_mm256_loadu_pd(base + 3), k3));
                acc = _mm256_add_pd(acc, _mm256_mul_pd(_mm256_loadu_pd(base + 4), k4));
                acc = _mm256_add_pd(acc, _mm256_mul_pd(_mm256_loadu_pd(base + 5), k5));
                acc = _mm256_add_pd(acc, _mm256_mul_pd(_mm256_loadu_pd(base + 6), k6));
                _mm256_storeu_pd(&y[iy], acc);
            }
            // the scalar loop below covers [0, QSM_FSZ/2), the vector tail,
            // and the wrapped end region
            for (int e = 0; e < QSM_FSZ/2; e++) {
                int j = e - QSM_FSZ / 2 + sz;
                double acc = 0;
                for (int i = 0; i < QSM_FSZ; i++) {
                    acc += errs[j] * qsm_kernel[i];
                    if (++j == sz)
                        j = 0;
                }
                y[e] = acc;
            }
        }
#endif

        for (; iy < sz; iy++) {
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

    int mi = 0;

#ifdef __AVX2__
    // interior positions, 4 at a time: local maxima as a movemask, then
    // iterate its set bits (each compare result occupies 2 mask bits)
    {
        // wrap position i = 0 first
        if (errs[0] > errs[1] && errs[0] > errs[sz-1]) {
            maxima[nmaxima] = 0;
            maxima_errs[nmaxima] = errs[0];
            nmaxima++;
        }
        for (mi = 1; mi + 4 <= sz - 1; mi += 4) {
            __m256d e = _mm256_loadu_pd(&errs[mi]);
            __m256d gt_next = _mm256_cmp_pd(e, _mm256_loadu_pd(&errs[mi+1]), _CMP_GT_OQ);
            __m256d gt_prev = _mm256_cmp_pd(e, _mm256_loadu_pd(&errs[mi-1]), _CMP_GT_OQ);
            int m = _mm256_movemask_pd(_mm256_and_pd(gt_next, gt_prev));
            while (m) {
                int b = __builtin_ctz(m);
                m &= m - 1;
                maxima[nmaxima] = mi + b;
                maxima_errs[nmaxima] = errs[mi + b];
                nmaxima++;
            }
        }
    }
#endif

    // without AVX2 this covers everything from 0; with it, the tail
    for (int i = mi; i < sz; i++) {
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
        // throw out all but the best handful of maxima: the threshold is
        // the (max_nmaxima+1)-th largest error -- the same order statistic
        // the historical descending qsort selected -- found by quickselect
        double *sel = scratch->yfilt; // free again at this point
        memcpy(sel, maxima_errs, sizeof(double)*nmaxima);

        int k = max_nmaxima; // index in descending order
        int lo = 0, hi = nmaxima - 1;
        while (lo < hi) {
            double pivot = sel[(lo + hi) / 2];
            int i = lo, j = hi;
            while (i <= j) {
                while (sel[i] > pivot)
                    i++;
                while (sel[j] < pivot)
                    j--;
                if (i <= j) {
                    double tmp = sel[i];
                    sel[i] = sel[j];
                    sel[j] = tmp;
                    i++;
                    j--;
                }
            }
            if (k <= j)
                hi = j;
            else if (k >= i)
                lo = i;
            else
                break;
        }
        double maxima_thresh = sel[k];

        int out = 0;
        for (int in = 0; in < nmaxima; in++) {
            if (maxima_errs[in] <= maxima_thresh)
                continue;
            maxima[out++] = maxima[in];
        }
        nmaxima = out;
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
int quad_segment_agg(int sz, const struct lfps_soa *lfps, int indices[4])
{

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
// Accumulate the cumulative line-fit moments in angle-sorted order: entry i
// covers the points whose sort keys are keys[0..i] (the low key word holds
// the complemented index into pts). fxbuf/fybuf/wbuf are sz-sized scratch.
void compute_lfps(int sz, struct pt *pts, const uint64_t *keys, image_u8_t* im, const struct lfps_soa *lfps,
                  double *fxbuf, double *fybuf, double *wbuf) {
    // pass 1: per-point coordinates and gradient weights. The weight is
    // sqrt(grad^2)+1, with out-of-bounds points using grad = 0 so the same
    // expression yields exactly 1.
    int i = 0;

#ifdef __AVX2__
    {
        double g2[4];
        const uint8_t *ibuf = im->buf;
        int iw = im->width, ih = im->height, istride = im->stride;
        for (; i + 4 <= sz; i += 4) {
            for (int j = 0; j < 4; j++) {
                uint64_t pv;
                memcpy(&pv, &pts[i+j], sizeof(pv));
                unsigned px = pv & 0xffff;
                unsigned py = (pv >> 16) & 0xffff;
                // ix == (int)(px*0.5 + 0.5) for px >= 0, without the
                // double round trip
                int ix = (int)((px >> 1) + (px & 1));
                int iy = (int)((py >> 1) + (py & 1));
                fxbuf[i+j] = px * .5 + 0.5;
                fybuf[i+j] = py * .5 + 0.5;

                if (ix > 0 && ix+1 < iw && iy > 0 && iy+1 < ih) {
                    int grad_x = ibuf[iy * istride + ix + 1] -
                        ibuf[iy * istride + ix - 1];
                    int grad_y = ibuf[(iy+1) * istride + ix] -
                        ibuf[(iy-1) * istride + ix];
                    g2[j] = grad_x*grad_x + grad_y*grad_y;
                } else {
                    g2[j] = 0;
                }
            }
            __m256d w = _mm256_sqrt_pd(_mm256_loadu_pd(g2));
            _mm256_storeu_pd(&wbuf[i], _mm256_add_pd(w, _mm256_set1_pd(1.0)));
        }
    }
#endif

    for (; i < sz; i++) {
        struct pt *p = &pts[i];

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

        fxbuf[i] = x;
        fybuf[i] = y;
        wbuf[i] = W;
    }

    // pass 2: cumulative sums, in the same per-point order and with the
    // same operation order as the historical single loop
    double sum_Mx = 0, sum_My = 0, sum_Mxx = 0, sum_Myy = 0, sum_Mxy = 0, sum_W = 0;

    for (int k = 0; k < sz; k++) {
        uint32_t srcidx = ~(uint32_t)keys[k];
        double W = wbuf[srcidx];
        double fx = fxbuf[srcidx], fy = fybuf[srcidx];
        sum_Mx  += W * fx;
        sum_My  += W * fy;
        sum_Mxx += W * fx * fx;
        sum_Mxy += W * fx * fy;
        sum_Myy += W * fy * fy;
        sum_W   += W;

        // Store cumulative sums
        lfps->Mx[k] = sum_Mx;
        lfps->My[k] = sum_My;
        lfps->Mxx[k] = sum_Mxx;
        lfps->Mxy[k] = sum_Mxy;
        lfps->Myy[k] = sum_Myy;
        lfps->W[k] = sum_W;
    }
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

// Fused two-level merge: out gets exactly what
//   key_merge(key_merge(l1,l2), key_merge(r1,r2))
// would produce, without materializing the intermediate runs. The virtual
// left head replicates the inner merge's comparison (take l1 iff
// l1[i] < l2[j]), and the outer comparison takes the right stream on
// not-less, like key_merge.
static void key_merge4(const uint64_t *l1, int n1, const uint64_t *l2, int n2,
                       const uint64_t *r1, int n3, const uint64_t *r2, int n4,
                       uint64_t *out)
{
    int i1 = 0, i2 = 0, i3 = 0, i4 = 0, o = 0;

    while (i1 < n1 && i2 < n2 && i3 < n3 && i4 < n4) {
        uint64_t v1 = l1[i1], v2 = l2[i2], v3 = r1[i3], v4 = r2[i4];
        int lt1 = v1 < v2;
        int rt1 = v3 < v4;
        uint64_t lv = lt1 ? v1 : v2;
        uint64_t rv = rt1 ? v3 : v4;
        int tl = lv < rv;
        out[o++] = tl ? lv : rv;
        i1 += tl & lt1;
        i2 += tl & (!lt1);
        i3 += (!tl) & rt1;
        i4 += (!tl) & (!rt1);
    }

    // some run is empty: continue with bounds-checked virtual streams
    while ((i1 < n1 || i2 < n2) && (i3 < n3 || i4 < n4)) {
        int lt1 = i1 < n1 && (i2 >= n2 || l1[i1] < l2[i2]);
        int rt1 = i3 < n3 && (i4 >= n4 || r1[i3] < r2[i4]);
        uint64_t lv = lt1 ? l1[i1] : l2[i2];
        uint64_t rv = rt1 ? r1[i3] : r2[i4];
        if (lv < rv) {
            out[o++] = lv;
            i1 += lt1;
            i2 += !lt1;
        } else {
            out[o++] = rv;
            i3 += rt1;
            i4 += !rt1;
        }
    }

    // one side fully drained: finish the other with the plain 2-way merge
    if (i1 < n1 || i2 < n2)
        key_merge((uint64_t*)l1 + i1, n1 - i1, (uint64_t*)l2 + i2, n2 - i2, out + o);
    else
        key_merge((uint64_t*)r1 + i3, n3 - i3, (uint64_t*)r2 + i4, n4 - i4, out + o);
}

// Fused-tree drivers: same splits, same <=5 leaf networks, and merge
// results identical to the 2-way ping-pong's, but two merge levels
// collapse into one pass over the data.
static void keysort4_move(uint64_t *A, uint64_t *B, int sz);

static void keysort4_in_place(uint64_t *A, uint64_t *tmp, int sz)
{
    if (sz <= 5) {
        key_network_sort(A, sz);
        return;
    }

    int asz = sz/2, bsz = sz - asz;
    if (asz <= 5) {
        keysort_move(A, tmp, asz);
        keysort_move(A + asz, tmp + asz, bsz);
        key_merge(tmp, asz, tmp + asz, bsz, A);
        return;
    }

    int a1 = asz/2, a2 = asz - a1;
    int b1 = bsz/2, b2 = bsz - b1;
    keysort4_move(A, tmp, a1);
    keysort4_move(A + a1, tmp + a1, a2);
    keysort4_move(A + asz, tmp + asz, b1);
    keysort4_move(A + asz + b1, tmp + asz + b1, b2);
    key_merge4(tmp, a1, tmp + a1, a2, tmp + asz, b1, tmp + asz + b1, b2, A);
}

static void keysort4_move(uint64_t *A, uint64_t *B, int sz)
{
    if (sz <= 5) {
        key_network_sort(A, sz);
        memcpy(B, A, sz*sizeof(uint64_t));
        return;
    }

    int asz = sz/2, bsz = sz - asz;
    if (asz <= 5) {
        keysort_in_place(A, B, asz);
        keysort_in_place(A + asz, B + asz, bsz);
        key_merge(A, asz, A + asz, bsz, B);
        return;
    }

    int a1 = asz/2, a2 = asz - a1;
    int b1 = bsz/2, b2 = bsz - b1;
    keysort4_in_place(A, B, a1);
    keysort4_in_place(A + a1, B + a1, a2);
    keysort4_in_place(A + asz, B + asz, b1);
    keysort4_in_place(A + asz + b1, B + asz + b1, b2);
    key_merge4(A, a1, A + a1, a2, A + asz, b1, A + asz + b1, b2, B);
}

// Sort the keys in scratch->sort_keys. The point array itself is left
// untouched: the only consumer of the sorted order is compute_lfps, which
// reads points through the key indices.
static void pt_key_sort(int sz, struct quad_fit_scratch *scratch)
{
    if (sz < 2)
        return;

    keysort4_in_place(scratch->sort_keys, scratch->sort_tmp, sz);
}

// return 1 if the quad looks okay, 0 if it should be discarded
int fit_quad(
        apriltag_detector_t *td,
        image_u8_t *im,
        struct pt_list *cluster,
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
    struct pt *pts = cluster->pts;
    int szc = cluster->size;
    uint16_t xmax = pts[0].x;
    uint16_t xmin = pts[0].x;
    uint16_t ymax = pts[0].y;
    uint16_t ymin = pts[0].y;
    int pidx = 1;

#ifdef __AVX2__
    // 4 points per vector; x sits in u16 lanes 0,4,8,12 and y in
    // 1,5,9,13 (gx/gy lanes are reduced too but ignored)
    if (szc - pidx >= 8) {
        __m256i vmn = _mm256_set1_epi16(-1);
        __m256i vmx = _mm256_setzero_si256();
        for (; pidx + 4 <= szc; pidx += 4) {
            __m256i v = _mm256_loadu_si256((const __m256i*)&pts[pidx]);
            vmn = _mm256_min_epu16(vmn, v);
            vmx = _mm256_max_epu16(vmx, v);
        }
        uint16_t tmn[16], tmx[16];
        _mm256_storeu_si256((__m256i*)tmn, vmn);
        _mm256_storeu_si256((__m256i*)tmx, vmx);
        for (int k = 0; k < 16; k += 4) {
            if (tmn[k] < xmin) xmin = tmn[k];
            if (tmn[k+1] < ymin) ymin = tmn[k+1];
            if (tmx[k] > xmax) xmax = tmx[k];
            if (tmx[k+1] > ymax) ymax = tmx[k+1];
        }
    }
#endif

    for (; pidx < szc; pidx++) {
        struct pt *p = &pts[pidx];

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

    int sz = cluster->size;
    quad_fit_scratch_ensure(scratch, sz);

    // add some noise to (cx,cy) so that pixels get a more diverse set
    // of theta estimates. This will help us remove more points.
    // (Only helps a small amount. The actual noise values here don't
    // matter much at all, but we want them [-1, 1]. (XXX with
    // fixed-point, should range be bigger?)
    float cx = (xmin + xmax) * 0.5 + 0.05118;
    float cy = (ymin + ymax) * 0.5 + -0.028581;

    float dot = 0;

    float quadrants[2][2] = {{-1*(2 << 15), 0}, {2*(2 << 15), 2 << 15}};

    uint64_t *keys = scratch->sort_keys;

    pidx = 0;

#ifdef __AVX2__
    // 8 points per iteration. Every step of the key computation is a
    // single exact-rounded operation or a bit-level select, so the keys
    // match the scalar computation bit for bit. The dot accumulates in 8
    // lanes (summed at the end); it only decides the border-orientation
    // sign, which is far from zero for any usable cluster.
    {
        const __m256i xmask16 = _mm256_set1_epi32(0xffff);
        __m256 dotacc = _mm256_setzero_ps();
        const __m256 vcx = _mm256_set1_ps(cx);
        const __m256 vcy = _mm256_set1_ps(cy);
        const __m256 q00 = _mm256_set1_ps(quadrants[0][0]);
        const __m256 q01 = _mm256_set1_ps(quadrants[0][1]);
        const __m256 q10 = _mm256_set1_ps(quadrants[1][0]);
        const __m256 q11 = _mm256_set1_ps(quadrants[1][1]);
        const __m256 zero = _mm256_setzero_ps();
        const __m256 signbit = _mm256_set1_ps(-0.0f);
        const __m256i idx_base = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

        for (; pidx + 8 <= sz; pidx += 8) {
            // deinterleave x and y from 8 packed 8-byte points
            __m256i v0 = _mm256_loadu_si256((const __m256i*)&pts[pidx]);     // pts 0..3
            __m256i v1 = _mm256_loadu_si256((const __m256i*)&pts[pidx + 4]); // pts 4..7
            __m256i x0 = _mm256_and_si256(v0, xmask16);            // u32 lanes: x0,gx0,x1,gx1,...
            __m256i x1 = _mm256_and_si256(v1, xmask16);
            __m256i y0 = _mm256_srli_epi32(v0, 16);                // y0,gy0,y1,gy1,...
            __m256i y1 = _mm256_srli_epi32(v1, 16);
            // keep even u32 lanes (the x/y values), pack 8 together; the
            // blend leaves lanes in order p0,p4,p1,p5,p2,p6,p3,p7, which
            // the permute restores to p0..p7
            const __m256i unshuf = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
            __m256i xs = _mm256_permutevar8x32_epi32(
                _mm256_blend_epi32(x0, _mm256_slli_epi64(x1, 32), 0xaa), unshuf);
            __m256i ys = _mm256_permutevar8x32_epi32(
                _mm256_blend_epi32(y0, _mm256_slli_epi64(y1, 32), 0xaa), unshuf);

            __m256 dx = _mm256_sub_ps(_mm256_cvtepi32_ps(xs), vcx);
            __m256 dy = _mm256_sub_ps(_mm256_cvtepi32_ps(ys), vcy);

            // gradients live in the odd u32 lanes; pack like xs/ys and
            // sign-extend the 16-bit values
            __m256i gxs = _mm256_permutevar8x32_epi32(
                _mm256_blend_epi32(_mm256_srli_epi64(x0, 32), x1, 0xaa), unshuf);
            __m256i gys = _mm256_permutevar8x32_epi32(
                _mm256_blend_epi32(_mm256_srli_epi64(y0, 32), y1, 0xaa), unshuf);
            gxs = _mm256_srai_epi32(_mm256_slli_epi32(gxs, 16), 16);
            gys = _mm256_srai_epi32(_mm256_slli_epi32(gys, 16), 16);

            // dot += dx*gx + dy*gy, accumulated per lane
            dotacc = _mm256_add_ps(dotacc,
                     _mm256_add_ps(_mm256_mul_ps(dx, _mm256_cvtepi32_ps(gxs)),
                                   _mm256_mul_ps(dy, _mm256_cvtepi32_ps(gys))));

            __m256 dxpos = _mm256_cmp_ps(dx, zero, _CMP_GT_OQ);
            __m256 dypos = _mm256_cmp_ps(dy, zero, _CMP_GT_OQ);
            __m256 quadrant = _mm256_blendv_ps(
                _mm256_blendv_ps(q00, q01, dxpos),
                _mm256_blendv_ps(q10, q11, dxpos),
                dypos);

            // if (dy < 0) negate both
            __m256 dyneg = _mm256_cmp_ps(dy, zero, _CMP_LT_OQ);
            __m256 flip = _mm256_and_ps(dyneg, signbit);
            dx = _mm256_xor_ps(dx, flip);
            dy = _mm256_xor_ps(dy, flip);

            // if (dx < 0) rotate: dx' = dy, dy' = -dx
            __m256 dxneg = _mm256_cmp_ps(dx, zero, _CMP_LT_OQ);
            __m256 ndx = _mm256_blendv_ps(dx, dy, dxneg);
            __m256 ndy = _mm256_blendv_ps(dy, _mm256_xor_ps(dx, signbit), dxneg);

            __m256 slope = _mm256_add_ps(quadrant, _mm256_div_ps(ndy, ndx));

            // monotone float-bits -> u32 key transform
            __m256i bits = _mm256_castps_si256(slope);
            __m256i sgn = _mm256_srai_epi32(bits, 31);
            __m256i key = _mm256_xor_si256(bits,
                          _mm256_or_si256(sgn, _mm256_set1_epi32(0x80000000)));

            // complemented original indices (lanes are in point order)
            __m256i idx = _mm256_add_epi32(_mm256_set1_epi32(pidx), idx_base);
            __m256i nidx = _mm256_xor_si256(idx, _mm256_set1_epi32(-1));

            // interleave (key << 32) | ~index into u64 lanes:
            // lo = [K0,K1 | K4,K5], hi = [K2,K3 | K6,K7]
            __m256i lo = _mm256_unpacklo_epi32(nidx, key);
            __m256i hi = _mm256_unpackhi_epi32(nidx, key);
            _mm256_storeu_si256((__m256i*)&keys[pidx],
                                _mm256_permute2x128_si256(lo, hi, 0x20));
            _mm256_storeu_si256((__m256i*)&keys[pidx + 4],
                                _mm256_permute2x128_si256(lo, hi, 0x31));
        }

        // horizontal sum of the lane accumulators
        __m128 d4 = _mm_add_ps(_mm256_castps256_ps128(dotacc),
                               _mm256_extractf128_ps(dotacc, 1));
        d4 = _mm_add_ps(d4, _mm_movehl_ps(d4, d4));
        d4 = _mm_add_ss(d4, _mm_shuffle_ps(d4, d4, 1));
        dot += _mm_cvtss_f32(d4);
    }
#endif

    for (; pidx < sz; pidx++) {
        struct pt *p = &pts[pidx];

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

        // the angle ordering key; points are sorted by this rather than
        // by a stored slope field
        keys[pidx] = ((uint64_t)slope_sort_key(quadrant + dy/dx) << 32) | (uint32_t)~(uint32_t)pidx;
    }

    // Ensure that the black border is inside the white border.
    quad->reversed_border = dot < 0;
    if (!reversed_border && quad->reversed_border) {
        return 0;
    }
    if (!normal_border && !quad->reversed_border) {
        return 0;
    }

    // we now sort the points according to theta. This is a prepatory
    // step for segmenting them into four lines.
    if (1) {
        pt_key_sort(sz, scratch);
    }

    const struct lfps_soa *lfps = &scratch->lfps;
    // errs/yfilt/maxima_errs are free until quad_segment_maxima runs
    compute_lfps(sz, pts, keys, im, lfps, scratch->errs, scratch->yfilt, scratch->maxima_errs);

    int indices[4];
    if (1) {
        if (!quad_segment_maxima(td, sz, lfps, indices, scratch))
            goto finish;
    } else {
        if (!quad_segment_agg(sz, lfps, indices))
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
    int xmax = w - 2; // inclusive
    int start = 0;
    uint8_t v = row[0];
    int x = 1;

#ifdef __AVX2__
    // value-change positions 32 at a time
    for (; x + 32 <= xmax + 1; x += 32) {
        __m256i cur = _mm256_loadu_si256((const __m256i*)(row + x));
        __m256i prv = _mm256_loadu_si256((const __m256i*)(row + x - 1));
        uint32_t chg = ~(uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(cur, prv));
        while (chg) {
            int b = __builtin_ctz(chg);
            chg &= chg - 1;
            int cx = x + b;
            if (v != 127) {
                runs[n].start = start;
                runs[n].end = cx - 1;
                runs[n].v = v;
                n++;
            }
            start = cx;
            v = row[cx];
        }
    }
#endif

    for (; x <= xmax; x++) {
        if (row[x] != v) {
            if (v != 127) {
                runs[n].start = start;
                runs[n].end = x - 1;
                runs[n].v = v;
                n++;
            }
            start = x;
            v = row[x];
        }
    }
    if (v != 127) {
        runs[n].start = start;
        runs[n].end = xmax;
        runs[n].v = v;
        n++;
    }
    return n;
}

// number of runs rle_row would produce, without storing them
static int rle_row_count(const uint8_t *row, int w)
{
    int xmax = w - 2;
    int count = row[0] != 127;
    int x = 1;

#ifdef __AVX2__
    __m256i v127 = _mm256_set1_epi8(127);
    for (; x + 32 <= xmax + 1; x += 32) {
        __m256i cur = _mm256_loadu_si256((const __m256i*)(row + x));
        __m256i prv = _mm256_loadu_si256((const __m256i*)(row + x - 1));
        uint32_t chg = ~(uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(cur, prv));
        uint32_t n127 = ~(uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(cur, v127));
        count += __builtin_popcount(chg & n127);
    }
#endif

    for (; x <= xmax; x++)
        count += (row[x] != row[x-1]) && (row[x] != 127);
    return count;
}

// Shared per-frame run tables: row y's runs live at
// runs[row_off[y] .. row_off[y+1]). Built in two parallel passes (count,
// prefix-sum, fill) and consumed by the union-find, the stitch, and the
// gradient clustering.
struct rle_task
{
    int y0, y1;
    const uint8_t *buf;
    int w, s;
    uint32_t *row_off; // pass 1 writes counts at [y+1]; pass 2 reads offsets
    struct row_run *runs;
};

static void do_rle_count_task(void *p)
{
    struct rle_task *task = (struct rle_task*) p;
    for (int y = task->y0; y < task->y1; y++)
        task->row_off[y+1] = rle_row_count(&task->buf[y*task->s], task->w);
}

static void do_rle_fill_task(void *p)
{
    struct rle_task *task = (struct rle_task*) p;
    for (int y = task->y0; y < task->y1; y++)
        rle_row(&task->buf[y*task->s], task->w, &task->runs[task->row_off[y]]);
}

// builds the run tables into td->cached_runs_buf / td->cached_row_off
static void build_frame_runs(apriltag_detector_t *td, image_u8_t *threshim, int w, int h, int ts,
                             struct row_run **runs_out, uint32_t **row_off_out)
{
    if (td->cached_row_off_size < (int)((h+1)*sizeof(uint32_t))) {
        free(td->cached_row_off);
        td->cached_row_off = malloc((h+1)*sizeof(uint32_t));
        td->cached_row_off_size = (h+1)*sizeof(uint32_t);
    }
    uint32_t *row_off = td->cached_row_off;

    int chunksize = 1 + h / (APRILTAG_TASKS_PER_THREAD_TARGET * td->nthreads);
    struct rle_task *tasks = malloc(sizeof(struct rle_task)*(h / chunksize + 1));

    int ntasks = 0;
    for (int i = 0; i < h; i += chunksize) {
        tasks[ntasks].y0 = i;
        tasks[ntasks].y1 = imin(h, i + chunksize);
        tasks[ntasks].buf = threshim->buf;
        tasks[ntasks].w = w;
        tasks[ntasks].s = ts;
        tasks[ntasks].row_off = row_off;
        workerpool_add_task(td->wp, do_rle_count_task, &tasks[ntasks]);
        ntasks++;
    }
    workerpool_run(td->wp);

    row_off[0] = 0;
    for (int y = 0; y < h; y++)
        row_off[y+1] += row_off[y];

    int total = row_off[h];
    if (td->cached_runs_buf_size < (int)(total*sizeof(struct row_run))) {
        free(td->cached_runs_buf);
        int cap = total + total/2;
        td->cached_runs_buf = malloc(cap*sizeof(struct row_run));
        td->cached_runs_buf_size = cap*sizeof(struct row_run);
    }
    struct row_run *runs = (struct row_run*) td->cached_runs_buf;

    for (int i = 0; i < ntasks; i++) {
        tasks[i].runs = runs;
        workerpool_add_task(td->wp, do_rle_fill_task, &tasks[i]);
    }
    workerpool_run(td->wp);

    free(tasks);
    *runs_out = runs;
    *row_off_out = row_off;
}

// The union-find is indexed by GLOBAL RUN INDEX (row_off[y] + k), not by
// pixel id, so its arrays span ~#runs entries and stay cache resident.
// The run-less last column gets one virtual node per row, at index
// vcol_base + y (vcol_base == row_off[h], the total run count).

// Initialize each run as a union-find node owning the whole run.
static void unionfind_init_runs(unionfind_t *uf, uint32_t base, struct row_run *runs, int nruns)
{
    for (int i = 0; i < nruns; i++) {
        uf->parent[base + i] = base + i;
        uf->size[base + i] = runs[i].end - runs[i].start; // excludes the root
    }
}

// Connect two nodes. Equivalent to unionfind_connect, but walks the two
// root chains interleaved (they're independent latency chains) and applies
// full two-pass path compression; the resulting components and sizes are
// identical, only the tree shapes differ.
static inline void runfind_connect(unionfind_t *uf, uint32_t a, uint32_t b)
{
    uint32_t ra = a, rb = b;
    for (;;) {
        uint32_t pa = uf->parent[ra];
        uint32_t pb = uf->parent[rb];
        if (pa == ra && pb == rb)
            break;
        ra = pa;
        rb = pb;
    }

    if (ra == rb)
        return;

    // full path compression on both walks
    while (uf->parent[a] != ra) {
        uint32_t t = uf->parent[a];
        uf->parent[a] = ra;
        a = t;
    }
    while (uf->parent[b] != rb) {
        uint32_t t = uf->parent[b];
        uf->parent[b] = rb;
        b = t;
    }

    uint32_t asize = uf->size[ra] + 1;
    uint32_t bsize = uf->size[rb] + 1;
    if (asize > bsize) {
        uf->parent[rb] = ra;
        uf->size[ra] += bsize;
    } else {
        uf->parent[ra] = rb;
        uf->size[rb] += asize;
    }
}

// Union the runs of row y against the runs of row y-1: one union per pair
// of vertically (or, for white, diagonally) adjacent same-value runs. The
// per-pixel code's skip conditions already reduce its connects to exactly
// these pairs, so the resulting components and sizes are identical.
static void connect_runs_to_prev(unionfind_t *uf, const uint8_t *buf, int w, int s, int y,
                                 struct row_run *cur, int ncur, uint32_t cur_base,
                                 struct row_run *prev, int nprev, uint32_t prev_base,
                                 uint32_t vcol_base)
{
    int j = 0;
    for (int i = 0; i < ncur; i++) {
        int a0 = cur[i].start, a1 = cur[i].end;
        uint8_t v = cur[i].v;
        uint32_t head_a = cur_base + i;

        while (j < nprev && prev[j].end + 1 < a0)
            j++;

        for (int k = j; k < nprev && prev[k].start <= a1 + 1; k++) {
            if (prev[k].v != v)
                continue;
            int b0 = prev[k].start, b1 = prev[k].end;
            uint32_t head_b = prev_base + k;

            // direct vertical contact (only at x >= 1; the per-pixel
            // code never connects column 0 upward)
            int lo = imax(imax(a0, b0), 1);
            int hi = imin(a1, b1);
            if (lo <= hi) {
                runfind_connect(uf, head_a, head_b);
            } else if (v == 255) {
                // white is 8-connected: diagonal-only contact
                int xl = imax(imax(a0, b0 + 1), 1);
                if (xl <= imin(a1, b1 + 1)) {
                    runfind_connect(uf, head_a, head_b);
                } else {
                    int xr = imax(imax(a0, b0 - 1), 1);
                    if (xr <= imin(a1, b1 - 1)) {
                        runfind_connect(uf, head_a, head_b);
                    }
                }
            }
        }

        // The last column holds no runs, but a white run ending at w-2
        // reaches (w-1, y-1) diagonally. The per-pixel code only does
        // this connect when the pixel above the run end is not white
        // (otherwise its redundancy test skips it).
        if (v == 255 && a1 == w-2 && buf[(y-1)*s + (w-1)] == 255 && buf[(y-1)*s + (w-2)] != 255) {
            runfind_connect(uf, head_a, vcol_base + (y-1));
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
    struct row_run *runs = task->runs;
    uint32_t *row_off = task->row_off;
    uint32_t vcol_base = task->vcol_base;

    // no unions touch the prev row before this task runs, so initializing
    // its heads here is race-free (re-initialization before any union is
    // an identity for row 0 in the single-thread path)
    unionfind_init_runs(uf, row_off[task->y0 - 1], &runs[row_off[task->y0 - 1]],
                        row_off[task->y0] - row_off[task->y0 - 1]);

    for (int y = task->y0; y < task->y1; y++) {
        struct row_run *cur = &runs[row_off[y]];
        int ncur = row_off[y+1] - row_off[y];
        unionfind_init_runs(uf, row_off[y], cur, ncur);
        connect_runs_to_prev(uf, buf, w, s, y, cur, ncur, row_off[y],
                             &runs[row_off[y-1]], row_off[y] - row_off[y-1], row_off[y-1],
                             vcol_base);
    }
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

        struct pt_list **cluster;
        zarray_get_volatile(clusters, cidx, &cluster);

        // a cluster should contain only boundary points around the
        // tag. it cannot be bigger than the whole screen. (Reject
        // large connected blobs that will be prohibitively slow to
        // fit quads to.) A typical point along an edge is added two
        // times (because it has 2 unique neighbors). The maximum
        // perimeter is 2w+2h.
        if ((*cluster)->size >= td->qtp.min_cluster_pixels &&
            (*cluster)->size <= 2*(2*w+2*h)) {

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
        free(*cluster);
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

    int tx = 0;

#ifdef __AVX2__
    // 8 tiles (32 source columns) per iteration: reduce the four rows
    // pointwise, then each 4-byte lane horizontally
    const uint8_t *r0 = &im->buf[(ty*tilesz + 0)*s];
    const uint8_t *r1 = &im->buf[(ty*tilesz + 1)*s];
    const uint8_t *r2 = &im->buf[(ty*tilesz + 2)*s];
    const uint8_t *r3 = &im->buf[(ty*tilesz + 3)*s];

    const __m256i lane_lo = _mm256_setr_epi8(
        0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

    for (; tx + 8 <= tw; tx += 8) {
        __m256i a = _mm256_loadu_si256((const __m256i*)(r0 + 4*tx));
        __m256i b = _mm256_loadu_si256((const __m256i*)(r1 + 4*tx));
        __m256i c = _mm256_loadu_si256((const __m256i*)(r2 + 4*tx));
        __m256i d = _mm256_loadu_si256((const __m256i*)(r3 + 4*tx));

        __m256i mx = _mm256_max_epu8(_mm256_max_epu8(a, b), _mm256_max_epu8(c, d));
        __m256i mn = _mm256_min_epu8(_mm256_min_epu8(a, b), _mm256_min_epu8(c, d));

        // horizontal reduce within each 32-bit lane; shifted-in zeros are
        // neutral for max but must be masked to 0xff for min
        mx = _mm256_max_epu8(mx, _mm256_srli_epi32(mx, 8));
        mx = _mm256_max_epu8(mx, _mm256_srli_epi32(mx, 16));
        mn = _mm256_min_epu8(mn, _mm256_or_si256(_mm256_srli_epi32(mn, 8),
                                                 _mm256_set1_epi32(0xff000000)));
        mn = _mm256_min_epu8(mn, _mm256_or_si256(_mm256_srli_epi32(mn, 16),
                                                 _mm256_set1_epi32(0xffff0000)));

        __m256i pmx = _mm256_shuffle_epi8(mx, lane_lo);
        __m256i pmn = _mm256_shuffle_epi8(mn, lane_lo);

        uint32_t omax_lo = (uint32_t)_mm256_extract_epi32(pmx, 0);
        uint32_t omax_hi = (uint32_t)_mm256_extract_epi32(pmx, 4);
        uint32_t omin_lo = (uint32_t)_mm256_extract_epi32(pmn, 0);
        uint32_t omin_hi = (uint32_t)_mm256_extract_epi32(pmn, 4);
        memcpy(&task->im_max[ty*tw + tx], &omax_lo, 4);
        memcpy(&task->im_max[ty*tw + tx + 4], &omax_hi, 4);
        memcpy(&task->im_min[ty*tw + tx], &omin_lo, 4);
        memcpy(&task->im_min[ty*tw + tx + 4], &omin_hi, 4);
    }
#endif

    for (; tx < tw; tx++) {
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

    // columns [vec_lo, vec_hi) are written by the vector loop; the scalar
    // loop covers the rest (both edges and any partial-vector tail)
    int vec_lo = 0, vec_hi = 0;

#ifdef __AVX2__
    if (tw >= 34) {
        // vertical reduce of the (row-clamped) 3 rows, then horizontal
        // 3-tap min/max via unaligned loads
        const uint8_t *mxr0 = &im_max[(ty > 0 ? ty-1 : 0)*tw];
        const uint8_t *mxr1 = &im_max[ty*tw];
        const uint8_t *mxr2 = &im_max[(ty < th-1 ? ty+1 : th-1)*tw];
        const uint8_t *mnr0 = &im_min[(ty > 0 ? ty-1 : 0)*tw];
        const uint8_t *mnr1 = &im_min[ty*tw];
        const uint8_t *mnr2 = &im_min[(ty < th-1 ? ty+1 : th-1)*tw];

        vec_lo = 1;
        vec_hi = 1;
        for (int tx = 1; tx + 32 <= tw - 1; tx += 32) {
            __m256i vx0 = _mm256_max_epu8(_mm256_loadu_si256((const __m256i*)(mxr0 + tx - 1)),
                          _mm256_max_epu8(_mm256_loadu_si256((const __m256i*)(mxr1 + tx - 1)),
                                          _mm256_loadu_si256((const __m256i*)(mxr2 + tx - 1))));
            __m256i vx1 = _mm256_max_epu8(_mm256_loadu_si256((const __m256i*)(mxr0 + tx)),
                          _mm256_max_epu8(_mm256_loadu_si256((const __m256i*)(mxr1 + tx)),
                                          _mm256_loadu_si256((const __m256i*)(mxr2 + tx))));
            __m256i vx2 = _mm256_max_epu8(_mm256_loadu_si256((const __m256i*)(mxr0 + tx + 1)),
                          _mm256_max_epu8(_mm256_loadu_si256((const __m256i*)(mxr1 + tx + 1)),
                                          _mm256_loadu_si256((const __m256i*)(mxr2 + tx + 1))));
            _mm256_storeu_si256((__m256i*)(task->im_max_tmp + ty*tw + tx),
                                _mm256_max_epu8(vx0, _mm256_max_epu8(vx1, vx2)));

            __m256i vn0 = _mm256_min_epu8(_mm256_loadu_si256((const __m256i*)(mnr0 + tx - 1)),
                          _mm256_min_epu8(_mm256_loadu_si256((const __m256i*)(mnr1 + tx - 1)),
                                          _mm256_loadu_si256((const __m256i*)(mnr2 + tx - 1))));
            __m256i vn1 = _mm256_min_epu8(_mm256_loadu_si256((const __m256i*)(mnr0 + tx)),
                          _mm256_min_epu8(_mm256_loadu_si256((const __m256i*)(mnr1 + tx)),
                                          _mm256_loadu_si256((const __m256i*)(mnr2 + tx))));
            __m256i vn2 = _mm256_min_epu8(_mm256_loadu_si256((const __m256i*)(mnr0 + tx + 1)),
                          _mm256_min_epu8(_mm256_loadu_si256((const __m256i*)(mnr1 + tx + 1)),
                                          _mm256_loadu_si256((const __m256i*)(mnr2 + tx + 1))));
            _mm256_storeu_si256((__m256i*)(task->im_min_tmp + ty*tw + tx),
                                _mm256_min_epu8(vn0, _mm256_min_epu8(vn1, vn2)));

            vec_hi = tx + 32;
        }
    }
#endif

    for (int tx = 0; tx < tw; tx++) {
        if (tx == vec_lo && vec_hi > vec_lo) {
            tx = vec_hi - 1; // skip the vector-covered middle
            continue;
        }
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

// threshold one tile row (tilesz image rows, full tiles only)
static void threshold_tile_row(apriltag_detector_t *td, image_u8_t *im, image_u8_t *threshim,
                               uint8_t *im_max, uint8_t *im_min, int ty)
{
    const int tilesz = 4;
    int tw = im->width / tilesz;
    int s = im->stride;
    int min_white_black_diff = td->qtp.min_white_black_diff;

    int tx = 0;

#ifdef __AVX2__
    // 8 tiles (32 output columns) per iteration. Per tile: low-contrast
    // tiles write 127, others write 255 where v > thresh with
    // thresh = min + (max-min)/2. All integer, exactly like the scalar
    // code. Requires min_white_black_diff >= 1 so that the threshold path
    // only runs with max > min, keeping thresh+1 <= 255.
    if (min_white_black_diff >= 1) {
        const __m256i v127 = _mm256_set1_epi8(127);
        const __m256i v1 = _mm256_set1_epi8(1);
        const __m256i tm1 = _mm256_set1_epi8((char)(min_white_black_diff - 1 > 255 ? 255 : min_white_black_diff - 1));
        const __m256i shuf = _mm256_setr_epi8(
            0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
            4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7);

        for (; tx + 8 <= tw; tx += 8) {
            // 8 tile min/max bytes, expanded so each covers its 4 columns
            __m256i mn8 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*)&im_min[ty*tw + tx]));
            __m256i mx8 = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*)&im_max[ty*tw + tx]));
            __m256i mn = _mm256_shuffle_epi8(_mm256_permute4x64_epi64(mn8, 0x00), shuf);
            __m256i mx = _mm256_shuffle_epi8(_mm256_permute4x64_epi64(mx8, 0x00), shuf);

            __m256i diff = _mm256_sub_epi8(mx, mn); // max >= min, fits a byte
            // low contrast: diff < t  <=>  satsub(diff, t-1) == 0
            __m256i lc = _mm256_cmpeq_epi8(_mm256_subs_epu8(diff, tm1), _mm256_setzero_si256());

            // thresh = min + diff/2; on this path diff >= 1, so thresh < max
            // and thresh+1 <= 255
            __m256i half = _mm256_and_si256(_mm256_srli_epi16(diff, 1), _mm256_set1_epi8(0x7f));
            __m256i thresh1 = _mm256_add_epi8(_mm256_add_epi8(mn, half), v1);

            for (int dy = 0; dy < tilesz; dy++) {
                int y = ty*tilesz + dy;
                __m256i v = _mm256_loadu_si256((const __m256i*)&im->buf[y*s + tx*tilesz]);
                // v > thresh  <=>  v >= thresh+1  <=>  satsub(thresh+1, v) == 0
                __m256i gt = _mm256_cmpeq_epi8(_mm256_subs_epu8(thresh1, v), _mm256_setzero_si256());
                __m256i out = _mm256_blendv_epi8(gt, v127, lc); // gt mask is 0xff/0x00 = 255/0
                _mm256_storeu_si256((__m256i*)&threshim->buf[y*s + tx*tilesz], out);
            }
        }
    }
#endif

    for (; tx < tw; tx++) {
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
 
// Threshold a chunk of tile rows and run-length encode each finished line
// while it is still cache hot. Counts land in row_off[y+1]; the runs stay
// in a task-local buffer (rows in order) for one-memcpy compaction.
struct fused_threshold_task
{
    int ty0, ty1;
    apriltag_detector_t *td;
    image_u8_t *im;
    image_u8_t *threshim;
    uint8_t *im_max, *im_min;
    uint32_t *row_off;
    struct row_run *runs;
    int runs_n;
    int runs_cap;
};

static void do_fused_threshold_task(void *p)
{
    const int tilesz = 4;
    struct fused_threshold_task *task = (struct fused_threshold_task*) p;
    image_u8_t *im = task->im;
    image_u8_t *threshim = task->threshim;
    int w = im->width, s = im->stride;
    int tw = w / tilesz;
    uint8_t *im_max = task->im_max, *im_min = task->im_min;

    task->runs_cap = 4096 + w;
    task->runs = malloc(sizeof(struct row_run)*task->runs_cap);
    task->runs_n = 0;

    for (int ty = task->ty0; ty < task->ty1; ty++) {
        threshold_tile_row(task->td, im, threshim, im_max, im_min, ty);

        for (int dy = 0; dy < tilesz; dy++) {
            int y = ty*tilesz + dy;

            // right-edge pixels not covered by full tiles; like the
            // historical fixup, these threshold without the low-contrast
            // 127 marking, using the clamped last tile
            for (int x = tw*tilesz; x < w; x++) {
                int max = im_max[ty*tw + (tw-1)];
                int min = im_min[ty*tw + (tw-1)];
                int thresh = min + (max - min) / 2;
                threshim->buf[y*s+x] = (im->buf[y*s+x] > thresh) ? 255 : 0;
            }

            if (task->runs_cap - task->runs_n < w) {
                task->runs_cap *= 2;
                task->runs = realloc(task->runs, sizeof(struct row_run)*task->runs_cap);
            }
            int n = rle_row(&threshim->buf[y*s], w, &task->runs[task->runs_n]);
            task->runs_n += n;
            task->row_off[y+1] = n;
        }
    }
}

image_u8_t *threshold(apriltag_detector_t *td, image_u8_t *im,
                      struct row_run **runs_out, uint32_t **row_off_out)
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

    // row_off (counts during the fused pass, offsets after the prefix sum)
    if (td->cached_row_off_size < (int)((h+1)*sizeof(uint32_t))) {
        free(td->cached_row_off);
        td->cached_row_off = malloc((h+1)*sizeof(uint32_t));
        td->cached_row_off_size = (h+1)*sizeof(uint32_t);
    }
    uint32_t *row_off = td->cached_row_off;

    int tchunk = 1 + th / (APRILTAG_TASKS_PER_THREAD_TARGET * td->nthreads);
    struct fused_threshold_task *ftasks = malloc(sizeof(struct fused_threshold_task)*(th / tchunk + 1));
    int nft = 0;
    for (int i = 0; i < th; i += tchunk) {
        ftasks[nft].ty0 = i;
        ftasks[nft].ty1 = imin(th, i + tchunk);
        ftasks[nft].td = td;
        ftasks[nft].im = im;
        ftasks[nft].threshim = threshim;
        ftasks[nft].im_max = im_max;
        ftasks[nft].im_min = im_min;
        ftasks[nft].row_off = row_off;
        workerpool_add_task(td->wp, do_fused_threshold_task, &ftasks[nft]);
        nft++;
    }
    workerpool_run(td->wp);

    // bottom partial rows (fewer than tilesz of them): full-width
    // threshold using the clamped last tile row, then rle
    struct row_run *tail_runs = NULL;
    int tail_n = 0;
    if (th*tilesz < h) {
        tail_runs = malloc(sizeof(struct row_run)*(h - th*tilesz)*(w+1));
        for (int y = th*tilesz; y < h; y++) {
            int ty = th - 1;
            for (int x = 0; x < w; x++) {
                int tx = x / tilesz;
                if (tx >= tw)
                    tx = tw - 1;

                int max = im_max[ty*tw + tx];
                int min = im_min[ty*tw + tx];
                int thresh = min + (max - min) / 2;

                uint8_t v = im->buf[y*s+x];
                threshim->buf[y*s+x] = (v > thresh) ? 255 : 0;
            }
            int n = rle_row(&threshim->buf[y*s], w, &tail_runs[tail_n]);
            tail_n += n;
            row_off[y+1] = n;
        }
    }

    if (!td->qtp.deglitch) {
        // prefix the counts and compact the per-task buffers; each task's
        // rows are consecutive, so one copy per task suffices
        row_off[0] = 0;
        for (int y = 0; y < h; y++)
            row_off[y+1] += row_off[y];

        int total = row_off[h];
        if (td->cached_runs_buf_size < (int)(total*sizeof(struct row_run))) {
            free(td->cached_runs_buf);
            int cap = total + total/2;
            td->cached_runs_buf = malloc(cap*sizeof(struct row_run));
            td->cached_runs_buf_size = cap*sizeof(struct row_run);
        }
        struct row_run *runs = (struct row_run*) td->cached_runs_buf;

        for (int i = 0; i < nft; i++) {
            memcpy(&runs[row_off[ftasks[i].ty0 * tilesz]], ftasks[i].runs,
                   ftasks[i].runs_n * sizeof(struct row_run));
            free(ftasks[i].runs);
        }
        if (tail_n)
            memcpy(&runs[row_off[th*tilesz]], tail_runs, tail_n*sizeof(struct row_run));

        *runs_out = runs;
        *row_off_out = row_off;
    } else {
        for (int i = 0; i < nft; i++)
            free(ftasks[i].runs);
    }
    free(tail_runs);
    free(ftasks);

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

        // deglitch rewrote threshim after the fused rle; rebuild the runs
        build_frame_runs(td, threshim, w, h, s, runs_out, row_off_out);
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

unionfind_t* connected_components(apriltag_detector_t *td, image_u8_t* threshim, int w, int h, int ts,
                                  struct row_run *runs, uint32_t *row_off) {
    // nodes: one per run plus one virtual node per row for the run-less
    // last column (reachable as a diagonal neighbor of a white run)
    uint32_t vcol_base = row_off[h];
    uint32_t maxid = vcol_base + h;
    if (td->cached_uf) {
        if (td->cached_uf->maxid < maxid) {
            unionfind_resize(td->cached_uf, maxid);
        }
    } else {
        td->cached_uf = unionfind_create(maxid);
    }
    unionfind_t *uf = td->cached_uf;

    // No unionfind_reset between frames: every run node is (re)initialized
    // by the pass below, and the virtual last-column nodes here.
    for (int y = 0; y < h; y++) {
        uf->parent[vcol_base + y] = vcol_base + y;
        uf->size[vcol_base + y] = 0;
    }

    if (td->nthreads <= 1) {
        struct unionfind_task task;
        task.y0 = 1;
        task.y1 = h;
        task.h = h;
        task.w = w;
        task.s = ts;
        task.uf = uf;
        task.im = threshim;
        task.runs = runs;
        task.row_off = row_off;
        task.vcol_base = vcol_base;
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
            tasks[ntasks].runs = runs;
            tasks[ntasks].row_off = row_off;
            tasks[ntasks].vcol_base = vcol_base;

            ntasks++;
        }

        // The chunking can leave the final row uncovered (the historical
        // code left its pixels lazily initialized and never connected, so
        // their components stayed singletons). Initialize those run nodes
        // as size-0 singletons to match that gate behavior.
        int covered_end = ntasks > 0 ? tasks[ntasks-1].y1 : 1;
        for (uint32_t ri = row_off[covered_end]; ri < row_off[h]; ri++) {
            uf->parent[ri] = ri;
            uf->size[ri] = 0;
        }

        for (int i = 0; i < ntasks; i++) {
            workerpool_add_task(td->wp, do_unionfind_task2, &tasks[i]);
        }

        workerpool_run(td->wp);

        // stitch together the chunks: connect each gap row (whose heads the
        // task below initialized) to the row above it
        for (int i = 1; i < ntasks; i++) {
            int gy = tasks[i].y0 - 1;
            connect_runs_to_prev(uf, threshim->buf, w, ts, gy,
                                 &runs[row_off[gy]], row_off[gy+1] - row_off[gy], row_off[gy],
                                 &runs[row_off[gy-1]], row_off[gy] - row_off[gy-1], row_off[gy-1],
                                 vcol_base);
        }

        free(tasks);
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

// resolve the cluster entry for a representative pair (no point appended)
static inline struct uint64_zarray_entry *gc_pair_entry(struct gc_ctx *ctx, uint64_t rep0, uint64_t rep1)
{
    uint64_t clusterid;
    if (rep0 < rep1)
        clusterid = (rep1 << 32) + rep0;
    else
        clusterid = (rep0 << 32) + rep1;

    if (ctx->last_entry && ctx->last_entry->id == clusterid)
        return ctx->last_entry;

    uint32_t bucket = u64hash_2(clusterid) & ctx->bucket_mask;
    struct uint64_zarray_entry *entry = ctx->clustermap[bucket];
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
    return entry;
}

// lazily computed representative + size gate for one row run
struct run_rep
{
    uint32_t rep;
    int8_t state; // 0 = unknown, 1 = usable, 2 = component too small
};

// Read-only find: the union-find is complete by the time clustering runs,
// so skip path halving -- its writes to the small shared arrays would just
// ping-pong cache lines between the cluster tasks.
static inline uint32_t unionfind_representative_ro(const unionfind_t *uf, uint32_t id)
{
    while (uf->parent[id] != id)
        id = uf->parent[id];
    return id;
}

static inline int run_usable(unionfind_t *uf, uint32_t base, struct run_rep *cache, int idx,
                             int min_cluster_pixels, uint32_t *rep_out)
{
    if (cache[idx].state == 0) {
        uint32_t rep = unionfind_representative_ro(uf, base + idx);
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
zarray_t* do_gradient_clusters(image_u8_t* threshim, int ts, int y0, int y1, int w, int nclustermap, int min_cluster_pixels, unionfind_t* uf,
                               struct row_run *frame_runs, uint32_t *row_off, uint32_t vcol_base, struct cluster_hash **ch_pool_out, zarray_t* clusters) {
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

    struct run_rep *cache_a = malloc(sizeof(struct run_rep)*(w+1));
    struct run_rep *cache_b = malloc(sizeof(struct run_rep)*(w+1));

    struct row_run *runs_a = &frame_runs[row_off[y0]]; // row y
    int na = row_off[y0+1] - row_off[y0];
    for (int i = 0; i < na; i++)
        cache_a[i].state = 0;

    for (int y = y0; y < y1; y++) {
        struct row_run *runs_b = &frame_runs[row_off[y+1]]; // row y+1
        int nb = row_off[y+2] - row_off[y+1];
        uint32_t base_a = row_off[y], base_b = row_off[y+1];
        for (int i = 0; i < nb; i++)
            cache_b[i].state = 0;

        // did the previous pixel add a point via its (1,1) neighbor?
        bool connected_last = false;

        // next black/white run of row y+1 reaching the sweep position
        int pv_black = 0, pv_white = 0;

        for (int ia = 0; ia < na; ia++) {
            int a0 = runs_a[ia].start, a1 = runs_a[ia].end;
            uint8_t v0 = runs_a[ia].v;
            uint8_t vopp = 255 - v0;
            int vdiff = (int)vopp - (int)v0; // v1 - v0, +-255

            // a 127 gap before this run resets the (1,1) memory
            if (ia == 0 || runs_a[ia-1].end + 1 < a0)
                connected_last = false;
            bool cl_entry = connected_last;

            uint32_t rep0 = 0;
            int rep0_state = 0; // lazy, as in the per-pixel code

            int ax0 = a0 < 1 ? 1 : a0;
            bool done10 = false;
            bool fired11_at_a1 = false;

// resolve this run's component, once
#define RESOLVE_A() \
            if (rep0_state == 0) \
                rep0_state = run_usable(uf, base_a, cache_a, ia, min_cluster_pixels, &rep0) ? 1 : 2;

// the (1,0) emission at x == a1; runs before any other emission there
#define EMIT_10() \
            do { \
                done10 = true; \
                uint8_t v1 = buf[y*ts + a1 + 1]; \
                if (v0 + v1 == 255) { \
                    RESOLVE_A(); \
                    if (rep0_state == 1) { \
                        uint32_t rep1; \
                        int ok; \
                        if (ia + 1 < na && runs_a[ia+1].start == a1 + 1) { \
                            ok = run_usable(uf, base_a, cache_a, ia+1, min_cluster_pixels, &rep1); \
                        } else { \
                            rep1 = unionfind_representative_ro(uf, vcol_base + y); \
                            ok = (int)(uf->size[rep1] + 1) >= min_cluster_pixels; \
                        } \
                        if (ok) \
                            gc_add_point(&ctx, rep0, rep1, 2*a1 + 1, 2*y, vdiff, 0); \
                    } \
                } \
            } while (0)

            // sweep the opposite-value runs of row y+1 whose extended
            // window [b0-1, b1+1] reaches this run's span [ax0, a1]
            int *pv = (vopp == 255) ? &pv_white : &pv_black;
            while (*pv < nb && (runs_b[*pv].v != vopp || (int)runs_b[*pv].end < ax0 - 1))
                (*pv)++;

            for (int k = *pv; k < nb && (int)runs_b[k].start <= a1 + 1; k++) {
                if (runs_b[k].v != vopp)
                    continue;
                int b0 = runs_b[k].start, b1 = runs_b[k].end;

                RESOLVE_A();
                if (rep0_state == 2)
                    break;

                uint32_t rep1;
                if (!run_usable(uf, base_b, cache_b, k, min_cluster_pixels, &rep1))
                    continue;

                struct uint64_zarray_entry *entry = gc_pair_entry(&ctx, rep0, rep1);

                // p1 = b0-1: (1,1) only (a possible (-1,1) at this x
                // belongs to the previous opposite run and was emitted in
                // its iteration, in the correct order)
                int p1 = b0 - 1;
                if (p1 >= ax0 && p1 <= a1) {
                    if (p1 == a1 && !done10)
                        EMIT_10();
                    gc_entry_append(&ctx, entry, 2*p1 + 1, 2*y + 1, vdiff, vdiff);
                    if (p1 == a1)
                        fired11_at_a1 = true;
                }

                // interior [xs, xe]: (0,1) + (1,1) pairs
                int xs = imax(b0, ax0);
                int xe = imin(b1 - 1, a1);
                if (xs <= xe) {
                    // first position: may carry an entry (-1,1) when this
                    // run starts inside the opposite run
                    if (xs == a1 && !done10)
                        EMIT_10();
                    gc_entry_append(&ctx, entry, 2*xs, 2*y + 1, 0, vdiff);
                    if (xs == ax0 && b0 < ax0 && !cl_entry)
                        gc_entry_append(&ctx, entry, 2*xs - 1, 2*y + 1, -vdiff, vdiff);
                    gc_entry_append(&ctx, entry, 2*xs + 1, 2*y + 1, vdiff, vdiff);
                    if (xs == a1)
                        fired11_at_a1 = true;

                    // batch strictly-interior pairs (never at a1, never
                    // at ax0) as packed 8-byte stores; only the x field
                    // (low half-word) advances, by 2 per pixel, and
                    // 2*x+1 < 2^16 so it never carries into y
                    int bend = imin(xe, a1 - 1);
                    if (bend > xs) {
                        uint64_t q0 = (uint64_t)(uint16_t)(2*(xs+1)) |
                                      ((uint64_t)(uint16_t)(2*y + 1) << 16) |
                                      ((uint64_t)(uint16_t)(int16_t)vdiff << 48);
                        uint64_t q1 = (uint64_t)(uint16_t)(2*(xs+1) + 1) |
                                      ((uint64_t)(uint16_t)(2*y + 1) << 16) |
                                      ((uint64_t)(uint16_t)(int16_t)vdiff << 32) |
                                      ((uint64_t)(uint16_t)(int16_t)vdiff << 48);

                        int remaining = bend - xs;
                        entry->npts += 2*remaining;
                        while (remaining > 0) {
                            struct gc_chunk *t = entry->tail;
                            int space = (GC_CHUNK_PTS - t->count) / 2;
                            if (space == 0) {
                                struct gc_chunk *c = gc_chunk_alloc(&ctx.chunk_pool);
                                t->next = c;
                                entry->tail = c;
                                t = c;
                                space = GC_CHUNK_PTS / 2;
                            }
                            int batch = remaining < space ? remaining : space;
                            uint64_t *dst = (uint64_t*)&t->pts[t->count];
                            t->count += 2*batch;
                            remaining -= batch;
                            int b = 0;
#ifdef __AVX2__
                            {
                                // two pairs (4 points) per store pair
                                __m256i q = _mm256_setr_epi64x(q0, q1, q0 + 2, q1 + 2);
                                const __m256i step = _mm256_set1_epi64x(4);
                                for (; b + 2 <= batch; b += 2) {
                                    _mm256_storeu_si256((__m256i*)&dst[2*b], q);
                                    q = _mm256_add_epi64(q, step);
                                }
                                q0 += 2*b;
                                q1 += 2*b;
                            }
#endif
                            for (; b < batch; b++) {
                                dst[2*b] = q0;
                                dst[2*b + 1] = q1;
                                q0 += 2;
                                q1 += 2;
                            }
                        }
                    }

                    // final interior position when it sits at a1: its
                    // (1,0) fires first
                    if (xe == a1 && xe > xs) {
                        if (!done10)
                            EMIT_10();
                        gc_entry_append(&ctx, entry, 2*xe, 2*y + 1, 0, vdiff);
                        gc_entry_append(&ctx, entry, 2*xe + 1, 2*y + 1, vdiff, vdiff);
                        fired11_at_a1 = true;
                    }
                }

                // p2 = b1: (0,1) only; the (-1,1) here is suppressed by
                // the previous pixel's (1,1) except at the run entry
                if (b1 >= ax0 && b1 <= a1 && b1 > xe) {
                    if (b1 == a1 && !done10)
                        EMIT_10();
                    gc_entry_append(&ctx, entry, 2*b1, 2*y + 1, 0, vdiff);
                    if (b1 == ax0 && b1 > b0 && !cl_entry)
                        gc_entry_append(&ctx, entry, 2*b1 - 1, 2*y + 1, -vdiff, vdiff);
                }

                // p3 = b1+1: (-1,1) only (no (1,1) fired at b1)
                int p3 = b1 + 1;
                if (p3 >= ax0 && p3 <= a1 && !(p3 == ax0 && cl_entry)) {
                    if (p3 == a1 && !done10)
                        EMIT_10();
                    gc_entry_append(&ctx, entry, 2*p3 - 1, 2*y + 1, -vdiff, vdiff);
                }
            }

            if (rep0_state != 2) {
                if (!done10)
                    EMIT_10();

                // x+1 == w-1 at the run end: the last column holds no
                // runs, so the (1,1) there resolves the virtual node
                if (a1 == w - 2) {
                    uint8_t v1 = buf[(y+1)*ts + a1 + 1];
                    if (v0 + v1 == 255) {
                        RESOLVE_A();
                        if (rep0_state == 1) {
                            uint32_t rep1 = unionfind_representative_ro(uf, vcol_base + (y+1));
                            if ((int)(uf->size[rep1] + 1) >= min_cluster_pixels) {
                                gc_add_point(&ctx, rep0, rep1, 2*a1 + 1, 2*y + 1, vdiff, vdiff);
                                fired11_at_a1 = true;
                            }
                        }
                    }
                }
            }

            connected_last = (rep0_state != 2) && fired11_at_a1;

#undef EMIT_10
#undef RESOLVE_A
        }

        // row y+1's runs and resolved representatives become row y's
        runs_a = runs_b;
        struct run_rep *ct = cache_a; cache_a = cache_b; cache_b = ct;
        na = nb;
    }

    free(cache_a);
    free(cache_b);

    struct uint64_zarray_entry **clustermap = ctx.clustermap;
    struct uint64_zarray_entry **mem_pools = ctx.mem_pools;
    int mem_pool_idx = ctx.mem_pool_idx;

    // one allocation backs every cluster_hash this task emits
    int n_entries = ctx.mem_pool_idx*ctx.mem_chunk_size + ctx.mem_pool_loc;
    struct cluster_hash *ch_pool = malloc(sizeof(struct cluster_hash)*(n_entries > 0 ? n_entries : 1));
    *ch_pool_out = ch_pool;
    int ch_n = 0;

    for (int i = 0; i < nclustermap; i++) {
        int start = zarray_size(clusters);
        for (struct uint64_zarray_entry *entry = clustermap[i]; entry; entry = entry->next) {
            struct cluster_hash* cluster_hash = &ch_pool[ch_n++];
            cluster_hash->hash = i; // == u64hash_2(entry->id) & bucket_mask
            cluster_hash->id = entry->id;

            // materialize the chunk list into one exact-size allocation
            struct pt_list *cl = malloc(sizeof(struct pt_list) + entry->npts*sizeof(struct pt));
            cl->size = entry->npts;
            struct pt *dst = cl->pts;
            for (struct gc_chunk *c = entry->head; c; c = c->next) {
                memcpy(dst, c->pts, c->count*sizeof(struct pt));
                dst += c->count;
            }
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

    do_gradient_clusters(task->im, task->s, task->y0, task->y1, task->w, task->nclustermap, task->min_cluster_pixels, task->uf, task->runs, task->row_off, task->vcol_base, &task->ch_pool, task->clusters);
}

// concatenate recorded fragment groups into their output slots
struct cluster_concat_task
{
    int g0, g1;
    struct pt_list **frags;
    uint32_t *gstart;
    struct pt_list **out;
};

static void do_cluster_concat_task(void *p)
{
    struct cluster_concat_task *task = (struct cluster_concat_task*) p;
    for (int g = task->g0; g < task->g1; g++) {
        uint32_t f0 = task->gstart[g], f1 = task->gstart[g+1];
        if (f1 - f0 == 1) {
            task->out[g] = task->frags[f0];
            continue;
        }
        int tot = 0;
        for (uint32_t f = f0; f < f1; f++)
            tot += task->frags[f]->size;
        struct pt_list *cl = malloc(sizeof(struct pt_list) + tot*sizeof(struct pt));
        cl->size = tot;
        struct pt *dst = cl->pts;
        for (uint32_t f = f0; f < f1; f++) {
            memcpy(dst, task->frags[f]->pts, task->frags[f]->size*sizeof(struct pt));
            dst += task->frags[f]->size;
            free(task->frags[f]);
        }
        task->out[g] = cl;
    }
}

zarray_t* gradient_clusters(apriltag_detector_t *td, image_u8_t* threshim, int w, int h, int ts, unionfind_t* uf,
                            struct row_run *runs, uint32_t *row_off) {
    uint32_t vcol_base = row_off[h];
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
        tasks[ntasks].runs = runs;
        tasks[ntasks].row_off = row_off;
        tasks[ntasks].vcol_base = vcol_base;

        workerpool_add_task(td->wp, do_cluster_task, &tasks[ntasks]);
        ntasks++;
    }

    workerpool_run(td->wp);

    // Single-pass k-way merge of the per-task lists (each sorted by
    // (hash, id)) with a small index heap. Equal-key clusters from
    // different tasks concatenate their points in task order -- the same
    // order the old pairwise merge tree produced.
    int total = 0;
    for (int i = 0; i < ntasks; i++)
        total += zarray_size(tasks[i].clusters);

    clusters = zarray_create(sizeof(struct pt_list*));
    zarray_ensure_capacity(clusters, total);

    int *heap = malloc(sizeof(int)*(ntasks > 0 ? ntasks : 1));
    int *pos = calloc(ntasks > 0 ? ntasks : 1, sizeof(int));
    int hn = 0;

    // key of task t's current head; (hash, id, t) ascending
#define HEAD(t) ((struct cluster_hash**)tasks[t].clusters->data)[pos[t]]
#define KEY_LT(ta, tb) (HEAD(ta)->hash != HEAD(tb)->hash ? HEAD(ta)->hash < HEAD(tb)->hash : \
                        (HEAD(ta)->id != HEAD(tb)->id ? HEAD(ta)->id < HEAD(tb)->id : (ta) < (tb)))

    for (int t = 0; t < ntasks; t++) {
        if (zarray_size(tasks[t].clusters) == 0)
            continue;
        // sift up
        int i = hn++;
        heap[i] = t;
        while (i > 0) {
            int parent = (i-1)/2;
            if (!KEY_LT(heap[i], heap[parent]))
                break;
            int tmp = heap[i]; heap[i] = heap[parent]; heap[parent] = tmp;
            i = parent;
        }
    }

    uint32_t last_hash = 0;
    uint64_t last_id = 0;

    // The heap walk only records fragments (in pop order) and group
    // boundaries; the split-cluster concatenations happen in parallel
    // afterwards, into order-preserving output slots.
    struct pt_list **frags = malloc(sizeof(struct pt_list*)*(total > 0 ? total : 1));
    uint32_t *gstart = malloc(sizeof(uint32_t)*(total + 1));
    int nfrags = 0, ngroups = 0;

    while (hn > 0) {
        int t = heap[0];
        struct cluster_hash *ch = HEAD(t);

        if (!(nfrags > 0 && ch->hash == last_hash && ch->id == last_id))
            gstart[ngroups++] = nfrags;
        frags[nfrags++] = ch->data;
        last_hash = ch->hash;
        last_id = ch->id;

        pos[t]++;
        if (pos[t] == zarray_size(tasks[t].clusters)) {
            heap[0] = heap[--hn];
        }
        // sift down
        int i = 0;
        while (1) {
            int l = 2*i + 1, r = 2*i + 2, m = i;
            if (l < hn && KEY_LT(heap[l], heap[m]))
                m = l;
            if (r < hn && KEY_LT(heap[r], heap[m]))
                m = r;
            if (m == i)
                break;
            int tmp = heap[i]; heap[i] = heap[m]; heap[m] = tmp;
            i = m;
        }
    }
    gstart[ngroups] = nfrags;

#undef KEY_LT
#undef HEAD

    clusters->size = ngroups; // capacity reserved above; slots written below

    int gchunk = 1 + ngroups / (4*td->nthreads);
    struct cluster_concat_task *ctasks = malloc(sizeof(struct cluster_concat_task) * (ngroups / gchunk + 1));
    int ncct = 0;
    for (int g = 0; g < ngroups; g += gchunk) {
        ctasks[ncct].g0 = g;
        ctasks[ncct].g1 = imin(ngroups, g + gchunk);
        ctasks[ncct].frags = frags;
        ctasks[ncct].gstart = gstart;
        ctasks[ncct].out = (struct pt_list**)clusters->data;
        ncct++;
    }
    for (int i = 0; i < ncct; i++)
        workerpool_add_task(td->wp, do_cluster_concat_task, &ctasks[i]);
    workerpool_run(td->wp);

    free(ctasks);
    free(frags);
    free(gstart);
    free(heap);
    free(pos);
    for (int i = 0; i < ntasks; i++) {
        zarray_destroy(tasks[i].clusters);
        free(tasks[i].ch_pool);
    }
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

    // thresholding also produces the shared per-frame run tables, consumed
    // by the union-find pass and the gradient clustering
    struct row_run *frame_runs;
    uint32_t *row_off;
    image_u8_t *threshim = threshold(td, im, &frame_runs, &row_off);
    int ts = threshim->stride;

    if (td->debug)
        image_u8_write_pnm(threshim, "debug_threshold.pnm");


    ////////////////////////////////////////////////////////
    // step 2. find connected components.

    unionfind_t* uf = connected_components(td, threshim, w, h, ts, frame_runs, row_off);

    // make segmentation image.
    if (td->debug) {
        image_u8x3_t *d = image_u8x3_create(w, h);

        uint32_t *colors = (uint32_t*) calloc(row_off[h] + h, sizeof(*colors));

        // the union-find is indexed by run: paint each run with its
        // component color
        for (int y = 0; y < h; y++) {
            for (uint32_t ri = row_off[y]; ri < row_off[y+1]; ri++) {
                uint32_t v = unionfind_get_representative(uf, ri);

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

                for (int x = frame_runs[ri].start; x <= frame_runs[ri].end; x++) {
                    d->buf[y*d->stride + 3*x + 0] = r;
                    d->buf[y*d->stride + 3*x + 1] = g;
                    d->buf[y*d->stride + 3*x + 2] = b;
                }
            }
        }

        free(colors);

        image_u8x3_write_pnm(d, "debug_segmentation.pnm");
        image_u8x3_destroy(d);
    }


    timeprofile_stamp(td->tp, "unionfind");

    zarray_t* clusters = gradient_clusters(td, threshim, w, h, ts, uf, frame_runs, row_off);

    if (td->debug) {
        image_u8x3_t *d = image_u8x3_create(w, h);

        for (int i = 0; i < zarray_size(clusters); i++) {
            struct pt_list *cluster;
            zarray_get(clusters, i, &cluster);

            uint32_t r, g, b;

            if (1) {
                const int bias = 50;
                r = bias + (random() % (200-bias));
                g = bias + (random() % (200-bias));
                b = bias + (random() % (200-bias));
            }

            for (int j = 0; j < cluster->size; j++) {
                struct pt *p = &cluster->pts[j];

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
