/**
 * @file image_u8_parallel.c
 * @author MqCreaple (gmq14159@gmail.com)
 * @brief Parallelized processing of various image_u8 related functions.
 * @version 0.1
 * @date 2025-08-07
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include "common/image_u8_parallel.h"
#include "common/workerpool.h"
#include "common/math_util.h"

#ifdef __AVX2__
#include <immintrin.h>
#endif

static void convolve(const uint8_t *x, uint8_t *y, int sz, const uint8_t *k, int ksz)
{
    assert((ksz&1)==1);

    for (int i = 0; i < ksz/2 && i < sz; i++)
        y[i] = x[i];

    for (int i = 0; i < sz - ksz + 1; i++) {
        uint32_t acc = 0;

        for (int j = 0; j < ksz; j++)
            acc += k[j]*x[i+j];

        y[ksz/2 + i] = acc >> 8;
    }

    for (int i = sz - ksz/2; i < sz; i++)
        y[i] = x[i];
}

struct image_u8_convolve_2D_task {
    image_u8_t *im;
    const uint8_t *k;
    int ksz;
    int idx_st;
    int idx_ed;
};

static void _image_u8_convolve_2D_thread_1(void *p) {
    struct image_u8_convolve_2D_task *params = (struct image_u8_convolve_2D_task*) p;
    image_u8_t *im = params->im;
    const uint8_t *k = params->k;
    int ksz = params->ksz;
    int y_st = params->idx_st;
    int y_ed = params->idx_ed;

    assert((ksz & 1) == 1); // ksz must be odd.

    uint8_t *x = malloc(sizeof(uint8_t)*im->stride);
    for (int y = y_st; y < y_ed; y++) {
        memcpy(x, &im->buf[y*im->stride], im->stride);
        convolve(x, &im->buf[y*im->stride], im->width, k, ksz);
    }
    free(x);
}

static void _image_u8_convolve_2D_thread_2(void *p) {
    struct image_u8_convolve_2D_task *params = (struct image_u8_convolve_2D_task*) p;
    image_u8_t *im = params->im;
    const uint8_t *k = params->k;
    int ksz = params->ksz;
    int x_st = params->idx_st;
    int x_ed = params->idx_ed;

    uint8_t *xb = malloc(sizeof(uint8_t)*im->height);
    uint8_t *yb = malloc(sizeof(uint8_t)*im->height);
    for (int x = x_st; x < x_ed; x++) {

        for (int y = 0; y < im->height; y++)
            xb[y] = im->buf[y*im->stride + x];

        convolve(xb, yb, im->height, k, ksz);

        for (int y = 0; y < im->height; y++)
            im->buf[y*im->stride + x] = yb[y];
    }
    free(xb);
    free(yb);
}

void image_u8_convolve_2D_parallel(workerpool_t *wp, image_u8_t *im, const uint8_t *k, int ksz) {
    if(im->width * im->height < 65536) {
        // for small images, directly use single threaded convolution
        image_u8_convolve_2D(im, k, ksz);
        return;
    }
    int nthreads = workerpool_get_nthreads(wp);

    struct image_u8_convolve_2D_task *params = malloc(sizeof(struct image_u8_convolve_2D_task) * nthreads);
    int y_inc = im->height / nthreads;
    int y_remainder = im->height % nthreads;
    int last_y = 0;
    for(int idx = 0; idx < nthreads; idx++) {
        params[idx].im = im;
        params[idx].k = k;
        params[idx].ksz = ksz;
        params[idx].idx_st = last_y;
        last_y += y_inc;
        if(idx < y_remainder) {
            last_y += 1;     // distribute the remainders across the n threads
        }
        params[idx].idx_ed = last_y;
        workerpool_add_task(wp, _image_u8_convolve_2D_thread_1, &params[idx]);
    }
    workerpool_run(wp);

    int x_inc = im->width / nthreads;
    int x_remainder = im->width % nthreads;
    int last_x = 0;
    for(int idx = 0; idx < nthreads; idx++) {
        params[idx].im = im;
        params[idx].k = k;
        params[idx].ksz = ksz;
        params[idx].idx_st = last_x;
        last_x += x_inc;
        if(idx < x_remainder) {
            last_x += 1;     // distribute the remainders across the n threads
        }
        params[idx].idx_ed = last_x;
        workerpool_add_task(wp, _image_u8_convolve_2D_thread_2, &params[idx]);
    }
    workerpool_run(wp);

    free(params);
}

struct decimate2_box_task {
    const image_u8_t *in;
    image_u8_t *out;
    int sy0, sy1;
};

// one output row: out[sx] = round-ish((a+b+c+d)/4) over the 2x2 input block.
// avg_epu8 rounds each pairwise average up, so the result can differ from the
// exact rounded mean by at most 1 gray level — irrelevant downstream.
static void decimate2_box_rows(const image_u8_t *in, image_u8_t *out, int sy0, int sy1)
{
    int swidth = out->width;
    for (int sy = sy0; sy < sy1; sy++) {
        const uint8_t *r0 = &in->buf[(2*sy + 0)*in->stride];
        const uint8_t *r1 = &in->buf[(2*sy + 1)*in->stride];
        uint8_t *o = &out->buf[sy*out->stride];
        int sx = 0;
#ifdef __AVX2__
        const __m256i ones = _mm256_set1_epi8(1);
        const __m256i round1 = _mm256_set1_epi16(1);
        for (; sx + 32 <= swidth; sx += 32) {
            __m256i a0 = _mm256_loadu_si256((const __m256i *)(r0 + 2*sx));
            __m256i a1 = _mm256_loadu_si256((const __m256i *)(r0 + 2*sx + 32));
            __m256i b0 = _mm256_loadu_si256((const __m256i *)(r1 + 2*sx));
            __m256i b1 = _mm256_loadu_si256((const __m256i *)(r1 + 2*sx + 32));
            __m256i v0 = _mm256_avg_epu8(a0, b0);
            __m256i v1 = _mm256_avg_epu8(a1, b1);
            // horizontal pair sums in 16-bit lanes, then round and halve
            __m256i s0 = _mm256_srli_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(v0, ones), round1), 1);
            __m256i s1 = _mm256_srli_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(v1, ones), round1), 1);
            __m256i packed = _mm256_packus_epi16(s0, s1);
            packed = _mm256_permute4x64_epi64(packed, 0xD8);
            _mm256_storeu_si256((__m256i *)(o + sx), packed);
        }
#endif
        for (; sx < swidth; sx++) {
            int x = 2*sx;
            o[sx] = (r0[x] + r0[x+1] + r1[x] + r1[x+1] + 2) >> 2;
        }
    }
}

static void decimate2_box_task_fn(void *p)
{
    struct decimate2_box_task *t = (struct decimate2_box_task *)p;
    decimate2_box_rows(t->in, t->out, t->sy0, t->sy1);
}

image_u8_t *image_u8_decimate2_box_parallel(workerpool_t *wp, const image_u8_t *im)
{
    int swidth = im->width / 2, sheight = im->height / 2;
    image_u8_t *decim = image_u8_create(swidth, sheight);

    int nthreads = wp ? workerpool_get_nthreads(wp) : 1;
    if (nthreads <= 1 || sheight < 64) {
        decimate2_box_rows(im, decim, 0, sheight);
        return decim;
    }

    struct decimate2_box_task *tasks = malloc(sizeof(struct decimate2_box_task) * nthreads);
    int chunk = sheight / nthreads, rem = sheight % nthreads;
    int sy = 0;
    for (int i = 0; i < nthreads; i++) {
        tasks[i].in = im;
        tasks[i].out = decim;
        tasks[i].sy0 = sy;
        sy += chunk + (i < rem ? 1 : 0);
        tasks[i].sy1 = sy;
        workerpool_add_task(wp, decimate2_box_task_fn, &tasks[i]);
    }
    workerpool_run(wp);
    free(tasks);

    return decim;
}

void image_u8_gaussian_blur_parallel(workerpool_t *wp, image_u8_t *im, double sigma, int ksz) {
    if (sigma == 0)
        return;

    assert((ksz & 1) == 1); // ksz must be odd.

    // build the kernel.
    double *dk = malloc(sizeof(double)*ksz);

    // for kernel of length 5:
    // dk[0] = f(-2), dk[1] = f(-1), dk[2] = f(0), dk[3] = f(1), dk[4] = f(2)
    for (int i = 0; i < ksz; i++) {
        int x = -ksz/2 + i;
        double v = exp(-.5*sq(x / sigma));
        dk[i] = v;
    }

    // normalize
    double acc = 0;
    for (int i = 0; i < ksz; i++)
        acc += dk[i];

    for (int i = 0; i < ksz; i++)
        dk[i] /= acc;

    uint8_t *k = malloc(sizeof(uint8_t)*ksz);
    for (int i = 0; i < ksz; i++)
        k[i] = dk[i]*255;

    free(dk);

    image_u8_convolve_2D_parallel(wp, im, k, ksz);
    free(k);
}
