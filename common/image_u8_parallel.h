/**
 * @file image_u8_parallel.h
 * @author MqCreaple (gmq14159@gmail.com)
 * @brief Parallelized processing of various image_u8 related functions.
 * @version 0.1
 * @date 2025-08-07
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#pragma once

#include "image_u8.h"
#include "workerpool.h"
#include "math_util.h"

void image_u8_convolve_2D_parallel(workerpool_t *wp, image_u8_t *im, const uint8_t *k, int ksz);

void image_u8_gaussian_blur_parallel(workerpool_t *wp, image_u8_t *im, double sigma, int ksz);

// Decimate the image by ffactor, parallelized over output rows on the
// worker pool. ffactor == 1.5 uses the dedicated 3x3 -> 2x2 averaging
// scheme; other factors point-sample at integer steps.
image_u8_t *image_u8_decimate_parallel(workerpool_t *wp, const image_u8_t *im, float ffactor);
