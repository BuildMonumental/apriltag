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

// Anti-aliased factor-2 decimation: each output pixel is the average of the
// corresponding 2x2 input block. Output is floor(w/2) x floor(h/2). In the
// pixel-corner coordinate convention used by the quad detector, a coordinate
// c on the decimated image maps to 2*c on the input image.
image_u8_t *image_u8_decimate2_box_parallel(workerpool_t *wp, const image_u8_t *im);
