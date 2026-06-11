/* OpenCL (Intel iGPU) acceleration for the AprilTag detector front-end.
 *
 * The GPU path targets the integrated GPU on Intel Core Ultra parts
 * (Arc 130T/140T class) using zero-copy unified shared memory: CPU and
 * GPU share DRAM, so stage handoffs cost microseconds. Every kernel
 * reproduces the CPU implementation's integer arithmetic exactly; the
 * threshold stage output (threshim + run tables) is bit-identical to
 * the CPU path.
 *
 * All entry points degrade gracefully: on any failure (no device, no
 * USM support, allocation failure) they return NULL/false once and the
 * caller keeps using the CPU implementation. Set APRILTAG_OPENCL=0 in
 * the environment to disable the GPU path entirely.
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "apriltag.h"
#include "apriltag_quad_internal.h"
#include "common/image_u8.h"
#include "common/unionfind.h"

typedef struct at_ocl at_ocl_t;

#ifdef __cplusplus
extern "C" {
#endif

// Lazily initializes the OpenCL context on first call (stored in
// td->ocl). Returns NULL if the GPU is unavailable or disabled.
at_ocl_t *at_ocl_get(apriltag_detector_t *td);

void at_ocl_destroy(at_ocl_t *ocl);

// Full threshold stage on the GPU: tile min/max, 3x3 tile blur,
// per-pixel threshold, and per-row run-length encoding. Returns the
// thresholded image (owned by the module, valid until the next call or
// destroy) and the frame run tables in shared memory, or NULL on
// failure (caller falls back to the CPU path).
image_u8_t *at_ocl_threshold(at_ocl_t *ocl, apriltag_detector_t *td, image_u8_t *im,
                             struct row_run **runs_out, uint32_t **row_off_out);

// Run-based connected components on the GPU (label-propagation with
// atomic-min hooking). Requires runs/row_off from at_ocl_threshold this
// frame. Returns a union-find whose parent/size arrays live in shared
// memory, flattened to canonical min-id labels — the same form
// canonicalize_uf produces on the CPU path. NULL on failure.
unionfind_t *at_ocl_connected_components(at_ocl_t *ocl, apriltag_detector_t *td,
                                         image_u8_t *threshim, int w, int h, int ts,
                                         struct row_run *runs, uint32_t *row_off);

#ifdef __cplusplus
}
#endif
