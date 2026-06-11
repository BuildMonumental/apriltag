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

// One emitted boundary point, in exact CPU emission order; slot is an
// index into probe_dense.
struct at_gc_rec { uint32_t slot; uint16_t x, y; int16_t gx, gy; };

struct at_gc_out {
    const struct at_gc_rec *recs; // flat record stream, emission order
    uint32_t nrecs;
    const uint32_t *probe_dense;  // hash-probe slot -> dense cluster index
    const uint64_t *dir_ids;      // dense cluster index -> clusterid
    uint32_t nclusters;
};

// Gradient-cluster sweep on the GPU: emits every boundary point (with
// its cluster) in the exact order the CPU sweep produces. The caller
// groups records into clusters. Requires this frame's threshold + CCL
// to have run on the GPU. False on failure.
bool at_ocl_gradient_clusters(at_ocl_t *ocl, apriltag_detector_t *td,
                              int w, int h, int ts, int min_cluster_pixels,
                              struct at_gc_out *out);

// Enqueues the whole GPU front end (threshold, CCL, cluster sweep) for
// im and returns without waiting for the back half; the next detect of
// the same image picks up the in-flight results. Lets the GPU work
// overlap unrelated CPU work (image loading, the previous frame's
// decode, ...).
bool at_ocl_frontend_begin(at_ocl_t *ocl, apriltag_detector_t *td, image_u8_t *im,
                           int min_cluster_pixels);

// Copies this frame's GPU labels into the shared union-find arrays
// (deferred from the CCL batch; needed before any CPU code reads the
// union-find returned by at_ocl_connected_components).
bool at_ocl_ensure_uf(at_ocl_t *ocl);

// One quad-fit result per cluster (cluster order).
struct at_quad_out { float p[4][2]; int32_t valid; int32_t reversed; };

// Shared-memory arrays for the grouped clusters: the caller places
// points (in cluster-sorted order) into pts_slab and fills coff/csz
// per cluster, then calls at_ocl_fit_quads.
bool at_ocl_quad_prepare(at_ocl_t *ocl, uint32_t ncl, uint32_t total_pts,
                         struct pt **pts_slab, uint32_t **coff, uint32_t **csz);

// Full fit_quad per cluster on the GPU (angle keys, the exact CPU
// merge sort, line-fit prefix sums, maxima search, candidate quads).
// Returns the per-cluster results, or NULL on failure.
const struct at_quad_out *at_ocl_fit_quads(at_ocl_t *ocl, apriltag_detector_t *td,
                                           uint32_t ncl, int w, int h,
                                           int min_cluster_pixels, int tag_width,
                                           bool normal_border, bool reversed_border);

#ifdef __cplusplus
}
#endif
