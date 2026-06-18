#pragma once

#include "apriltag.h"
#include "common/image_u8.h"
#include "common/zarray.h"

// GPU implementation of the adaptive tile threshold stage. Returns NULL
// whenever the GPU path is unavailable or disabled (APRILTAG_OPENCL unset),
// in which case the caller must run the CPU implementation. A non-NULL
// result is byte-identical to the CPU implementation's output.
image_u8_t *oclThreshold(apriltag_detector_t *td, image_u8_t *im);

// GPU implementation of connected components + gradient clustering over a
// threshold image. Returns a zarray of zarray-of-struct-pt clusters with
// content and within-cluster point order identical to the CPU
// implementation (cluster order in the outer array may differ), or NULL
// when the GPU path is unavailable, in which case the caller must run
// connected_components + gradient_clusters on the CPU.
zarray_t *oclClusters(apriltag_detector_t *td, image_u8_t *threshim, int w, int h, int ts);

// Full GPU frontend over the (already decimated/blurred) input image:
// threshold, connected components, and gradient clustering in one device
// chain — the threshold image never materializes on the host. Same return
// contract as oclClusters.
zarray_t *oclFrontend(apriltag_detector_t *td, image_u8_t *im);

// Bridge the clusters returned by oclFrontend (zarray of zarray-of-OclPt) into
// the CPU fit's packed struct pt_list representation. `handled` is the
// per-cluster mask from oclFitQuads (or NULL): GPU-decided clusters are emitted
// empty (skipped by the CPU fit), the rest are materialized first. Consumes the
// input clusters.
zarray_t *oclClustersToPtList(zarray_t *gpuClusters, const uint8_t *handled);

// GPU fit_quads tail over the clusters most recently returned by oclFrontend
// (requires APRILTAG_OPENCL_FIT=1): fits quads to the device-resident sorted
// clusters and appends accepted quads — corner bits identical to the CPU's
// fit_quad — to quads. Returns a malloc'd per-cluster array where a 1 marks
// clusters fully decided on the GPU (the caller must skip those and run the
// CPU fit only for the rest), or NULL when the GPU fit is unavailable, in
// which case the caller runs the CPU path for every cluster. The caller
// frees the array.
uint8_t *oclFitQuads(apriltag_detector_t *td, zarray_t *clusters, image_u8_t *im, zarray_t *quads);
