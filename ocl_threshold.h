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
// threshold image. Returns a zarray of zarray-of-struct-pt clusters whose
// content matches the CPU implementation (cluster and point order may
// differ), or NULL when the GPU path is unavailable, in which case the
// caller must run connected_components + gradient_clusters on the CPU.
zarray_t *oclClusters(apriltag_detector_t *td, image_u8_t *threshim, int w, int h, int ts);

// Full GPU frontend over the (already decimated/blurred) input image:
// threshold, connected components, and gradient clustering in one device
// chain — the threshold image never materializes on the host. Same return
// contract as oclClusters.
zarray_t *oclFrontend(apriltag_detector_t *td, image_u8_t *im);
