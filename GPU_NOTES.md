# GPU acceleration campaign (NUC15, Intel Arc iGPU)

Implementation of the hardware-acceleration plan from
`nuc15-hardware-report.md` on the ASUS NUC 15 Pro (Core Ultra 5 225H:
4 P + 8 E + 2 LP-E cores, Arc 130T-class iGPU, 112 EUs, unified
memory). Per-commit benchmarks live in `results.tsv` (133-image
vide_images2 corpus, 3088×2064, tagStandard52h13, hyperfine via
`bench.sh`); output equivalence is gated by `check.sh`.

## What runs where (default configuration)

| stage | where | notes |
| --- | --- | --- |
| threshold + RLE | GPU | tile min/max, 3×3 blur, fused threshold+RLE (movemask bit tricks), run tables in shared memory |
| unionfind (CCL) | GPU | run-graph edges reproducing `connect_runs_to_prev`, single-pass ECL-CC lock-free union, canonical min-id labels |
| make clusters | GPU sweep + CPU grouping | per-run sweep replays the CPU emission exactly; order-stable parallel grouping into a shared slab |
| fit quads | CPU (GPU optional) | CPU fitter reads the GPU-grouped slab directly; `APRILTAG_GPU_QUADS=1` runs the full bit-exact GPU kernel instead |
| decode + refine | CPU | per the report's recommended B2 architecture ("keep this repo's already-optimized decode") |

Every GPU stage is **bit-identical** to the CPU pipeline (4583/4583
detections on the corpus, max coordinate delta 0.0 px) and falls back
to the CPU implementation transparently on any failure
(`APRILTAG_OPENCL=0` disables the GPU entirely).

## Key design decisions

- **Intel USM everywhere.** Host USM for CPU-consumed outputs
  (threshim, run tables, cluster slab), device USM for GPU-internal
  data (image staging, labels, edges, scratch). The 6.4 MB frame
  handoff costs ~0.9 ms of coherency traffic no matter which side
  pays; everything else crosses for free.
- **Canonical min-id labels** (`canonicalize_uf`): union-by-size
  representatives depend on union order; rewriting every component's
  label to its minimum member id makes the labels a pure function of
  the input — which the GPU's atomic-min CCL produces natively. This
  is what makes CPU and GPU bit-comparable. Verified output-identical.
- **Exact emission-order preservation** in the cluster sweep: cluster
  identity, point order, and the (bucket-hash, id) cluster ordering
  all reproduce the CPU task/merge pipeline, so the angle-sort tie
  behavior downstream is untouched.
- **Bit-exact GPU floating point**: `-cl-fp32-correctly-rounded-divide-sqrt`,
  `FP_CONTRACT OFF`, sqrtf modeled as double→float→sqrt→double, the
  CPU merge sort replayed with identical splits/leaf networks/tie
  rules, Gaussian taps computed with host libm.

## Findings (measured, steady state)

- Thread re-tune alone (Phase 0): 39.9 → 22.6 ms/image. 12 workers on
  the 4P+8E cores; the LP-E cores hurt.
- Default hybrid: 23.6 ms/image with threshold, CCL, and the cluster
  sweep on the GPU and the CPU freed during the front-end.
- Per-dispatch overhead on this stack is ~0.5–1 ms (enqueue + USM
  residency + wake-up); batching all of CCL into one submission and
  the whole cluster stage into another was worth more than any
  in-kernel micro-optimization.
- The divergent per-run sweep walk costs ~2.5–3.5 ms per pass; a
  run→row lookup table beats per-run binary searches.
- The quad-fit kernel is bit-exact but ~17× slower than the
  hand-vectorized CPU stage (fp64-heavy, per-cluster serial structure,
  ~20 ns/point even for small clusters vs ~3 ns/point on the CPU).
  The report's hybrid split (GPU front-end, CPU geometry/decode) is
  empirically right for Xe-LPG-class iGPUs; an all-GPU configuration
  measures 139.7 ms/image (see results.tsv).

## Environment toggles

- `APRILTAG_OPENCL=0` — disable the GPU path.
- `APRILTAG_GPU_QUADS=1` — run quad fitting on the GPU (bit-exact,
  slower on this hardware).
- `APRILTAG_OCL_PROF=1` — per-kernel GPU timings on stderr.
- `APRILTAG_CCL_VERIFY=1` — compare GPU CCL labels/sizes node-for-node
  against the CPU union-find every frame.
- `OCL_ICD_VENDORS=/run/opengl-driver/etc/OpenCL/vendors` — required
  on NixOS for ICD discovery.
