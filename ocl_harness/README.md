# GPU frontend validation harnesses

Standalone benchmarks and equivalence tests for the OpenCL frontend
(`ocl_threshold.c`). Build against the library, e.g.:

    gcc -O2 -o detbench ocl_harness/detect_harness.c -I. -Lbuild -lapriltag -lm -lpthread

Run with `OCL_ICD_VENDORS` pointing at the Intel OpenCL ICD and
`LD_LIBRARY_PATH` at the built library. Environment flags:

- `APRILTAG_OPENCL=1` enables the GPU frontend (silent CPU fallback).
- `APRILTAG_OPENCL_EXACT=1` bit-exact output vs the CPU path (validation).
- `APRILTAG_OPENCL_PROFILE=1` per-kernel GPU timings on stderr.
- `APRILTAG_OPENCL_DEBUG=1` fallback diagnostics on stderr.

- `detect_harness.c` — full-detect CPU/GPU benchmark + detection equivalence.
- `corpus_harness.c` — multi-image × family × decimation equivalence sweep.
- `parity_harness.c` — threshold stage byte-parity + CPU accounting.
- `uf_harness.c` — GPU connected-components equivalence vs CPU unionfind.
- `cluster_harness.c` — GPU cluster extraction equivalence vs gradient_clusters.
- `profile_harness.c` — per-stage timeprofile of apriltag_detector_detect.
