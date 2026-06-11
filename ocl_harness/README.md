# GPU frontend validation harnesses

Standalone benchmarks and equivalence tests for the OpenCL frontend
(`ocl_threshold.c`). Build against the library, e.g.:

    gcc -O2 -o detbench ocl_harness/detect_harness.c -I. -Lbuild -lapriltag -lm -lpthread

Run with `OCL_ICD_VENDORS` pointing at the Intel OpenCL ICD and
`LD_LIBRARY_PATH` at the built library. Environment flags:

- `APRILTAG_OPENCL=1` enables the GPU frontend (silent CPU fallback).
  Output is bit-exact vs the CPU path by default.
- `APRILTAG_OPENCL_GATHER=1` fit-port groundwork (P1b): the build walk
  emits a permutation and a GPU gather materializes cluster-contiguous
  records on-device.
- `APRILTAG_OPENCL_GATHER_VALIDATE=1` reads the gathered records back and
  verifies them against the CPU-built clusters (dev gate, slow).
- `APRILTAG_OPENCL_SORTED=1` P1 validation scaffolding: GPU radix-sort
  grouping instead of the hash build walk. Not for production.
- `APRILTAG_OPENCL_PROFILE=1` per-kernel GPU timings + host-side frontend
  stamps on stderr.
- `APRILTAG_OPENCL_DEBUG=1` fallback diagnostics on stderr.

- `detect_harness.c` — full-detect CPU/GPU benchmark + detection equivalence.
- `corpus_harness.c` — multi-image × family × decimation equivalence sweep.
- `parity_harness.c` — threshold stage byte-parity + CPU accounting.
- `uf_harness.c` — GPU connected-components equivalence vs CPU unionfind.
- `cluster_harness.c` — GPU cluster extraction equivalence vs gradient_clusters.
- `profile_harness.c` — per-stage timeprofile of apriltag_detector_detect.
- `fp64_probe.c` — Arc 140T fp64 throughput probe in the lfps/fit shape.
