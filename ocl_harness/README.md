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
- `APRILTAG_OPENCL_FIT=1` fit-port P2 (implies the gather path): GPU
  per-cluster filter cascade, center, gradient dot, and a slope sort
  replicating CPU ptsort exactly, tie order included.
- `APRILTAG_OPENCL_FIT_VALIDATE=1` checks every cluster's flags, center
  and dot bits, and sorted order against a host replication of the CPU
  pre-sort semantics (dev gate, slow).
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
- `vide_rehearsal.py` — production-entry-point A/B: loads the stock library
  and this build side by side through dt_apriltags (vision-prod python,
  prod settings) and compares corners bitwise plus wall/CPU per detect.
- `vide_markers.json` — `SetWorldMarkersRequest` payload (deprecated
  single-scene form) that arms a robot's vide detector loop with four
  virtual tagStandard52h13 markers (ids 40000+ so nothing physical can
  match): `grpcurl -plaintext -d @ <robot>:5001
  terraform.vide.VideService/SetWorldMarkers < vide_markers.json`.

## Live test on a robot (vide)

Verified on w3cj 2026-06-11 (stock 3.54 -> GPU fit 1.29 cores of vide
service CPU). All robot-local, no closure rebuild:

1. Stage `libapriltag.so` at `/var/lib/vide-gputest/` (persistent; pin its
   nix-store deps with gcroot symlinks against GC).
2. If the robot has no VideConfig in rotunda, add a YAML config
   (camera serial + `arcade_udp: 127.0.0.1:4040 source_port 15000`) and an
   ExecStart drop-in appending `--config`.
3. GPU drop-in: `BindReadOnlyPaths=` our .so over the closure's
   `dt_apriltags/libapriltag.so` (the bind follows the store symlink to the
   real file) plus the APRILTAG_OPENCL/OCL_ICD_VENDORS environment.
   NOTE: `/etc` is immutable on NixOS 25.11 images — drop-ins go in
   `/run/systemd/system/vide.service.d/` and must be re-applied after a
   reboot.
4. Arm the detector with `vide_markers.json` (markers persist across vide
   restarts via its state cache).
5. Rollback: remove the GPU drop-in, `daemon-reload`, restart — stock
   binary guaranteed; in-lib failures already fall back to the bit-exact
   CPU path silently.
