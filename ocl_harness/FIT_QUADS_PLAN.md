# fit_quads GPU port — design and measured feasibility

Goal: move the last big CPU stage (~12-15 ms wall, fit_quads) onto the GPU,
taking the detect pipeline to roughly 25 ms wall with CPU usage in the tens
of core-ms. With fit on the GPU, boundary records never return to the host:
the CPU receives ~100 candidate quads (KB) instead of ~43 MB of records, and
the cluster build walk disappears entirely.

## Measured feasibility (w3cj, Arc 140T, 2026-06-11)

- `cl_khr_fp64` is exposed, including `__opencl_c_ext_fp64_global_atomic_add`.
- `ocl_harness/fp64_probe.c`: 9000 clusters x 300 pts of sequential
  double-precision moment accumulation with one image sample per point
  (the compute_lfps + fit_line shape, calibrated to the measured 2.7M
  record count): **2.56 ms**. The serial-per-cluster scan pattern that
  preserves CPU summation order is affordable.

## End-state architecture

1. emitSegments leaves records on the device (raster order, key-tagged).
2. **Permutation + gather** (P1b, replaces the P1 radix sort): the host
   build walk already discovers the grouping, so it emits a permutation
   (source record index per output slot, clusters contiguous, points in
   raster order) and one coalesced gather kernel materializes
   cluster-contiguous records on-device (bufRecordsAlt). Within-cluster
   point order is exactly the CPU emitter's order by construction.
3. Per-cluster descriptor array (offset, count) uploaded alongside the
   permutation + size/hull filters from fit_quads' caller.
4. Per-cluster fit, one workgroup per cluster:
   - bbox + center reduce; per-point slope; gradient dot -> reversed_border
     filter (threads parallel, local reduce).
   - sort by slope: segmented bitonic on (slope, position-index) pairs.
   - lfps prefix moments: ONE lane per cluster, sequential double
     accumulation (preserves CPU summation order; probe says this is cheap).
     Image weight samples read from the device-resident input image.
   - windowed errs: parallel per point via prefix moments (fit_line is O(1)).
   - maxima extraction + top-K by err (local compact + small sort).
   - corner combination search (<= ~210 combos for max_nmaxima=10):
     parallel across threads, local argmin reduce.
   - final 4 fit_line + intersections + angle/area checks: lane 0.
5. Readback: quad array only. CPU runs decode unchanged.

## Exactness contract

- Doubles end-to-end; per-cluster serial accumulation preserves the CPU's
  floating-point operation order, so corners should match to the last bit
  except where slope TIES exist: CPU ptsort's tie order is a sorting-network
  artifact; the GPU sorts by (slope, raster index), a total order. The
  fit_quad center-noise constants make exact float slope ties rare; the
  corpus harness quantifies any residual difference. Contract: identical
  detection sets; corners < 1e-6 px except tie-affected clusters.
- Determinism run-to-run: total-order sort + fixed reduction shapes = yes.

## Phases (each gated on the corpus harness)

- P1 (DONE, measured): stable GPU radix sort by compacted key, env-gated
  APRILTAG_OPENCL_SORTED. Correctness: bit-exact (detect 0.000000 px,
  corpus 108/108) — stability + raster emission provably yields exact
  CPU cluster content and point order. Performance: radixScatter is
  ~10.9 ms/pass x 6 (scattered 16 B writes + low-occupancy ranking) —
  a full global sort is the wrong tool. KEPT in tree as validation
  scaffolding; do not enable in production.
- P1b (DONE, measured on w3cj 2026-06-11): permutation + gather, gated
  APRILTAG_OPENCL_GATHER=1 until the GPU fit consumes it. The permutation
  is computed as a counting sort, not per-cluster index lists — the list
  version cost ~13 ms/frame of host bookkeeping (walk +5.8 ms from 2.7M
  zarray appends, 2.7 ms single-thread flatten, 3.0 ms freeing 9k
  zarrays) and was rewritten. Final shape: the walk stores each record's
  task-local cluster index into a flat uint32 array (pass A, one
  sequential store per record); the merge records each task-local
  cluster's final index and chunk start within the final cluster; once
  final cluster offsets are known (running sum of sizes, also fills the
  descriptors), a parallel workerpool pass computes each record's output
  slot and writes the permutation straight into the mapped staging
  buffer (pass B). One coalesced gather kernel (~1.5 ms GPU, async with
  downstream CPU fit) materializes cluster-contiguous records in
  bufRecordsAlt. Measured host cost: descFill 0.13 ms, permPass 1.8 ms,
  map/unmap ~1.1 ms — detect wall/cpu indistinguishable from no-gather.
  Gates: detect PASS 0.000000 px; corpus 108/108 @ 0.0000; gathered
  content vs CPU clusters (APRILTAG_OPENCL_GATHER_VALIDATE=1) 216/216.
  P3 note: with fit on-GPU the walk's appendPt/merge copying disappears;
  what remains on the host is the hash probe + pass A/B — the slim walk.
- P2: per-cluster slope/filter/slope-sort on GPU; validate sorted point
  order against ptsort output (tie cases logged).
- P3: lfps + maxima + combos + line fits; quads-only readback; corpus
  detection equivalence + timing.

## Also still open (smaller)

- mergeEdges tiled local-memory CCL (~2.3 ms -> est. ~1 ms).
- compressAndCount (~2.5 ms): vertical run aggregation or subgroup reduce.
- Vide soak prep is blocked on distribution: ninja mode means no public
  branch; the nix overlay needs either a private remote or a local-path src.
