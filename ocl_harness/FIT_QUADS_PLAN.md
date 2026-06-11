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

- Doubles where the CPU evaluates in double; per-cluster serial
  accumulation preserves the CPU's floating-point operation order.
- Slope ties are NOT rare — the center-noise constants do not prevent
  them (vide frame: 2266 of 2510 sorted clusters carry ties, 199k tied
  pairs; consistent with PERF_NOTES in the faster branch). A total-order
  sort therefore reorders points in ~88% of clusters and would break
  bit-exactness downstream. The GPU sort instead REPLICATES ptsort's
  exact comparison network (its recursion tree is pure arithmetic on the
  cluster size), so tie order matches the CPU bit for bit and the
  contract strengthens to: identical detections, corners bit-identical.
- Float details that matter: FP_CONTRACT OFF (the CPU build has no FMA),
  -cl-fp32-correctly-rounded-divide-sqrt for the slope divide, the
  center computed in double then narrowed exactly as the CPU expression,
  and the gradient dot summed serially in point order from
  parallel-computed terms.
- Determinism run-to-run: fixed network + fixed reduction shapes = yes.

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
- P2 (DONE, measured on w3cj 2026-06-11): per-cluster preparation and
  slope sort, gated APRILTAG_OPENCL_FIT=1 (implies the gather path).
  Three kernels over the gathered records + descriptors:
  - fitPrep: one WG per cluster, tiny SLM for occupancy — the
    do_quad_task/fit_quad filter cascade, bbox reduce, double-evaluated
    center, per-point slope keys (orderedSlopeBits<<32 | point index),
    parallel dot terms, then one lane sums the terms in point order
    (bit-exact dot) and writes flags/meta.
  - fitSortSlm: ptsort-replica sort in two 4 KB SLM buffers for clusters
    of at most FIT_SLM_CAP=512 points (host-prefiltered id list).
  - fitSortBig: same replica in global memory for bigger clusters,
    batched FIT_BATCH=256 workgroups per launch with per-WG scratch
    slices; shallow depths split each merge across lanes with an exact
    right-biased merge-path search.
  The first cut ran the sort inside a monolithic fitPrep: 30 ms — SLM
  footprint (33 KB/WG) crushed occupancy and serial global merges burned
  the rest (a FIT_SLM_CAP=512 experiment halved it, proving occupancy).
  The split gets fitPrep 3.7 ms + fitSortSlm 4.8 ms + fitSortBig 4x0.6 ms
  (~11 ms chain, +8 ms detect wall over gather-only — it hides behind
  the CPU fit stage it will replace in P3). Gates: detect 0.000000 px;
  corpus 108/108; fit validation (APRILTAG_OPENCL_FIT_VALIDATE=1, host
  replication incl. a verbatim ValPt ptsort) 25/25 on the vide frame and
  216/216 across the corpus — flags, center/dot bits, bbox, and the full
  sorted sequence including every tie cluster match ptsort exactly.
  Still open (P3 polish): merge-path for the SLM sort's top levels;
  trim fitSortBig's list (host lists border-undecided candidates).
- P3 (DONE, measured on w3cj 2026-06-11): lfps + maxima + combos + line
  fits + corner checks on the GPU, quads-only readback. Three kernels
  after the P2 sort:
  - fitLfpsPrep/fitLfpsScan: compute_lfps over a six-PLANE layout
    (stride = total fit points). Prep (one WG per cluster) resolves the
    sorted-key indirection, samples the grayscale weight, and writes
    each point's six moment TERMS — the CPU's exact per-statement
    products — coalesced into the planes; scan runs one lane per
    (cluster, field) doing the in-place sequential add chain (the
    minimal serial work the summation-order contract allows). The
    first cut (serial lane inside the 256-wide WG, AoS rows) cost
    12-13 ms; the plane split runs prep 1.4 + scan 7.0 ms, now
    bandwidth-bound (~100 MB of plane traffic on shared DDR).
  - fitErrs: windowed fit_line errors per point (sqrtf-on-double
    narrowing kept as (double)sqrt((float)x)), the fixed 7-tap low-pass
    with filter constants computed by the host's libm at init and baked
    in as exact hex float build defines, order-preserving maxima
    compaction (per-256-chunk local scan), then the max_nmaxima cut as
    a lane-0 top-(K+1) multiset selection + strict-threshold filter
    (replicates the CPU's qsort-threshold exactly). ~4.4 ms.
  - fitCombos: all C(m,2) forward pair fits + C(m,2) wraparound closers
    computed once into SLM (the same fit_line values the CPU recomputes
    in its loop nest), combo scan in CPU lex-rank order as pure table
    lookups, (err, rank) argmin reduce (exact-tie -> lower rank = CPU
    first-wins), then lane 0 re-fits the winning four lines with params,
    intersections, float corner narrowing exactly where the CPU assigns
    quad->p, and the area/angle rejections over those float corners
    (which subtract in FLOAT before promoting). Per-combo fitLineC
    version cost 5.7-7.2 ms; the pair-table version runs ~2.2 ms.
  Host: runClusterChain leaves a pendingFit tag (oclFrontend path only —
  lfps needs the grayscale resident in bufIm); fit_quads() calls
  oclFitQuads(), which blocks on the P2 meta, splits clusters into
  fitPrep-flag rejections / GPU fit slots / CPU fallback (too big, over
  the FIT_POINT_CAP scratch cap, or chain failure -> fitOut status 0),
  uploads the (clusterIdx, lfpsOffset) list, enqueues the chain, maps
  the quads-only fitOut buffer (status + 8 corner floats per cluster),
  and returns a handled[] mask so do_quad_task skips decided clusters.
  Gates: detect 32/32 at 0.000000 px and deterministic run-to-run;
  corpus 120/120 at 0.0000 px; APRILTAG_OPENCL_FIT_VALIDATE=1 re-fits
  every GPU cluster with the production CPU fit_quad — verdicts and
  corner BITS match on all 2510 vide fits (443 quads, 2067 rejects)
  and across the corpus.
  Measured (vide 3088x2064, interleaved same-session): detect with
  APRILTAG_OPENCL_FIT=1 runs ~63-71 ms wall / ~82-88 core-ms vs
  gather-only ~47-54 / ~174-179 vs CPU ~58-60 / ~275-290. The fit mode
  trades ~+12 ms wall for another ~90 core-ms of CPU freed (-70% vs
  pure CPU overall): the GPU tail (fitPrep 3.1 + sorts ~6 + chain ~15)
  exceeds the ~13 ms CPU fit it replaces because the exactness contract
  pins the heavy stages to fp64, where the Arc 140T is weaker than the
  8-thread CPU. Both modes stay env-selectable: APRILTAG_OPENCL=1 alone
  for the fastest wall, +APRILTAG_OPENCL_FIT=1 for max CPU offload.

## P3 follow-ups (next session)

- The slim walk: with the fit on-GPU, the build walk's appendPt/merge
  copying (most of its 16-29 ms) only feeds desc sizes, the permutation
  passes, and CPU-fallback clusters. Track counts instead of
  materializing zarrays; reconstruct the rare fallback cluster from a
  bufRecordsAlt range readback (payloads are (x, y, gx, gy) in CPU
  point order — exactly struct pt). Biggest remaining CPU+wall lever.
- fitErrs/fitLfpsScan sit at the plane-traffic bandwidth floor; further
  wall wins come from the P2 sort polish (merge-path for the SLM sort's
  top levels, fitPrep dot throughput) or overlap across frames.

## Also still open (smaller)

- mergeEdges tiled local-memory CCL (~2.3 ms -> est. ~1 ms).
- compressAndCount (~2.5 ms): vertical run aggregation or subgroup reduce.
- Vide soak prep is blocked on distribution: ninja mode means no public
  branch; the nix overlay needs either a private remote or a local-path src.
