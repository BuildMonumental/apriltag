# Detector optimization notes (faster branch)

Campaign: 88 ms → 39.8 ms/image detector total (2.21x) on the 133-image
`vide_images` corpus, 4 threads, default parameters, tagStandard52h13.
Output is byte-equivalent to baseline acf5e20 at every commit (4583/4583
detections, coords within 1e-4 px — FMA contraction noise only).

## Measurement (do not skip this section)

- Machine: AMD EPYC 9454P (8 CCDs × 6 cores), NixOS, shared with a CI
  runner that randomly loads it 20-100x. Absolute timings swing wildly.
- `./benchmark.sh` — official number (hyperfine wall + per-stage detector
  table from `--save-timing`). Pinned via `taskset -c 0-3`.
- `./ab.sh <buildA> <buildB> [rounds]` — interleaved A/B with paired
  per-round ratios; the only trustworthy comparison under load. Additive
  noise biases ratios toward 1, so quiet-window numbers are the honest ones.
- `./check.sh` — epsilon-gated output equivalence vs the baseline corpus
  run (id/hamming exact, coords ≤0.1 px, margin ≤1.0).
- **Pinning is worth ~28 ms/frame**: unpinned, the idle-machine scheduler
  spreads the 4 workers across CCDs and the shared structures bounce
  between L3s. Exactly 4 CPUs is optimal; 5-6 in the affinity set lets the
  scheduler migrate threads and is *much* worse. Under CI load the numbers
  can look *better* than quiet because load packs the threads onto one CCD.

## What the speedup is made of (stage ms, baseline → now)

- threshold 3.6 → 1.7: SIMD tile min/max + blur + compare; RLE fused into
  the threshold tasks (per-task contiguous buffers, one memcpy each);
  buffers cached across frames.
- unionfind 13.8 → 4.7: union-find indexed by *run* (~1.6 MB, not 50 MB);
  one union per adjacent same-value run pair; interleaved-root connect
  with full path compression; no per-frame reset (cached).
- make clusters 27.1 → 14.0: segment-driven emission (entry resolved once
  per run pair, interior pairs as packed u64 / AVX2 stores); per-run
  rep+gate cache with read-only finds; k-way heap merge of per-task lists
  with exact-size fragment-group concatenation (parallelized); pooled
  cluster_hash entries; flat single-allocation clusters {size, pts[]}.
- fit quads 34.6 → ~14: 8-byte points; u64 sort keys (slope<<32 | ~idx);
  vectorized key/bbox/dot loops; 4-way fused merge sort; SoA moments;
  memoized segment fits; vectorized window-error + 7-tap filter + maxima
  scan; quickselect threshold; per-task scratch reuse; lfps pass 1 in
  emission order (image-locality), pass 2 gathers through key indices.
- post-fit serial 2.2 → ~0.07: per-task quad accumulation.
- decode+refine 6.5 → ~3.2: per-task scratch; truncating casts replacing
  modf; homography + graymodel coefficient hoists; refine_edges sampling
  loop vectorized 4-wide (masked lanes contribute exact 0.0).

## Load-bearing invariants (violating these changes detections)

- **Angle-sort tie order.** 87% of clusters contain equal slope keys: the
  ±2^16 quadrant constants crush the dy/dx mantissa. The legacy order is
  produced by hi-word-only leaf networks plus take-right-on-tie merges —
  an inconsistent comparator, so NO standard sort reproduces it; only the
  exact merge-tree structure does. Full-u64 leaves (a consistent order)
  lose 1 detection in 4583 with 0.25 px shifts.
- The (y, x, conn-order) emission sequence of cluster points, the
  connected_last suppression, and the component-size gates (lazy,
  evaluated against sizes at scan time).
- Union-find *rep values* may change freely (tree shape is internal);
  components and sizes may not.
- The union-find task chunking can leave the last image row uncovered for
  some heights; those run nodes must be initialized (see covered_end).

## Measured dead ends (don't re-try without new information)

- LSD radix sort on full-u64 keys: only −2 ms vs the 4-way fused merge,
  and −1 detection. Even ignoring output, sorts are within ~2 ms of the
  committed one.
- 8-way fused merge: +2.5 ms (7-compare tournament too serial). 4-way is
  the optimal fusion depth; 2-way costs an extra pass.
- Two-pass emission (count, allocate exact, write direct): +8.5 ms — the
  per-pair sweep machinery dominates, not the point stores.
- Bit-loop and patterns-loop SIMD in decode: neutral (too few samples per
  quad to amortize lane fold-out). refine_edges SIMD was the win (−1.5).
- Frame arena for cluster allocations: neutral (glibc tcache already
  covers it). Merged clusters can exceed 256 KB — size slots if revived.
- Big-cluster-first scheduling in fit_quads: +4 ms (count-based chunking
  concentrates the heavy clusters; hash order already scatters them).
- `__attribute__((flatten))` on the sort: −3% (I-cache bloat).
- `-funroll-loops`: −7%. PGO: +7% bit-identical (excluded by decision).
- Hugepage threshim, SMT-sibling affinity, malloc tunables, 512-bit
  vector width: all neutral or worse.
- Rem's union-find: incompatible with the size gates (splices merge
  components without visiting roots).
- No duplicate cluster points exist (measured) — nothing to dedup.
- No over-cap fragments exist (measured) — nothing to truncate early.

## Where the remaining time is (quiet, pinned, 39.8 total)

threshold 1.7 · unionfind 4.7 · make clusters 14.0 · fit quads ~14 ·
decode ~3.2 · serial glue ~1.5. Profiles are flat inside the big two
(cost spread across pair machinery and merge passes; no hotspot).
3x (29.3 ms) was determined infeasible under the output-equivalence
constraint on this machine — and relaxing the constraint only buys ~2 ms.

# Decimation mode (faster-dec branch, 2026-06-11)

Goal: a step change beyond the exact-output ceiling by running the quad
*search* at half resolution while keeping edges, refinement and decode on
the full-resolution image. Detection loss budget ≤5% vs `-x 1.0`, accuracy
must stay (refine_edges already anchors corners to full-res gradients).

## What was built

1. **Anti-aliased factor-2 decimator** (`image_u8_decimate2_box_parallel`):
   2x2 box average, AVX2 + workerpool, replacing the strided point-sampler
   for `quad_decimate=2` (0.21 ms vs 1.59 ms, and no aliasing).
2. **Full-resolution thresholding, half-resolution geometry**: with
   `quad_decimate=2` the adaptive threshold runs on the *full* image (its
   4px tiles and per-pixel decisions see full contrast on thin borders),
   then each 2x2 block of {0,127,255} decisions is collapsed by
   black-priority voting (`vote_decim_row`, AVX2) into the half-res
   threshim that union-find/clustering/fit consume. Fused into the
   threshold+RLE tasks; the full-res threshim is kept for step 3.
3. **Full-resolution ROI fallback** (`run_roi_fallback`): tags whose
   borders are ~1px at half res never form a closed boundary cluster, no
   matter the gates (verified: gate sweeps changed nothing; the loss is
   topological). But they leave a signature: tag-sized cluster bounding
   boxes (collected during the decimated pass) that nest concentrically
   (inner border ring inside the outer data-ring boundary, area ratio
   ~2.8) with the polarity the tag family demands. Regions with that
   signature and no detection are cropped from the full-res image into a
   padded atlas (127 fill = "skip"), the standard pipeline runs once over
   the atlas reusing the *already computed* full-res threshold decisions
   (no re-threshold, no tile artifacts at crop edges), and the resulting
   quads are translated back and decoded. Selection is priority-ordered
   (geometry-true pairs, then near-detection singletons) under a
   per-frame area budget: `td->roi_fallback_budget` (px², default 8e5,
   0 disables; demo flag `--roi-budget`).

## Results (133-frame corpus, 3088x2064, 4 threads, quiet machine)

| config                  | ms/img | speedup | detections lost | corner err |
|-------------------------|--------|---------|-----------------|------------|
| `-x 1.0` (reference)    | 39.6   | 1.00x   | —               | —          |
| `-x 2.0` budget 0       | 17.5   | 2.26x   | 18.7%           | 0.052 px   |
| `-x 2.0` budget 8e5 (default) | 24.6 | 1.61x | 4.8%          | 0.064 px   |
| `-x 2.0` budget 1.2e6   | ~27    | ~1.45x  | 4.1%            | 0.063 px   |

Matched-tag accuracy is unchanged in practice (mean corner error 0.06 px,
99.4% of corners within 0.5 px) because refinement and decode always ran
at full resolution. The loss budget is a smooth dial: every ~200k px² of
fallback budget buys roughly 0.5-0.7% retention and costs ~1 ms.
`-x 1.0` output is byte-identical to the pre-decimation branch.

## Dead ends (measured)

- Box filter alone: no retention change (the loss is in segmentation
  topology, not sampling phase).
- Gate relaxation (min_cluster_pixels, min_white_black_diff,
  max_line_fit_mse, critical_rad): zero recovered detections — the
  missing tags' clusters fail fit_quad on corner geometry (winding
  dominates), or never form.
- Sharpening the decimated image (quad_sigma<0): 19.7%→15.4% loss for
  +9 ms. Poor trade vs the ROI fallback.
- refine_edges range/sample tuning and iteration: neutral to harmful
  (initial corners off by more than a tight search range; extra passes
  over-pull onto neighboring data-cell edges).
- Fragment-pair (overlapping same-scale boxes) ROI class: worse than
  spending the same budget on nested pairs + singletons.
- min_cluster_pixels=10 to surface more candidates: +21 detections,
  +13 ms. Not worth it.
