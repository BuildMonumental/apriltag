# Hardware-Accelerated AprilTag Detection on the ASUS NUC 15 Pro (NUC15CRK)

*What the machine actually contains, and how to take detection past an already-optimized CPU pipeline.*

June 10, 2026 · baseline: AprilRobotics/apriltag `faster` branch, 39.8 ms/frame

---

This report answers one question: the AprilTag detector has been hand-optimized on CPU from 88 ms to **39.8 ms/image** (2.21×, 4 threads, 1456×1088-class grayscale, tagStandard52h13), profiles are flat, and a CPU-only 3× was judged infeasible — *so what does the NUC15CRK's silicon offer beyond CPU SIMD?* Findings were produced by a 104-agent research run (22 sources fetched, 110 claims extracted, 25 adversarially verified by 3-vote panels: 23 confirmed, 2 refuted) plus targeted spec verification.

Every load-bearing number is tagged:

- **[VERIFIED]** — survived 3-0 adversarial verification with a primary source
- **[CORROBORATED]** — multiple secondary sources agree
- **[DERIVED]** — arithmetic from verified specs
- **[EXTRAPOLATED]** — estimate with no direct measurement; treat as a hypothesis

## TL;DR — the big swing is the iGPU, and there is a proven blueprint for it.

- **The NUC15CRK's Arc 140T iGPU is the right accelerator for exactly the stages that dominate the profile.** FRC Team 971's open-source CUDA detector runs threshold, union-find connected components, boundary extraction, angle sort, line fitting, and quad fitting — i.e. the stages costing 4.7 + 14.0 + 14 ms here — entirely as GPU kernels, then decodes on CPU. Its connected-components core is a peer-reviewed, vendor-neutral algorithm (Allegretti et al., BKE). Nobody has ported it to Intel; a SYCL/Level Zero port is the project. **[VERIFIED]**
- **Zero-copy makes the hybrid split nearly free on this machine.** CPU and iGPU share DRAM; Intel documents that zero-copy buffer sharing always beats copying on integrated graphics. The frame is 1.6 MB — handing buffers between iGPU front-end and CPU decode costs microseconds, not milliseconds. **[VERIFIED]**
- **The ceiling is roughly 13–20× below the current latency.** NVIDIA's GPU AprilTag pipeline benchmarks at 2.0–2.9 ms per 720p frame (RTX 5090 / AGX Thor) — different hardware, resolution and tag family, but it proves the architecture, not just the silicon, is what gets you to single-digit milliseconds. A realistic Arc 140T target is **6–14 ms/frame**. **[EXTRAPOLATED]**
- **The NPU (13 TOPS) is a sidecar, not the engine.** Full-resolution learned detection doesn't fit (a Deep-ChArUco-class net scales to ~200 ms-equivalent at this resolution), and learned detectors that are fast (YoloTag) don't decode tag IDs. The credible NPU role: a low-res proposal/ROI network that lets the classical pipeline skip empty frames or crop to tag regions — at zero CPU/GPU cost. **[VERIFIED]**
- **Cheap wins first:** the 39.8 ms figure was measured on 4 pinned EPYC cores. The NUC has 16 cores (6 P @ 5.1 GHz + 8 E + 2 LP-E) and much higher per-core clocks — re-baseline and re-tune thread count and P/E placement before writing any GPU code. There is no AVX-512 to chase. **[CORROBORATED]**

## 1 · Starting point: where the 39.8 ms goes

Stage costs from the optimization campaign (`PERF_NOTES.md`, quiet machine, pinned, 4 threads, 133-image corpus). The two cluster-geometry stages are 70% of the budget and are precisely what GPU prior art accelerates:

| Stage | Cost |
| --- | --- |
| threshold | 1.7 ms |
| unionfind (CCL) | 4.7 ms |
| make clusters | 14.0 ms |
| fit quads | ~14 ms |
| decode + refine | ~3.2 ms |
| serial glue | ~1.5 ms |

Caveat that matters for everything below: this baseline was measured on an AMD EPYC 9454P with 4 pinned cores, not on the NUC. Per-core, Arrow Lake-H P-cores (Lion Cove, 5.1 GHz turbo) are substantially faster; core count and cache topology differ completely. **Phase 0 of any plan is re-measuring this baseline on the NUC itself.**

## 2 · Verified hardware inventory

| Component | What the NUC15CRK has | Status |
| --- | --- | --- |
| **CPU options** | Arrow Lake-H: Core Ultra 5 225H / 235H (vPro), Ultra 7 255H / 265H (vPro), all cTDP 40 W (plus non-Ultra 100U/120U/210H/240H tiers at 25 W). The Ultra 9 285H belongs to the separate NUC 15 Pro+ (NUC15CRS), not this model. [ASUS techspec] | **[VERIFIED]** |
| **Cores / threads** | Ultra 7 255H: 6 P-cores (Lion Cove, up to 5.1 GHz) + 8 E-cores (Skymont, up to 4.4 GHz) + 2 LP-E cores = 16 cores / 16 threads — **SMT removed**. Ultra 5 225H: 4P + 8E + 2LP-E = 14 cores. 24 MB (255H) / 18 MB (225H) smart cache. | **[CORROBORATED]** |
| **ISA** | AVX2 + AVX-VNNI. **No AVX-512** — disabled across Arrow Lake client parts. The existing AVX2 hand-vectorization is already the right width for this machine. | **[CORROBORATED]** |
| **iGPU** | Intel **Arc 140T** on Ultra 7 SKUs (130T on Ultra 5): 8 Xe cores (Xe-LPG+, first-gen Xe with XMX added) × 16 vector engines, 8 RT units, 8 MB L2, up to 2.25 GHz. **XMX matrix engines present: 77 TOPS INT8** — Lunar-Lake-class AI throughput. ASUS's spec page only says "Intel Arc GPU"; the 140T attribution is from retail listings + Notebookcheck. | **[CORROBORATED]** |
| **iGPU compute** | ~4.6 TFLOPS FP32 vector (8 Xe × 16 XVE × SIMD8 × 2 FMA × 2.25 GHz) — about 2–3× a Jetson Orin NX's FP32. For comparison: the entire 4-thread CPU pipeline today uses a fraction of one TFLOP. | **[DERIVED]** |
| **NPU** | Intel AI Boost, same NPU generation as Meteor Lake (NPU 3): **13 TOPS INT8**. OpenVINO-only in practice; FP16 hardware precision, INT8/mixed quantized inference, **static input shapes only**. | **[VERIFIED]** |
| **Memory** | Dual-channel DDR5: 2× 48 GB DDR5-5600 SO-DIMM or DDR5-6400 CSO-DIMM (ASUS-validated max 96 GB). Unified-memory bandwidth ceiling **~102.4 GB/s** (DDR5-6400 dual channel), shared by CPU + iGPU + NPU. The 134 GB/s Arrow Lake-H platform figure is LPDDR5x-8400 only — not available on this socketed machine. | **[VERIFIED]** |
| **Expansion** | 2× Thunderbolt 4 (USB4, PCIe tunneling at 32 Gbps ≈ PCIe 3.0 ×4) → **eGPU is feasible**. Two M.2 Key-M slots: 2280 PCIe 5.0 ×4 + 2242 PCIe 4.0 ×4. The 2242 slot is an electrically plausible — but *not ASUS-validated* ("for NVMe only") — fit for a Hailo-8 M.2 accelerator (2242 Key-M, PCIe Gen3 ×4, 8.25 W). | **[VERIFIED]** |

> **Bandwidth sanity check** — a 1456×1088 grayscale frame is 1.6 MB. Even at 200 fps with the iGPU re-reading the frame five times per stage, the pipeline consumes <2 GB/s of the 102 GB/s budget. *Shared-memory bandwidth is a non-issue for this workload; kernel-launch latency and occupancy are what will actually determine iGPU performance.* **[DERIVED]**

## 3 · Unit-by-unit analysis

### 3.1 · Arc 140T iGPU — the main event

**The existence proof.** FRC Team 971 built a complete, production GPU AprilTag detector in CUDA for Jetson Orin NX, and its kernel inventory maps one-to-one onto this repo's expensive stages **[VERIFIED]**:

| This repo's stage (cost) | frc971 GPU equivalent | Portability to Intel |
| --- | --- | --- |
| threshold (1.7 ms) | Halide-generated threshold kernel | Trivial — embarrassingly parallel |
| unionfind / CCL (4.7 ms) | Allegretti–Bolelli–Grana **BKE** block-based GPU connected components (IEEE TPDS, via YACCLAB) | **Vendor-neutral, peer-reviewed algorithm** — the hardest stage is the best-documented one |
| make clusters (14.0 ms) | BlobDiff boundary extraction + `cub::DeviceSelect` stream compaction | cub → oneDPL/SYCL group algorithms |
| fit quads, incl. angle sort (~14 ms) | cub radix sort-by-angle + `DeviceScan` prefix sums + FitLines/FitQuads kernels + peak detection | oneDPL radix sort; scans are standard SYCL |
| decode + refine (~3.2 ms) | **Stays on CPU** (frc971 calls the stock apriltag decoder) | Keep this repo's already-optimized decode — it is family-agnostic, so **tagStandard52h13 keeps working** |

A standalone extraction exists ([Team766/apriltags_cuda](https://github.com/Team766/apriltags_cuda) — `cuda_frc971.cu`, `apriltag_gpu.cu`, `labeling_allegretti_2019_BKE.cu`, `threshold.cu`) with the FRC framework dependencies already cut out. It is CUDA pinned to NVIDIA (compute capability 8.7); **no SYCL/oneAPI port exists anywhere** — that port is the engineering project. ZLUDA is not an escape hatch (it never shipped Intel Arc support); Intel's SYCLomatic migration tool is the realistic starting point, with cub calls mapped to oneDPL. **[VERIFIED]**

**Why the hybrid split works here specifically.** On a discrete GPU the "GPU front-end, CPU decode" split pays PCIe round-trips. On this machine it doesn't: CPU and iGPU share physical DRAM, and Intel's own optimization guidance states zero-copy sharing *always* beats copying on integrated graphics (oneAPI USM / `zeMemAllocShared` / OpenCL `CL_MEM_USE_HOST_PTR` with 4096-byte alignment and 64-byte-multiple sizes — get the discipline wrong and the driver silently copies). The quad list handed back to the CPU decoder is a few kilobytes. **[VERIFIED]**

**Programming routes, by ecosystem maturity on Linux:** Level Zero + SYCL (oneAPI) is the first-class path on Arc and the one Intel invests in; OpenCL 3.0 is mature and fine for hand-written kernels; OpenCV's UMat/OpenCL path gets the threshold stage nearly for free but nothing else in this pipeline; Vulkan compute works but buys nothing over SYCL here. The Xe-LPG+ XMX engines (77 TOPS INT8) are *not* needed for the classical pipeline — it is integer/branch-heavy, not matmul-heavy — but they're the reason the iGPU is also a credible inference device (§3.2).

> **Repo-specific risk — byte-equivalence dies on the GPU.** The campaign's invariant analysis (`PERF_NOTES.md`) showed the legacy angle-sort tie order is reproduced only by the exact CPU merge-tree (an inconsistent comparator); even a consistent full-u64 CPU sort changes 1 detection in 4583. A GPU radix sort will not reproduce it either. frc971 accepted ε-level output drift; this project would too. The existing `check.sh` epsilon harness (id/hamming exact, coords ≤0.1 px) is the right acceptance gate for the GPU port — byte-equivalence has to be formally retired as the bar.

### 3.2 · NPU (13 TOPS) — a sidecar with one good job

**What's been proven elsewhere** **[VERIFIED]**: [YoloTag](https://arxiv.org/abs/2409.02334) (RO-MAN 2024) runs a lightweight YOLOv8 fiducial detector at 55 fps (~18 ms) on a Quadro P2200 — beating the classical CPU detector's 24 fps in their UAV setup — and [DeepTag](https://arxiv.org/abs/2105.13731) (IEEE TVCG) shows CNNs can detect existing marker families including AprilTag. But the fine print bounds the ambition: *YoloTag does not decode tag IDs* (markers are one generic class; its training labels are bootstrapped from the classical detector), and DeepTag demonstrates capability, not speed (3 fps in YoloTag's comparison).

**The compute-scale wall** **[VERIFIED]**: Deep ChArUco's corner network runs ~100 fps at 320×240 on a GTX 1080 (~8.9 TFLOPS). Fully-convolutional cost scales with pixels: at 1456×1088 that's ~20.6× the work — ~200 ms-equivalent on that GPU, far beyond a 13 TOPS NPU per frame. A full-resolution learned replacement for the pipeline is out.

**The credible role:** a small static-shape INT8 proposal network at ~364×272 (¼ scale) on the NPU, running concurrently with — and costing nothing on — the CPU and iGPU:

- **Frame gating:** skip the classical pipeline entirely on tag-free frames (huge average-fps win in sparse scenes, zero worst-case win).
- **ROI cropping:** run the classical detector only on proposed tag regions; with tags covering ~10% of the frame, the classical stages shrink near-proportionally.
- **Tracking prior:** seed next-frame ROIs from current detections; the NPU net only handles (re)acquisition.

OpenVINO constraints to design around **[VERIFIED]**: FP16 hardware precision (INT8/mixed via quantization), static shapes only — fix the input resolution at export time. The accuracy risk is real: a missed proposal is a missed tag, so the conservative deployment is gating/prioritization, never hard filtering, until recall is measured on your corpus.

### 3.3 · CPU — what's still on the table (less than you'd hope)

- **Re-baseline + thread scaling**: the EPYC campaign found exactly 4 pinned cores optimal *on that topology*. The 255H has 6 fast P-cores on one ring with 24 MB shared cache — the sweet spot is likely 6 workers on P-cores, possibly +8 E-core workers for the parallelizable stages. This is a config experiment, not an engineering project. **[EXTRAPOLATED]**
- **Thread Director / P-E placement**: pin the latency-critical serial sections to P-cores; E-cores take per-cluster work. Linux ≥6.6 handles this passably by default; explicit affinity will beat it for a pipeline this tuned.
- **AVX-VNNI**: int8 dot products could touch the threshold tile min/max and decode sampling, but those stages are 1.7 + 3.2 ms with flat profiles — the campaign already established there's no hotspot left to vectorize harder. Expect ≤1–2 ms total. **[EXTRAPOLATED]**
- **Intel IPP / oneAPI image primitives**: generic morphology/CCL routines are very unlikely to beat this repo's specialized run-based unionfind (already 3× faster than baseline and structured around the output invariants). Not worth the integration cost.
- **No AVX-512** on Arrow Lake — the 512-bit experiment was already a measured dead end on EPYC anyway.

### 3.4 · Expansion hardware — the brute-force options

- **Thunderbolt 4 eGPU** **[VERIFIED]** (feasibility): 32 Gbps PCIe tunneling moves a 1.6 MB frame in ~0.5 ms. An NVIDIA eGPU runs Team766's CUDA code *today* with CPU decode keeping tagStandard52h13. It's the fastest path to single-digit milliseconds but abandons the mini-PC's point — cost, size, and a dangling enclosure. Best used as a *development rig*: validate the GPU pipeline's output quality on CUDA before investing in the SYCL port.
- **Hailo-8 in the M.2 2242 slot**: electrically plausible, not ASUS-validated ("NVMe only" wording), 26 TOPS — but it only runs neural workloads, so it merely upgrades the NPU sidecar role (§3.2), it cannot touch the classical pipeline. Skip unless the NPU proposal net proves valuable and NPU-bound. **[EXTRAPOLATED]**
- **Isaac ROS / cuAprilTags is not a shortcut** **[VERIFIED]**: NVIDIA's GPU detector core is closed-source and its CUDA backend decodes only tag36h11 — it could not handle tagStandard52h13 even on an eGPU. Only the architecture transfers.

## 4 · Candidate architectures compared

All latency estimates for Arc 140T are extrapolations — *no GPU AprilTag implementation has ever been measured on Intel Arc*. Reference points: Isaac ROS measures 2.0 ms (RTX 5090), 2.4 ms (DGX Spark), 2.9 ms (AGX Thor T5000) per 720p tag36h11 frame — and 11 ms on the cheaper Thor T4000, so embedded-class GPUs don't uniformly hit 3 ms. Our frame has 2.4× the pixels of 720p; the Arc 140T has far less compute than any of those.

| # | Architecture | Est. latency | Effort | Risk | Verdict |
| --- | --- | --- | --- | --- | --- |
| A0 | **Current CPU pipeline on the NUC** (re-baselined, threads re-tuned for 6P+8E) | ~30–45 ms [EXTRAP] | Days | None | **Do first**, it's the denominator for everything |
| B1 | **iGPU threshold + CCL + cluster extraction; CPU fit-quads + decode** (port the front half of frc971) | ~15–22 ms [EXTRAP] | Weeks | Medium | **Good first GPU milestone**; retires unionfind + make-clusters (18.7 ms) |
| **B2** | **Full frc971 architecture on iGPU: everything through quad fitting on GPU, decode on CPU** (SYCL/Level Zero port, zero-copy USM, frames pipelined) | **~6–14 ms** [EXTRAP] | 1–3 months | Medium | **Recommended target.** Proven design, vendor-neutral CCL algorithm, keeps 52h13 decode, frees the CPU almost entirely |
| C | **Everything on iGPU incl. decode/refine** | ~5–10 ms [EXTRAP] | +1–2 months over B2 | High | Only if B2's CPU handoff proves to be the bottleneck — frc971 deliberately didn't bother |
| D | **NPU proposal net + classical pipeline on ROIs** (composable with A0/B1/B2) | scene-dependent; ~2–4× avg-fps in sparse scenes [EXTRAP] | Weeks (training + OpenVINO export) | Recall risk | Worth a prototype; never let it hard-filter until recall is proven |
| E | **TB4 eGPU + Team766 CUDA code** | ~3–6 ms [EXTRAP] | Days–weeks | Low (tech) / High (form factor) | Use as dev rig to de-risk B2's output quality; not the product |
| F | **Hailo-8 M.2 + learned detector** | n/a — NN-only roles | Weeks + unvalidated hardware | High | Skip; the NPU covers the sidecar role without slot games |
| G | **Full learned replacement (YoloTag/DeepTag-style) on iGPU XMX** | ~15–30 ms, no ID decode [EXTRAP] | Months (research-grade) | Very high | Doesn't decode IDs at family scale today; revisit if the field moves |

Latency comparison:

| Configuration | Latency |
| --- | --- |
| today (EPYC, 4T) | 39.8 ms |
| B1 partial iGPU | ~15–22 ms |
| B2 frc971-on-Arc | ~6–14 ms |
| Isaac ROS ceiling* | 2.0–2.9 ms |

*Isaac ROS: 720p, tag36h11, RTX 5090 / AGX Thor — shown as the architecture's demonstrated ceiling, not an Arc 140T prediction. **[VERIFIED]** (the numbers) / not comparable (the conditions).

## 5 · Recommended roadmap

1. **Phase 0 — Re-baseline on the NUC (days).** Build the `faster` branch on the NUC15CRK, re-run `benchmark.sh`/`ab.sh`, sweep worker count (4 / 6P / 6P+8E) and explicit P-core affinity. Decide everything else against *this* number, not 39.8 ms-on-EPYC.
2. **Phase 1 — De-risk on CUDA (1–2 weeks, optional but cheap).** Run Team766/apriltags_cuda on any NVIDIA box (or TB4 eGPU) against the 133-image corpus with the `check.sh` epsilon gate. This answers the two scary questions — output quality of the GPU front-end and real per-stage costs — before writing a line of SYCL. Note the license gap: Team766's repo has no LICENSE file; upstream frc971 is Apache-2.0, so port from upstream.
3. **Phase 2 — SYCL port, front half (3–6 weeks).** threshold → BKE CCL (from the YACCLAB reference, BSD, paper-documented) → boundary extraction with oneDPL compaction, zero-copy USM buffers, CPU does the rest. Milestone: B1, expect ~15–22 ms. The BKE port is the heart of it — budget accordingly.
4. **Phase 3 — Finish the front-end (3–6 weeks).** oneDPL radix sort-by-angle, line-fit prefix scans, FitQuads kernel, peak detection. Pipeline two frames deep (GPU on frame N+1 while CPU decodes N). Milestone: B2, target ≤10 ms latency and >150 fps throughput. Accept ε-drift via `check.sh`; byte-equivalence is formally retired.
5. **Phase 4 — The sidecars (parallel track, weeks).** NPU proposal net (OpenVINO, INT8, static ¼-res input) for frame gating and ROI mode; V4L2 `DMABUF` → Level Zero import for a memcpy-free camera path (unverified on Arrow Lake — prototype early). Each composes with B2 multiplicatively in real deployments.
6. **Think-big endgame:** B2 + frame pipelining + NPU gating on the 40 W NUC ≈ **200+ fps sustained full-frame detection with the CPU ~90% idle** — the CPU budget that used to be the whole detector becomes available for the actual application. **[EXTRAPOLATED]**

## 6 · Risks and unknowns

- **No Arc datapoint exists.** Every GPU AprilTag number in the wild is NVIDIA. The B1/B2 estimates are extrapolations from compute ratios and the Isaac ROS ceiling; Phase 2's first milestone is also the first real measurement. If kernel-launch overhead on Level Zero dominates at this small frame size, latency lands at the high end of the range (throughput via pipelining is more robust).
- **Output equivalence must be renegotiated.** GPU sorting/CCL cannot reproduce the legacy tie-order (PERF_NOTES invariant); expect ~0.02% detection deltas like the consistent-comparator CPU experiment showed. If a downstream consumer requires bit-stable output, that's a hard blocker for every GPU option.
- **tagStandard52h13 decode cost on crowded frames.** Decode stays on CPU at ~3.2 ms *for this corpus*; frames with many more candidate quads would raise the CPU floor. Mitigation: the decode stage already parallelizes per-tag.
- **Driver/stack maturity.** Arc compute on Linux (i915/xe + Level Zero) is solid in 2026 but version-sensitive; pin kernel + compute-runtime versions. NPU needs kernel ≥6.6 (`intel_vpu`) and OpenVINO 2025.x+ for Core Ultra 200H.
- **Unverified items** (research found no surviving claims): per-SKU 130T/140T mapping from Intel ARK; V4L2 DMABUF → Level Zero import on Arrow Lake-H specifically; published thresholding/CCL throughput on Intel iGPUs; Hailo-8 behavior in an "NVMe only" slot. The Wikipedia per-SKU core-count table was garbled in extraction — 255H = 6P+8E+2LP-E per Intel ARK product naming, but verify on the actual unit (`lscpu`) at Phase 0.
- **Refuted during verification** (claims that died 0-3, recorded so nobody re-imports them): "Isaac ROS AprilTag requires NVIDIA hardware / can't run without an eGPU" (its CPU backend runs anywhere; it's the GPU path that's NVIDIA-bound) and "NVIDIA ships a drop-in GPU replacement for the AprilTag library with selectable backends" (it's a ROS node wrapping closed-source cuAprilTags, not a library replacement).

## 7 · Sources

1. [ASUS NUC 15 Pro technical specifications](https://www.asus.com/displays-desktops/nucs/nuc-mini-pcs/asus-nuc-15-pro/techspec/) — CPU SKUs, memory, TB4, M.2 slots (primary)
2. [frc971/971-Robot-Code — frc971/orin GPU detector](https://github.com/frc971/971-Robot-Code/tree/main/frc971/orin) — per-stage CUDA pipeline, Apache-2.0 (primary; archived 2025-07)
3. [Team766/apriltags_cuda](https://github.com/Team766/apriltags_cuda) — standalone extraction of the frc971 detector (primary; no LICENSE file)
4. Allegretti, Bolelli, Grana — block-based GPU connected components (BKE), IEEE TPDS, DOI 10.1109/TPDS.2019.2934683, reference implementation in YACCLAB (BSD)
5. [Isaac ROS performance benchmarks](https://nvidia-isaac-ros.github.io/performance/index.html) + [isaac_ros_apriltag](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_apriltag) — 2.0–2.9 ms/720p figures; tag36h11-only CUDA backend; closed-source cuAprilTags (primary, vendor benchmark)
6. [Intel — zero-copy on processor graphics](https://www.intel.com/content/www/us/en/developer/articles/training/getting-the-most-from-opencl-12-how-to-increase-performance-by-minimizing-buffer-copies-on-intel-processor-graphics.html) (primary; OpenCL-era, reaffirmed by current oneAPI guides)
7. [OpenVINO NPU device documentation](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html) — FP16/INT8, static shapes (primary)
8. [YoloTag (RO-MAN 2024)](https://arxiv.org/abs/2409.02334), [Deep ChArUco (CVPR 2019)](https://arxiv.org/abs/1812.03247), [DeepTag (IEEE TVCG)](https://arxiv.org/abs/2105.13731) — learned-detector capability and compute bounds (peer-reviewed)
9. [Notebookcheck — Arc 140T analysis](https://www.notebookcheck.net/Intel-Arc-Graphics-140T-analysis-Arrow-Lake-H-iGPU-gains-frame-generation-support.960242.0.html), [Wikipedia — Arrow Lake](https://en.wikipedia.org/wiki/Arrow_Lake_(microprocessor)) — Xe-LPG+ w/ XMX, 77 TOPS, 13 TOPS NPU, no AVX-512/SMT (secondary)

---

*Methodology: deep-research workflow — 5 search angles, 22 sources fetched, 110 claims extracted, top 25 verified by 3-vote adversarial panels (23 confirmed 3-0, 2 refuted 0-3), 104 agents total — plus targeted spec verification against Notebookcheck/Wikipedia for gaps the panel left open (iGPU model, XMX, NPU TOPS, ISA). Stage costs and invariants from `PERF_NOTES.md` in this repository. Latency estimates for Intel Arc are explicitly extrapolated and labeled; no GPU AprilTag implementation has been measured on Intel hardware to date. Report generated 2026-06-10.*
