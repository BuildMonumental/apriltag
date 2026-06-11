# Fleet campaign notes (2026-06-11)

Live validation and benchmarking across four robots: w3cj (NUC15,
225H, Arc 140T), w3y / w3bd / w3u (NUC14, 125H, Xe-LPG). Methods and
scripts in this directory; raw frames preserved off-robot
(~/Documents/apriltag-realframes, ~/Documents/apriltag-bursts).

## Tier results (quiet machines, real full-res robot frames)

per-frame, prod settings (52h13, 4 threads, decimate 1.0), bit-exact
parity verified in every cell:

NUC15 / 225H (broken balance_performance EPP, the fleet default):
  stock 88-108 ms / 238-260 core-ms; midway 41-46 / 90-112; fit 34-35 / 23-24
NUC15 / 225H (EPP=performance, sustained 60-300 s, tier_burn):
  stock 94 ms / 304 core-ms, 105 C, clock 4850->2199 (throttled, wall ~flat);
  midway 60 / 167, 92 C; fit 50 / 42, 68 C
NUC14 / 125H (healthy units w3y/w3bd):
  stock 65-112 / 206-364; midway 35-58 / 70-168; fit 32-50 / 26-40

In-service (vide, dark-evening conditions): stock ~1.9-3.5 cores ->
fit tier 0.48-1.35 cores. Combo with the spectacular-vio optimisation
deploy on w3y: system 3.95 -> 2.58 cores.

## Findings

- "Dense-frame cliff" RETRACTED: the slow unit (w3u) runs ALL frames
  ~10x slow on GPU (cross-tested both directions); content exonerated
  (boundary-record and component-structure analysis showed the "dense"
  frame is unexceptional). w3u GPU defect: full clocks/topology/
  bandwidth reported healthy, survives reboot. Hardware case.
  => rollout needs a startup self-bench gate, not a content breaker.
- GPU->CPU power coupling: NOT material on fleet hardware. Cool quiet
  boosted 225H: busy core 4900 MHz alone, 4700 beside CPU detect, 4688
  beside GPU detect (-0.3% marginal). Package 38.5 W peak vs 64 W PL2.
  GPU detect draws LESS package power than CPU detect (32 vs 38.5 W)
  and runs ~35 C cooler sustained (68 vs 105 C).
- FLEET BUG (ticket-worthy): NUC15/225H + kernel 6.12 intel_pstate
  balance_performance clamps busy P-cores to ~1.7 GHz (below base;
  hwmax misreported as 6.2 GHz). EPP=performance restores spec 4.9 GHz
  turbo. NUC14 unaffected. All NUC15 CPU benchmarks under the default
  EPP are ~2x pessimistic.
- Cameras deliver 7-19.5 fps (exposure-limited), not nominal 25; dark
  scenes throttle vide's loop via frame-freshness rejection.
- vide --mock is broken upstream (FilesystemCameraReader lacks the
  auto-exposure interface the camera-switch path calls).
- Parallel work: BuildMonumental/apriltag branch nuc15-hardware
  (Bouke, same day) — independent bit-exact GPU port, run-based
  frontend, CPU-fit hybrid default, detect_prepare overlap API.
  Complementary: their CPU stages + overlap API, this repo's fit
  chain + drop-in deployment + fleet validation.
