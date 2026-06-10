#!/usr/bin/env bash
# Benchmark apriltag_demo with hyperfine over the vide_images corpus.
#
# Usage:
#   ./benchmark.sh                 # incremental build + benchmark current tree
#   RUNS=20 WARMUP=5 ./benchmark.sh
#   LABEL=baseline ./benchmark.sh  # tag results with a custom label
#
# Results land in benchmark_results/:
#   hyperfine-<label>.json/.md  - wall-clock stats across runs
#   dets-<label>.tsv            - detections from the last run (diff against
#                                 another build to verify identical output)
#   timing-<label>.tsv          - per-stage detector timing from the last run
set -euo pipefail

cd "$(dirname "$0")"

BUILD_DIR=${BUILD_DIR:-build}
IMAGE_DIR=${IMAGE_DIR:-vide_images}
OUT_DIR=${OUT_DIR:-benchmark_results}
WARMUP=${WARMUP:-3}
RUNS=${RUNS:-10}

DEMO_FLAGS=(-t 4 -i 1 -x 1.0 -f tagStandard52h13
            --save-detections dets.tsv --save-timing timing.tsv)

if [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
    cmake -S . -B "$BUILD_DIR" -DBUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_FLAGS="-march=native" > /dev/null
fi
# always build: a stale binary would be measured under the current rev's label
cmake --build "$BUILD_DIR" -j "$(nproc)" > /dev/null

images=("$IMAGE_DIR"/*.jpg)
if [ ! -e "${images[0]}" ]; then
    echo "no images found in $IMAGE_DIR" >&2
    exit 1
fi

rev=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
git diff --quiet 2>/dev/null || rev="$rev-dirty"
LABEL=${LABEL:-$rev}

mkdir -p "$OUT_DIR"

governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)
echo "cpu cores: $(nproc), governor: $governor"
if [ "$governor" != "performance" ] && [ "$governor" != "unknown" ]; then
    echo "note: cpu governor is '$governor'; 'performance' gives more stable numbers"
fi
echo "benchmarking ${#images[@]} images, $RUNS runs (+$WARMUP warmup), label: $LABEL"
echo

export LD_LIBRARY_PATH="$BUILD_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# remove leftovers so a failed run can't pass off stale artifacts as fresh
rm -f dets.tsv timing.tsv

hyperfine \
    --shell=none \
    --warmup "$WARMUP" \
    --runs "$RUNS" \
    --command-name "apriltag_demo@$LABEL" \
    --export-json "$OUT_DIR/hyperfine-$LABEL.json" \
    --export-markdown "$OUT_DIR/hyperfine-$LABEL.md" \
    --setup "true" \
    "taskset -c 0-3 $BUILD_DIR/apriltag_demo ${DEMO_FLAGS[*]} ${images[*]}"

cp dets.tsv "$OUT_DIR/dets-$LABEL.tsv"
cp timing.tsv "$OUT_DIR/timing-$LABEL.tsv"

echo
echo "per-stage detector timing (mean ms/image over ${#images[@]} images, last run):"
awk -F'\t' 'NR > 1 {
    key = sprintf("%02d %s", $3, $4)
    sum[key] += $5
    if (!(key in seen)) { order[++stages] = key; seen[key] = 1 }
    n[key]++
} END {
    total = 0
    for (i = 1; i <= stages; i++) {
        k = order[i]
        mean = sum[k] / n[k]
        total += mean
        printf "  %-36s %9.3f ms\n", substr(k, 4), mean
    }
    printf "  %-36s %9.3f ms\n", "total (detector only)", total
}' timing.tsv

echo
echo "results saved to $OUT_DIR/ (label: $LABEL)"
echo "compare builds with: diff $OUT_DIR/dets-<a>.tsv $OUT_DIR/dets-<b>.tsv"
