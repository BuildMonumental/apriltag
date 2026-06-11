#!/usr/bin/env bash
# Per-commit benchmark for the NUC15 hardware-acceleration campaign.
# Runs hyperfine over the vide_images2 corpus, extracts per-stage detector
# timing, and appends a row to results.tsv.
#
# Usage: ./bench.sh "description of the change"
set -euo pipefail
cd "$(dirname "$0")"

BUILD_DIR=${BUILD_DIR:-build}
IMAGE_DIR=${IMAGE_DIR:-vide_images/vide_images2}
THREADS=${THREADS:-4}
CPUS=${CPUS:-0-3}
WARMUP=${WARMUP:-2}
RUNS=${RUNS:-10}
RESULTS=${RESULTS:-results.tsv}
DESC=${1:?usage: bench.sh "description"}

cmake --build "$BUILD_DIR" -j "$(nproc)" > /dev/null

images=("$IMAGE_DIR"/*.jpg)
rev=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
git diff --quiet 2>/dev/null || rev="$rev-dirty"

export LD_LIBRARY_PATH="$BUILD_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
DEMO_FLAGS=(-t "$THREADS" -i 1 -x 1.0 -f tagStandard52h13
            --save-detections /tmp/bench_dets.tsv --save-timing /tmp/bench_timing.tsv)

mkdir -p benchmark_results
hyperfine \
    --shell=none \
    --warmup "$WARMUP" \
    --runs "$RUNS" \
    --command-name "apriltag@$rev" \
    --export-json "benchmark_results/hyperfine-$rev.json" \
    "taskset -c $CPUS $BUILD_DIR/apriltag_demo ${DEMO_FLAGS[*]} ${images[*]}"

cp /tmp/bench_dets.tsv "benchmark_results/dets-$rev.tsv"
cp /tmp/bench_timing.tsv "benchmark_results/timing-$rev.tsv"

wall_ms=$(jq -r '.results[0].mean * 1000 | floor' "benchmark_results/hyperfine-$rev.json")

if [ ! -f "$RESULTS" ]; then
    printf 'commit\tthreshold\tunionfind\tmake_clusters\tfit_quads\tdecode_refine\tother\tdetector_total\twall_ms_corpus\tdescription\n' > "$RESULTS"
fi

awk -F'\t' -v rev="$rev" -v wall="$wall_ms" -v desc="$DESC" '
NR > 1 { sum[$4] += $5; n[$4]++; img[$1]=1 }
END {
    nimg = 0; for (i in img) nimg++
    split("threshold unionfind", _, " ")
    th = sum["threshold"]/nimg
    uf = sum["unionfind"]/nimg
    mc = sum["make clusters"]/nimg
    fq = sum["fit quads to clusters"]/nimg
    dr = sum["decode+refinement"]/nimg
    total = 0; for (k in sum) total += sum[k]/nimg
    other = total - th - uf - mc - fq - dr
    printf "%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%s\t%s\n", \
        rev, th, uf, mc, fq, dr, other, total, wall, desc
}' /tmp/bench_timing.tsv >> "$RESULTS"

echo
tail -1 "$RESULTS" | tr '\t' '\n' | paste <(head -1 "$RESULTS" | tr '\t' '\n') - | column -t
