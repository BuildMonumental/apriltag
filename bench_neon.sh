#!/usr/bin/env bash
# macOS/arm64 benchmark for the NEON port: hyperfine wall-clock over the
# vide_images2 corpus plus per-stage detector timing, appended as one row
# to results.tsv.
#
# Usage:
#   ./bench_neon.sh <label> [description]
#   RUNS=10 WARMUP=3 BUILD_DIR=build-neon IMAGE_DIR=vide_images2 ./bench_neon.sh baseline "scalar"
set -euo pipefail
cd "$(dirname "$0")"

LABEL=${1:?usage: bench_neon.sh <label> [description]}
DESC=${2:-}

BUILD_DIR=${BUILD_DIR:-build-neon}
IMAGE_DIR=${IMAGE_DIR:-vide_images2}
OUT_DIR=${OUT_DIR:-benchmark_results}
WARMUP=${WARMUP:-3}
RUNS=${RUNS:-10}
RESULTS=${RESULTS:-results.tsv}

cmake --build "$BUILD_DIR" -j "$(sysctl -n hw.ncpu)" > /dev/null

# this machine is shared; numbers taken under load are garbage. Wait for a
# quiet window (1-min load average below threshold), up to MAX_WAIT seconds.
LOAD_MAX=${LOAD_MAX:-5.0}
MAX_WAIT=${MAX_WAIT:-900}
waited=0
while :; do
    load=$(sysctl -n vm.loadavg | awk '{print $2}')
    ok=$(awk -v l="$load" -v m="$LOAD_MAX" 'BEGIN { print (l < m) ? 1 : 0 }')
    [ "$ok" = 1 ] && break
    if [ "$waited" -ge "$MAX_WAIT" ]; then
        echo "warning: load still $load after ${MAX_WAIT}s; benchmarking anyway" >&2
        break
    fi
    echo "load $load >= $LOAD_MAX, waiting for a quiet window..." >&2
    sleep 30; waited=$((waited+30))
done

images=("$IMAGE_DIR"/*.jpg)
[ -e "${images[0]}" ] || { echo "no images in $IMAGE_DIR" >&2; exit 1; }

mkdir -p "$OUT_DIR"
rm -f dets.tsv timing.tsv

DEMO_FLAGS=(-t 4 -i 1 -x 1.0 -f tagStandard52h13
            --save-detections dets.tsv --save-timing timing.tsv)

hyperfine \
    --shell=none \
    --warmup "$WARMUP" \
    --runs "$RUNS" \
    --command-name "apriltag_demo@$LABEL" \
    --export-json "$OUT_DIR/hyperfine-$LABEL.json" \
    "$BUILD_DIR/apriltag_demo ${DEMO_FLAGS[*]} ${images[*]}"

cp dets.tsv "$OUT_DIR/dets-$LABEL.tsv"
cp timing.tsv "$OUT_DIR/timing-$LABEL.tsv"

commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
git diff --quiet -- '*.c' '*.h' 2>/dev/null || commit="$commit-dirty"

wall_mean=$(python3 -c "import json;r=json.load(open('$OUT_DIR/hyperfine-$LABEL.json'))['results'][0];print(f\"{r['mean']*1000:.1f}\")")
wall_stddev=$(python3 -c "import json;r=json.load(open('$OUT_DIR/hyperfine-$LABEL.json'))['results'][0];print(f\"{r['stddev']*1000:.1f}\")")

# per-stage means (ms/image) in stage order, from the last run's timing.tsv
if [ ! -f "$RESULTS" ]; then
    printf 'commit\tlabel\tthreshold_ms\tunionfind_ms\tmake_clusters_ms\tfit_quads_ms\tquads_ms\tdecode_refine_ms\tother_ms\ttotal_detector_ms\twall_mean_ms\twall_stddev_ms\tdescription\n' > "$RESULTS"
fi

awk -F'\t' -v commit="$commit" -v label="$LABEL" -v wm="$wall_mean" -v ws="$wall_stddev" -v desc="$DESC" '
NR > 1 {
    sum[$4] += $5
    if (!($1 in img)) { img[$1] = 1; nimg++ }
}
END {
    main["threshold"]=1; main["unionfind"]=1; main["make clusters"]=1
    main["fit quads to clusters"]=1; main["quads"]=1; main["decode+refinement"]=1
    other = 0; total = 0
    for (s in sum) {
        total += sum[s]
        if (!(s in main)) other += sum[s]
    }
    printf "%s\t%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%s\t%s\t%s\n",
        commit, label,
        sum["threshold"]/nimg, sum["unionfind"]/nimg, sum["make clusters"]/nimg,
        sum["fit quads to clusters"]/nimg, sum["quads"]/nimg, sum["decode+refinement"]/nimg,
        other/nimg, total/nimg, wm, ws, desc
}' timing.tsv >> "$RESULTS"

echo
echo "per-stage detector timing (mean ms/image, last run):"
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

tail -1 "$RESULTS"
