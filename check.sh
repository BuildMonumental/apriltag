#!/usr/bin/env bash
# Fast verify loop for optimization work: incremental build, one full-corpus
# run, dets compared against the baseline within epsilon, per-stage timing
# next to the baseline's. Use ./benchmark.sh for the statistically solid
# numbers; this is the quick inner-loop check.
set -euo pipefail
cd "$(dirname "$0")"

BUILD_DIR=${BUILD_DIR:-build}
BASE=${BASE:-benchmark_results/dets-baseline.tsv}
BASE_TIMING=${BASE_TIMING:-benchmark_results/timing-baseline.tsv}

cmake --build "$BUILD_DIR" -j "$(nproc)" > /dev/null

IMAGE_DIR=${IMAGE_DIR:-vide_images/vide_images2}
THREADS=${THREADS:-4}
CPUS=${CPUS:-0-3}

export LD_LIBRARY_PATH="$BUILD_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
taskset -c "$CPUS" "$BUILD_DIR/apriltag_demo" -t "$THREADS" -i 1 -x 1.0 -f tagStandard52h13 \
    --save-detections /tmp/dets_new.tsv --save-timing /tmp/timing_new.tsv \
    "$IMAGE_DIR"/*.jpg > /dev/null

# detections: (image,id) sets must match exactly; hamming exact;
# corners/center within 0.1px; margin within 1.0
awk -F'\t' '
NR == FNR {
    if (FNR == 1) next
    key = $1 SUBSEP $3
    if (key in base) { base_dup[key]++ }
    base[key] = $0
    nbase++
    next
}
FNR == 1 { next }
{
    key = $1 SUBSEP $3
    nnew++
    if (!(key in base)) { print "NEW detection not in baseline: " $1 " id " $3; bad=1; next }
    split(base[key], b, "\t")
    if ($4 != b[4]) { print "hamming differs: " $1 " id " $3 ": " b[4] " -> " $4; bad=1 }
    if ((($5 - b[5]) > 1.0) || ((b[5] - $5) > 1.0)) { print "margin differs: " $1 " id " $3 ": " b[5] " -> " $5; bad=1 }
    for (i = 6; i <= 15; i++) {
        d = $i - b[i]; if (d < 0) d = -d
        if (d > maxd) maxd = d
        if (d > 0.1) { print "coord field " i " differs by " d ": " $1 " id " $3; bad=1 }
    }
    delete base[key]
}
END {
    for (key in base) { split(key, k, SUBSEP); print "MISSING detection: " k[1] " id " k[2]; bad=1 }
    printf "detections: %d baseline vs %d new, max coord delta %.5f px\n", nbase, nnew, maxd
    exit bad
}' "$BASE" /tmp/dets_new.tsv

# per-stage timing: new vs baseline means
awk -F'\t' '
NR == FNR { if (FNR > 1) { bsum[$4] += $5; bn[$4]++ } next }
FNR > 1 {
    key = sprintf("%02d %s", $3, $4)
    sum[key] += $5; n[key]++
    if (!(key in seen)) { order[++stages] = key; seen[key] = 1 }
}
END {
    printf "%-36s %10s %10s %8s\n", "stage", "baseline", "new", "delta"
    total = 0; btotal = 0
    for (i = 1; i <= stages; i++) {
        k = order[i]; name = substr(k, 4)
        mean = sum[k] / n[k]; bmean = (name in bsum) ? bsum[name] / bn[name] : 0
        total += mean; btotal += bmean
        if (bmean < 0.05 && mean < 0.05) continue
        printf "%-36s %9.3f  %9.3f  %+7.3f\n", name, bmean, mean, mean - bmean
    }
    printf "%-36s %9.3f  %9.3f  %+7.3f ms/image\n", "TOTAL (detector)", btotal, total, total - btotal
}' "$BASE_TIMING" /tmp/timing_new.tsv
