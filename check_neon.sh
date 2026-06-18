#!/usr/bin/env bash
# Fast correctness loop for the NEON port on macOS: incremental build, one
# full vide_images2 run, detections compared against the scalar arm64
# baseline within epsilon (same gates as check.sh).
set -euo pipefail
cd "$(dirname "$0")"

BUILD_DIR=${BUILD_DIR:-build-neon}
IMAGE_DIR=${IMAGE_DIR:-vide_images2}
BASE=${BASE:-benchmark_results/dets-baseline.tsv}

cmake --build "$BUILD_DIR" -j "$(sysctl -n hw.ncpu)" > /dev/null

"$BUILD_DIR/apriltag_demo" -t 4 -i 1 -x 1.0 -f tagStandard52h13 \
    --save-detections /tmp/dets_new.tsv --save-timing /tmp/timing_new.tsv \
    "$IMAGE_DIR"/*.jpg > /dev/null

# detections: (image,id) sets must match exactly; hamming exact;
# corners/center within 0.1px; margin within 1.0
awk -F'\t' '
NR == FNR {
    if (FNR == 1) next
    key = $1 SUBSEP $3
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
