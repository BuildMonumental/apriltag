#!/usr/bin/env bash
# Interleaved A/B of two apriltag_demo builds. Runs alternate A,B,A,B,...
# so machine-state noise (CI load, frequency scaling) biases both sides
# equally; the detector-total ratio is what to trust.
#
# usage: ./ab.sh <build_dir_A> <build_dir_B> [rounds]
set -euo pipefail
cd "$(dirname "$0")"

A=${1:?usage: ab.sh <build_dir_A> <build_dir_B> [rounds]}
B=${2:?}
ROUNDS=${3:-8}

ARGS=(-t 4 -i 1 -x 1.0 -f tagStandard52h13
      --save-detections /tmp/ab_dets.tsv --save-timing /tmp/ab_timing.tsv)
imgs=(vide_images/*.jpg)

run_one() { # <build_dir> -> prints detector ms/image
    LD_LIBRARY_PATH=$1 "$1/apriltag_demo" "${ARGS[@]}" "${imgs[@]}" > /dev/null
    awk -F'\t' 'NR>1 { s += $5; img[$1]=1 } END { c=0; for (i in img) c++; printf "%.3f\n", s/c }' /tmp/ab_timing.tsv
}

# warmup
run_one "$A" > /dev/null
run_one "$B" > /dev/null

a_runs=()
b_runs=()
for ((r = 0; r < ROUNDS; r++)); do
    a=$(run_one "$A")
    b=$(run_one "$B")
    a_runs+=("$a")
    b_runs+=("$b")
    echo "round $((r+1)): A=$a B=$b ms/image"
done

stats() { printf '%s\n' "$@" | sort -n | awk '{v[NR]=$1; s+=$1} END {printf "mean %.3f median %.3f min %.3f", s/NR, v[int((NR+1)/2)], v[1]}'; }
echo
echo "A ($A): $(stats "${a_runs[@]}")"
echo "B ($B): $(stats "${b_runs[@]}")"
awk -v a="$(printf '%s\n' "${a_runs[@]}" | sort -n | awk '{v[NR]=$1} END {print v[int((NR+1)/2)]}')" \
    -v b="$(printf '%s\n' "${b_runs[@]}" | sort -n | awk '{v[NR]=$1} END {print v[int((NR+1)/2)]}')" \
    'BEGIN { printf "speedup (A median / B median): %.3fx\n", a/b }'
