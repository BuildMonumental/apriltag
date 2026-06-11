#!/usr/bin/env bash
# Compare two dets.tsv files: detection retention and corner accuracy.
#   ./compare_dets.sh <reference.tsv> <candidate.tsv>
# Reference is treated as ground truth (typically the undecimated run).
set -euo pipefail

awk -F'\t' '
NR == FNR {
    if (FNR == 1) next
    key = $1 SUBSEP $3
    ref[key] = $0
    nref++
    next
}
FNR == 1 { next }
{
    key = $1 SUBSEP $3
    ncand++
    if (!(key in ref)) { extra++; next }
    split(ref[key], r, "\t")
    matched++
    if ($4 != r[4]) hamming_diff++
    # center error
    dx = $6 - r[6]; dy = $7 - r[7]
    cerr = sqrt(dx*dx + dy*dy)
    csum += cerr; if (cerr > cmax) cmax = cerr
    # mean corner error over 4 corners
    for (c = 0; c < 4; c++) {
        dx = $(8+2*c) - r[8+2*c]; dy = $(9+2*c) - r[9+2*c]
        e = sqrt(dx*dx + dy*dy)
        esum += e; n_e++
        if (e > emax) emax = e
        if (e > 1.0) over1++
        if (e > 0.5) over05++
    }
    delete ref[key]
}
END {
    missing = nref - matched
    printf "reference: %d  candidate: %d  matched: %d\n", nref, ncand, matched
    printf "missing: %d (%.2f%%)  extra: %d  hamming-diff: %d\n", missing, 100.0*missing/nref, extra, hamming_diff
    if (matched > 0) {
        printf "center err:  mean %.4f px  max %.4f px\n", csum/matched, cmax
        printf "corner err:  mean %.4f px  max %.4f px  >0.5px: %d/%d (%.2f%%)  >1px: %d (%.2f%%)\n",
            esum/n_e, emax, over05, n_e, 100.0*over05/n_e, over1, 100.0*over1/n_e
    }
}' "$1" "$2"
