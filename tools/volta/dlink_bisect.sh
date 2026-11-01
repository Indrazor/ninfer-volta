#!/usr/bin/env bash
# Bisect nvlink "uses too much shared data" carriers under sm_70 whole-program
# device link. Run inside ninfer-build:cu129 with /src = build root parent.
# Usage: dlink_bisect.sh [target-lib]  (default lib/libninfer_ops.so)
set -u
cd "$(dirname "$0")/../../build/v100-release"
TARGET="${1:-lib/libninfer_ops.so}"
cmd=$(ninja -t commands "$TARGET" | grep -- "-dlink" | tail -1)
eval "args=($cmd)"
fixed=(); objs=()
skip=0; i=0
for a in "${args[@]}"; do
    i=$((i + 1)); [ $i -eq 1 ] && continue           # drop compiler path
    if [ $skip -eq 1 ]; then skip=0; continue; fi    # drop -o output
    if [ "$a" = "-o" ]; then skip=1; continue; fi
    case "$a" in
        *.o|*.a) objs+=("$a") ;;
        *)       fixed+=("$a") ;;
    esac
done
mand=(); ar=(); rest=()
for o in "${objs[@]}"; do
    case "$o" in
        # entries that nvlink reported as the poisoned ones:
        *vocabulary_logprobs*|*target_logprobs*|*block_scaled/launch.cu.o|*block_scaled/shared.cu.o)
            mand+=("$o") ;;
        *.a) ar+=("$o") ;;
        *)   rest+=("$o") ;;
    esac
done
echo "MAND=${#mand[@]} AR=${#ar[@]} REST=${#rest[@]}"
nvcheck() {
    nvcc "${fixed[@]}" "${mand[@]}" "${ar[@]}" "$@" -o /tmp/t.cubin >/tmp/nv.log 2>&1
    grep -q "too much shared" /tmp/nv.log
}
if ! nvcheck "${rest[@]}"; then
    echo "baseline does not reproduce; check mand list"
    exit 1
fi
cur=("${rest[@]}")
while [ ${#cur[@]} -gt 1 ]; do
    half=$(( ${#cur[@]} / 2 ))
    if nvcheck "${cur[@]:0:$half}"; then cur=("${cur[@]:0:$half}"); else cur=("${cur[@]:$half}"); fi
    echo "shrink -> ${#cur[@]}"
done
echo "CULPRIT: ${cur[0]}"
