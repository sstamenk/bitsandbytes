#!/bin/bash
# fp32 tile sweeps at the remaining M, and the tiles with under 32 rows at M >= 64 (dev build in src/); runs inside job.sh
D=~/bnbtune/data/sweep; mkdir -p $D
ALL=$(seq -s, 1 39),0
export BNB_GEMM4BIT_KERNEL=wmma
run() { # run <tag> <dtype> <ms> [extra sweep args]
  [ -e $D/sweep_$1.csv.done ] && return
  local tag=$1 dt=$2 ms=$3; shift 3
  echo "$(date -u +%T) $tag"; python ~/bnbtune/bin/sweep1103.py --dtype $dt --shapes $ALL --ms $ms --cool ${COOL:-3} "$@" --out $D/sweep_$tag.csv > $D/sweep_$tag.log 2>&1 && touch $D/sweep_$tag.csv.done
}
run fp32c fp32 64,96,128,192,256,384 --minmt 0
run fp32b fp32 6,12,24,48
run fp16d fp16 64,96,128,192,256,384,512 --minmt 0 --mtmax 16
echo "$(date -u +%T) done"
