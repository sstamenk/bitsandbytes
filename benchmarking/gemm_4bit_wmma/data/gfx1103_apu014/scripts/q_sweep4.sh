#!/bin/bash
# extra tile sweeps (dev build in src/): fp16 tiles with under 32 rows at M = 80, bf16 at M = 80 and 96; runs inside job.sh
D=~/bnbtune/data/sweep; mkdir -p $D
ALL=$(seq -s, 1 39),0
export BNB_GEMM4BIT_KERNEL=wmma
run() { # run <tag> <dtype> <ms> [extra sweep args]
  [ -e $D/sweep_$1.csv.done ] && return
  local tag=$1 dt=$2 ms=$3; shift 3
  echo "$(date -u +%T) $tag"; python ~/bnbtune/bin/sweep1103.py --dtype $dt --shapes $ALL --ms $ms --cool ${COOL:-3} "$@" --out $D/sweep_$tag.csv > $D/sweep_$tag.log 2>&1 && touch $D/sweep_$tag.csv.done
}
run fp16e fp16 80 --minmt 0 --mtmax 16
run bf16b bf16 80,96
echo "$(date -u +%T) done"
