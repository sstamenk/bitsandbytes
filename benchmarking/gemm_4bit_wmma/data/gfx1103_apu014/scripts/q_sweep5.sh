#!/bin/bash
# fp32 tile sweep at M = 64, 128, 256 with every tile (also under 32 rows); dev build in src/; runs inside job.sh.
# Replaces sweep3's fp32c (M 64,96,128,192,256,384), stopped after its first shape (kept as sweep_fp32c_part.csv).
D=~/bnbtune/data/sweep; mkdir -p $D
ALL=$(seq -s, 1 39),0
export BNB_GEMM4BIT_KERNEL=wmma
run() { # run <tag> <dtype> <ms> [extra sweep args]
  [ -e $D/sweep_$1.csv.done ] && return
  local tag=$1 dt=$2 ms=$3; shift 3
  echo "$(date -u +%T) $tag"; python ~/bnbtune/bin/sweep1103.py --dtype $dt --shapes $ALL --ms $ms --cool ${COOL:-3} "$@" --out $D/sweep_$tag.csv > $D/sweep_$tag.log 2>&1 && touch $D/sweep_$tag.csv.done
}
run fp32d fp32 64,128,256 --minmt 0
echo "$(date -u +%T) done"
