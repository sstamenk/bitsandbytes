#!/bin/bash
# extra tile sweeps (dev build in src/): bf16 at key M, fp16 at the remaining bucket-edge M; runs inside job.sh
D=~/bnbtune/data/sweep; mkdir -p $D
ALL=$(seq -s, 1 39),0
export BNB_GEMM4BIT_KERNEL=wmma
run() { # run <tag> <dtype> <ms>
  [ -e $D/sweep_$1.csv.done ] && return
  echo "$(date -u +%T) $1"; python ~/bnbtune/bin/sweep1103.py --dtype $2 --shapes $ALL --ms $3 --cool ${COOL:-3} --out $D/sweep_$1.csv > $D/sweep_$1.log 2>&1 && touch $D/sweep_$1.csv.done
}
run fp16c fp16 56,80,112,160,512
run bf16a bf16 1,4,8,16,32,64,128,256
echo "$(date -u +%T) done"
