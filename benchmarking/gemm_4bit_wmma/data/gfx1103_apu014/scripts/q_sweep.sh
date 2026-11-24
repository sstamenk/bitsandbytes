#!/bin/bash
# tile sweeps (dev build in src/), forced WMMA, every candidate tile; runs inside job.sh. Key M first, then the rest.
D=~/bnbtune/data/sweep; mkdir -p $D
ALL=$(seq -s, 1 39),0
export BNB_GEMM4BIT_KERNEL=wmma
run() { # run <tag> <dtype> <ms>
  [ -e $D/sweep_$1.csv.done ] && return
  echo "$(date -u +%T) $1"; python ~/bnbtune/bin/sweep1103.py --dtype $2 --shapes $ALL --ms $3 --cool ${COOL:-3} --out $D/sweep_$1.csv > $D/sweep_$1.log 2>&1 && touch $D/sweep_$1.csv.done
}
run fp16a fp16 1,4,8,16,32,64,128,256
run fp16b fp16 2,12,20,24,28,40,48,96,192,384
run fp32a fp32 3,4,8,16,32,64,128,256
run fp32b fp32 6,12,24,48,96,192,384
echo "$(date -u +%T) done"
