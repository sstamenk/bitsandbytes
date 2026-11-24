#!/bin/bash
# q_cap.sh <dtypes>: fused (default dispatch, build of TREE) vs dequant + F.linear, in-process interleaved, around the
# Python M limit. fp16/bf16: M = 384-512, 5 cycles, and blocksize 32 at M = 384 and 512, 3 cycles; fp32: M = 384 and
# 512, 3 cycles.
D=~/bnbtune/data/cap; mkdir -p $D
B=$HOME/bnbtune/bin; SH=$(seq -s, 1 39),0
run() { # run <tag> <dtype> <ms> <cycles> [blocksize]
  [ -s $D/$1.csv ] && return
  echo "$(date -u +%T) $1"; python $B/alt1103b.py --configs F=auto,Q=dequant --dtype $2 --ms $3 --shapes $SH --cycles $4 --blocksize ${5:-64} --out $D/$1.csv.tmp > $D/$1.log 2>&1 && mv $D/$1.csv.tmp $D/$1.csv
}
for dt in ${1//,/ }; do
  if [ $dt = fp32 ]; then run cap_fp32 fp32 384,512 3; else run cap_$dt $dt 384,416,448,480,512 5; run cap_bs32_$dt $dt 384,512 3 32; fi
done
echo "$(date -u +%T) done"
