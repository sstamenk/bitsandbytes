#!/bin/bash
# q_altB.sh: final dev build (TREE=fin1), other blocksizes, in-process interleaved, 3 cycles: A = the base dispatch
# through overrides (gfx1100 tables; fp32 with the gfx1103 cutoffs), F = default dispatch. fp16 at blocksize 32 and
# 128, fp32 at blocksize 32.
D=~/bnbtune/data/altF; mkdir -p $D
B=$HOME/bnbtune/bin; SH=$(seq -s, 1 39),0
CUT=2/1073741824+4/3840
run() { # run <tag> <dtype> <ms> <blocksize> <configs>
  [ -s $D/$1.csv ] && return
  echo "$(date -u +%T) $1"; python $B/alt1103b.py --configs $5 --dtype $2 --ms $3 --shapes $SH --cycles 3 --blocksize $4 --out $D/$1.csv.tmp > $D/$1.log 2>&1 && mv $D/$1.csv.tmp $D/$1.csv
}
run bs32_fp16 fp16 1,8,16,32,48,96 32 A=table:g1100_f16,F=auto
run bs128_fp16 fp16 1,8,16,32,48,96 128 A=table:g1100_f16,F=auto
run bs32_fp32 fp32 8,16,48,128 32 A=cut:$CUT:g1100_f32,F=auto
echo "$(date -u +%T) done"
