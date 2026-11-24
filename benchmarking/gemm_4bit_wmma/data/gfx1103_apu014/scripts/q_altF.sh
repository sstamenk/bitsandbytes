#!/bin/bash
# q_altF.sh <dtypes>: final dev build (TREE=fin1), in-process interleaved: A = the base default dispatch emulated with
# overrides (gfx1100 tile tables; fp32 with the gfx1103 cutoffs), F = default dispatch of the final build (no
# override), X = the final tables through the same overrides as A (must match F), D = the gfx1150 fp32 table, S = forced
# SIMT (bf16/fp16 at M = 1-2). 5 cycles; fp32 M >= 256 and the X / D runs 3 cycles.
D=~/bnbtune/data/altF; mkdir -p $D
B=$HOME/bnbtune/bin; SH=$(seq -s, 1 39),0
CUT=2/1073741824+4/3840
run() { # run <tag> <dtype> <ms> <cycles> <configs>
  [ -s $D/$1.csv ] && return
  echo "$(date -u +%T) $1"; python $B/alt1103b.py --configs $5 --dtype $2 --ms $3 --shapes $SH --cycles $4 --out $D/$1.csv.tmp > $D/$1.log 2>&1 && mv $D/$1.csv.tmp $D/$1.csv
}
for dt in ${1//,/ }; do
  if [ $dt = fp32 ]; then
    run fp32_lo fp32 1,2,3,4,5,6,8,9,12,16,17,24,32,33,48,49,64,72,80,96,112,128,160 5 A=cut:$CUT:g1100_f32,F=auto
    run fp32_hi fp32 256,384,512 3 A=cut:$CUT:g1100_f32,F=auto
    run fp32_x fp32 4,8,16,48,80 3 F=auto,X=cut:$CUT:$B/cand/F1.py
    run fp32_d fp32 5,8,16,48,64,128 3 F=auto,D=cut:$CUT:g1150_f32
  else
    run $dt $dt 1,2,3,4,6,8,9,12,16,17,20,24,28,32,33,40,48,49,56,64,65,72,80,88,96,97,128,192,256,384,512 5 A=table:g1100_f16,F=auto
    run ${dt}_x $dt 1,8,16,24,32,48,64,96 3 F=auto,X=table:$B/cand/S1.py
    run ${dt}_m1 $dt 1,2 5 F=auto,S=simt
  fi
done
echo "$(date -u +%T) done"
