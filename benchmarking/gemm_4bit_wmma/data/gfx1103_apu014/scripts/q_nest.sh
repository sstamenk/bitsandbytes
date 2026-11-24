#!/bin/bash
# q_nest.sh: double-quantized absmax (compress_statistics=True), blocksize 64, base dev build (TREE=src), in-process
# interleaved, 3 cycles: A = gfx1100 table, S = cand/S1.py, V3 = cand/V3.py (bf16/fp16); fp32: A = gfx1103 cutoffs +
# gfx1100 tiles, F = the same cutoffs + cand/F1.py.
D=~/bnbtune/data/altV; mkdir -p $D
B=$HOME/bnbtune/bin; SH=$(seq -s, 1 39),0
CUT=2/1073741824+4/3840
run() { # run <tag> <dtype> <ms> <configs>
  [ -s $D/$1.csv ] && return
  echo "$(date -u +%T) $1"; python $B/alt1103c.py --nested --configs $4 --dtype $2 --ms $3 --shapes $SH --cycles 3 --out $D/$1.csv.tmp > $D/$1.log 2>&1 && mv $D/$1.csv.tmp $D/$1.csv
}
T16=A=table:g1100_f16,S=table:$B/cand/S1.py,V3=table:$B/cand/V3.py
run nest_fp16 fp16 1,8,16,24,32,48,64,96,128 $T16
run nest_bf16 bf16 1,8,16,24,32,48,64,96 $T16
run nest_fp32 fp32 4,8,16,48,80,128 A=cut:$CUT:g1100_f32,F=cut:$CUT:$B/cand/F1.py
echo "$(date -u +%T) done"
