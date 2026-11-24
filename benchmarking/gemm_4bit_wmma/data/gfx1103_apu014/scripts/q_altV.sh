#!/bin/bash
# q_altV.sh: the M = 17-32 bucket at blocksize 32 and 64 (base dev build, TREE=src), in-process interleaved, 5 cycles:
# A = gfx1100 table, S = cand/S1.py, V2 = cand/V2.py, V3 = cand/V3.py, B = rulesB2.py; and bf16 at blocksize 32 over
# the other buckets (A against S, 3 cycles).
D=~/bnbtune/data/altV; mkdir -p $D
B=$HOME/bnbtune/bin; SH=$(seq -s, 1 39),0
run() { # run <tag> <dtype> <ms> <blocksize> <cycles> <configs>
  [ -s $D/$1.csv ] && return
  echo "$(date -u +%T) $1"; python $B/alt1103b.py --configs $6 --dtype $2 --ms $3 --shapes $SH --cycles $5 --blocksize $4 --out $D/$1.csv.tmp > $D/$1.log 2>&1 && mv $D/$1.csv.tmp $D/$1.csv
}
ALL=A=table:g1100_f16,S=table:$B/cand/S1.py,V2=table:$B/cand/V2.py,V3=table:$B/cand/V3.py,B=table:$B/rulesB2.py
run v32_fp16 fp16 17,20,24,28,32 32 5 $ALL
run v32_bf16 bf16 17,20,24,28,32 32 5 $ALL
run v64_fp16 fp16 17,20,24,28,32 64 5 A=table:g1100_f16,S=table:$B/cand/S1.py,V3=table:$B/cand/V3.py
run v64_bf16 bf16 17,20,24,28,32 64 5 A=table:g1100_f16,S=table:$B/cand/S1.py,V3=table:$B/cand/V3.py
run b32_bf16 bf16 1,8,16,48,96 32 3 A=table:g1100_f16,S=table:$B/cand/S1.py
echo "$(date -u +%T) done"
