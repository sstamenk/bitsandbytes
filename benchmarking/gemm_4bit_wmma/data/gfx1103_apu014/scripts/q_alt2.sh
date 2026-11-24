#!/bin/bash
# in-process interleaved tile tables (dev build in src/), fp16 and bf16: A = gfx1100 table, B = rulesB2.py (fit on
# this machine only), S = cand/S1.py, T = cand/S3.py (both also weigh the other 760M's 5-shape sweep)
D=~/bnbtune/data/alt; mkdir -p $D
B=$HOME/bnbtune/bin
MS=1,2,3,4,6,8,9,12,16,17,20,24,28,32,33,40,48,49,56,64,65,72,80,88,96,97,128
for dt in fp16 bf16; do
  [ -s $D/alt2_$dt.csv ] || { echo "$(date -u +%T) $dt"; python $B/alt1103b.py --configs A=table:g1100_f16,B=table:$B/rulesB2.py,S=table:$B/cand/S1.py,T=table:$B/cand/S3.py --dtype $dt --ms $MS --shapes $(seq -s, 1 39),0 --cycles 5 --out $D/alt2_$dt.csv.tmp > $D/alt2_$dt.log 2>&1 && mv $D/alt2_$dt.csv.tmp $D/alt2_$dt.csv; }
done
echo "$(date -u +%T) done"
