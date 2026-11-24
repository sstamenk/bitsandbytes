#!/bin/bash
# in-process interleaved tile tables (dev build in src/): A = gfx1100 table, B = rulesB.py, C = rulesC.py; fp16 and bf16
D=~/bnbtune/data/alt; mkdir -p $D
MS=1,2,3,4,6,8,9,12,16,17,20,24,28,32,33,40,48,49,56,64,72,80,96,97,128
for dt in fp16 bf16; do
  [ -s $D/alt1_$dt.csv ] || { echo "$(date -u +%T) $dt"; python ~/bnbtune/bin/alt1103.py --configs A=table:g1100_f16,B=table:$HOME/bnbtune/bin/rulesB.py,C=table:$HOME/bnbtune/bin/rulesC.py --dtype $dt --ms $MS --shapes $(seq -s, 1 39),0 --cycles 5 --out $D/alt1_$dt.csv.tmp > $D/alt1_$dt.log 2>&1 && mv $D/alt1_$dt.csv.tmp $D/alt1_$dt.csv; }
done
echo "$(date -u +%T) done"
