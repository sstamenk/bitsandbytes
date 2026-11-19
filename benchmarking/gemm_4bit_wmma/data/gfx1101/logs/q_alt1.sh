#!/bin/bash
# interleaved A (inherited gfx11) vs B2 (gfx1101 fit) tables, fp16 and bf16, 5 cycles; run inside job.sh
D=~/bnbtune/data/alt; mkdir -p $D
MS=1,3,5,8,9,12,16,17,20,24,32,33,40,48,49,56,64
for dt in fp16 bf16; do
  [ -s $D/alt1_$dt.csv.done ] || { echo "$(date -u +%T) $dt"; python ~/bnbtune/fastsleep.py alt.py --tables A=rules/gfx11_f16.py,B2=rules/B2.py --dtype $dt --ms $MS --cycles 5 --out $D/alt1_$dt.csv > $D/alt1_$dt.log && touch $D/alt1_$dt.csv.done; }
done
echo "$(date -u +%T) done"
