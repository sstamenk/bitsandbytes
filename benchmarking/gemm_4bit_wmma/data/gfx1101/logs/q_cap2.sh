#!/bin/bash
# M limit, intermediate M: fused (inherited table, unchanged at M > 96) vs dequant+F.linear interleaved, 3 cycles; run inside job.sh
D=~/bnbtune/data/cap; mkdir -p $D
MS=392,408,424,440,447,449,456,465,472,479,481
for dt in fp16 bf16; do
  [ -s $D/cap2_$dt.csv.done ] || { echo "$(date -u +%T) $dt"; python ~/bnbtune/fastsleep.py alt.py --tables A=rules/gfx11_f16.py --dequant --dtype $dt --ms $MS --cycles 3 --out $D/cap2_$dt.csv > $D/cap2_$dt.log && touch $D/cap2_$dt.csv.done; }
done
echo "$(date -u +%T) done"
