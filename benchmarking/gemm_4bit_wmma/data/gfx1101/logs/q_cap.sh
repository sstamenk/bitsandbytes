#!/bin/bash
# M limit: fused (inherited table, unchanged at M > 96) vs dequant+F.linear interleaved, 3 cycles; run inside job.sh
D=~/bnbtune/data/cap; mkdir -p $D
MS=352,384,400,416,432,448,464,480,496,512
for dt in fp16 bf16; do
  [ -s $D/cap_$dt.csv.done ] || { echo "$(date -u +%T) $dt"; python ~/bnbtune/fastsleep.py alt.py --tables A=rules/gfx11_f16.py --dequant --dtype $dt --ms $MS --cycles 3 --out $D/cap_$dt.csv > $D/cap_$dt.log && touch $D/cap_$dt.csv.done; }
done
echo "$(date -u +%T) done"
