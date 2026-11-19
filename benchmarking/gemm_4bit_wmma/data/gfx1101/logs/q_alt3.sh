#!/bin/bash
# interleaved A vs B2 vs B3 at M = 17-32, fp16 and bf16, 5 cycles
D=~/bnbtune/data/alt
for dt in fp16 bf16; do
  [ -s $D/alt3_$dt.csv.done ] || { echo "$(date -u +%T) $dt"; python ~/bnbtune/fastsleep.py alt.py --tables A=rules/gfx11_f16.py,B2=rules/B2.py,B3=rules/B3.py --dtype $dt --ms 17,20,24,28,32 --cycles 5 --out $D/alt3_$dt.csv > $D/alt3_$dt.log && touch $D/alt3_$dt.csv.done; }
done
