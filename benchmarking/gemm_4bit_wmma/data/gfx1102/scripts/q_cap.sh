#!/bin/bash
# M-limit check: fused (WMMA, gfx11 tiles = gfx1102 candidate tiles at M > 64) vs dequant + F.linear, same process,
# alternating order, 3 cycles; fp16 and bf16; runs inside job.sh
export BNB_BENCH_SLEEP=5e6
D=~/bnbtune/data/alt; mkdir -p $D
for dt in fp16 bf16; do
  [ -s $D/cap_$dt.csv.done ] || { echo "$(date -u +%T) $dt"; python alt.py --tables A=g11_f16 --dequant --dtype $dt --ms 352,385,416,448,480,512 --cycles 3 --out $D/cap_$dt.csv > $D/cap_$dt.log && touch $D/cap_$dt.csv.done; }
done
echo "$(date -u +%T) done"
