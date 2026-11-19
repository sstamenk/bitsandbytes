#!/bin/bash
# interleaved A (inherited gfx11 fp32) vs C32 (gfx1101 fp32 fit), 5 cycles; run inside job.sh
D=~/bnbtune/data/alt; mkdir -p $D
[ -s $D/alt2_fp32.csv.done ] || { echo "$(date -u +%T) fp32"; python ~/bnbtune/fastsleep.py alt.py --tables A=rules/gfx11_f32.py,C32=rules/C32.py --dtype fp32 --ms 5,6,7,8,9,10,12,13,14,16 --cycles 5 --out $D/alt2_fp32.csv > $D/alt2_fp32.log && touch $D/alt2_fp32.csv.done; }
echo "$(date -u +%T) done"
