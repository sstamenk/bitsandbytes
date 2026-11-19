#!/bin/bash
# fp32 in-process interleaved tile tables A (gfx11) / B32 / C32 via BNB_WMMA_CFG, forced WMMA, 5 cycles; inside job.sh
export BNB_BENCH_SLEEP=5e6
D=~/bnbtune/data/alt; mkdir -p $D
[ -s $D/alt32.csv.done ] || { echo "$(date -u +%T) fp32"; python alt.py --tables A=g11_f32,B=/home/ubuntu/bnbtune/rulesB32.py,C=/home/ubuntu/bnbtune/rulesC32.py --dtype fp32 --ms 4,5,6,7,8,9,12,16,17,20,24,25,32 --cycles 5 --out $D/alt32.csv > $D/alt32.log && touch $D/alt32.csv.done; }
echo "$(date -u +%T) done"
