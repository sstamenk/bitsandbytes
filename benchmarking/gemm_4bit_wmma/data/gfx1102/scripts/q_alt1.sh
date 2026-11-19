#!/bin/bash
# in-process interleaved A (gfx11 table) vs B (gfx1102 candidate) via BNB_WMMA_CFG, fp16 and bf16, 5 cycles; runs inside job.sh
export BNB_BENCH_SLEEP=5e6
D=~/bnbtune/data/alt; mkdir -p $D
MS=1,2,4,8,9,12,16,17,20,24,31,32,33,40,48,49,56,64,65
for dt in fp16 bf16; do
  [ -s $D/alt1_$dt.csv.done ] || { echo "$(date -u +%T) $dt"; python alt.py --tables A=g11_f16,B=/home/ubuntu/bnbtune/rulesB.py --dtype $dt --ms $MS --cycles 5 --out $D/alt1_$dt.csv > $D/alt1_$dt.log && touch $D/alt1_$dt.csv.done; }
done
echo "$(date -u +%T) done"
