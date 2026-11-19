#!/bin/bash
# tile sweeps (all 40 shapes), forced WMMA; run inside job.sh
D=~/bnbtune/data/sweep; mkdir -p $D
ALL=$(seq -s, 0 39)
export BNB_GEMM4BIT_KERNEL=wmma
[ -e $D/sweep_fp16.csv.done ] || { echo "$(date -u +%T) fp16"; python ~/bnbtune/fastsleep.py sweep.py --dtype fp16 --shapes $ALL --ms 1,2,4,8,12,16,20,24,28,32,40,48,64,96,128,160,192,256,384 --out $D/sweep_fp16.csv > $D/sweep_fp16.log && touch $D/sweep_fp16.csv.done; }
[ -e $D/sweep_fp32.csv.done ] || { echo "$(date -u +%T) fp32"; python ~/bnbtune/fastsleep.py sweep.py --dtype fp32 --shapes $ALL --ms 4,6,8,12,16,20,24,32,48,64,96,128,192,256,384,512 --out $D/sweep_fp32.csv > $D/sweep_fp32.log && touch $D/sweep_fp32.csv.done; }
echo "$(date -u +%T) done"
