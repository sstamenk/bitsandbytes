#!/bin/bash
D=~/bnbtune/data/sweep; ALL=$(seq -s, 0 39)
export BNB_GEMM4BIT_KERNEL=wmma
echo "$(date -u +%T) fp16 extra"; python sweep.py --dtype fp16 --shapes $ALL --ms 20,28,40,104,112,120,144 --out $D/sweep_fp16_extra.csv > $D/sweep_fp16_extra.log
echo "$(date -u +%T) fp32 extra"; python sweep.py --dtype fp32 --shapes $ALL --ms 10,20,28,40 --out $D/sweep_fp32_extra.csv > $D/sweep_fp32_extra.log
echo "$(date -u +%T) done"
