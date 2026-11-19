#!/bin/bash
# dequant + F.linear with the hipBLASLt fp32-fix overlay; runs inside job.sh
. ~/bnbtune/hipblaslt_fix/env.sh || exit 1
export BNB_BENCH_SLEEP=5e6  # ~50 ms idle before each timed batch (5e7 = 0.5 s dominated run time); launches queue in <5 ms
D=~/bnbtune/data/deqfix; mkdir -p $D
MS=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,22,24,26,28,30,32,36,40,44,48,52,56,60,64,72,80,88,96,104,112,120,128,144,160,176,192,208,224,240,256,288,320,352,384,416,448,480,512
[ -s $D/fp32_dequant.csv ] || { echo "$(date -u +%T) fp32"; python bench.py --dtypes fp32 --kernels dequant --ms $MS --out $D/fp32_dequant.csv; }
[ -s $D/f16_dequant.csv ] || { echo "$(date -u +%T) fp16/bf16"; python bench.py --dtypes fp16,bf16 --kernels dequant --ms 16,128,256,320,384,416,448,480,512 --out $D/f16_dequant.csv; }
echo "$(date -u +%T) done"
