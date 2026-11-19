#!/bin/bash
# Extra checks on the final dev build (srcb): (1) block order: BNB_WMMA_ROW_TILES_FIRST=0/1 alternating, 2 cycles each,
# fp16 and fp32, M = 48-512; (2) M limit with blocksize 32: auto vs dequant, fp16/bf16, M = 352-512. Inside job.sh.
export BNB_BENCH_SLEEP=5e6
D=~/bnbtune/data/extra; mkdir -p $D
cd ~/bnbtune/src/benchmarking/gemm_4bit_wmma
export PYTHONPATH=~/bnbtune/srcb
for c in 1 2; do
  for r in 1 0; do
    [ $c = 2 ] && r=$((1 - r))
    [ -s $D/rtf${r}_c$c.csv ] || { echo "$(date -u +%T) rtf=$r c$c"; BNB_WMMA_ROW_TILES_FIRST=$r python bench.py --kernels auto --dtypes fp16,fp32 --ms 48,64,96,128,192,256,384,512 --out $D/rtf${r}_c$c.csv; }
  done
done
[ -s $D/bs32.csv ] || { echo "$(date -u +%T) bs32"; python bench.py --kernels auto,dequant --dtypes fp16,bf16 --blocksize 32 --ms 352,385,416,448,480,512 --out $D/bs32.csv; }
echo "$(date -u +%T) done"
