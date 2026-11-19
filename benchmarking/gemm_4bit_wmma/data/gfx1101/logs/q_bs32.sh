#!/bin/bash
# M limit robustness: blocksize 32, fused (B build) vs dequant+F.linear, fp16/bf16, M around the cap
export BNB_SLEEP_SCALE=0.1
D=~/bnbtune/data/cap; cd ~/bnbtune/srcb/benchmarking/gemm_4bit_wmma; export PYTHONPATH=~/bnbtune/fastsite:~/bnbtune/srcb
[ -s $D/cap_bs32.csv ] || python bench.py --blocksize 32 --kernels auto,dequant --dtypes fp16,bf16 --ms 384,416,448,464,480,496,512 --out $D/cap_bs32.csv
