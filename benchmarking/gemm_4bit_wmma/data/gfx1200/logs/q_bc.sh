#!/bin/bash
# B (3-leaf M<=16) vs C (2-leaf M<=16): B/C/B, fp16+bf16, M=2..16
D=~/bnbtune/data/abc; mkdir -p $D
REST=${PYTHONPATH#*:}
MS=2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
run() { local tag=$1 src=$2; [ -s $D/$tag.csv ] && return; echo "$(date -u +%T) $tag"; PYTHONPATH=~/bnbtune/$src:$REST python bench.py --dtypes fp16,bf16 --kernels auto --ms $MS --out $D/$tag.csv; }
run B1 src_b; run C src_c; run B2 src_b
echo "$(date -u +%T) done"
