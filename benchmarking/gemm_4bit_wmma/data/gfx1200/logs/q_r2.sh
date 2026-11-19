#!/bin/bash
# Round 2: A/B/A with the final rules (D) incl. boundary M values, D wmma, cap edges, Astra repeat commands.
D=~/bnbtune/data/r2; mkdir -p $D
REST=${PYTHONPATH#*:}
MS=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,22,24,26,28,30,31,32,36,40,44,48,52,56,60,64,72,80,88,96,97,103,104,112,120,127,128,144,160,176,192,208,224,240,256,288,320,352,384,416,448,480,512
run() { # tag src kernels dtypes ms [extra]
  local tag=$1 src=$2 k=$3 dts=$4 ms=$5; shift 5
  [ -s $D/$tag.csv ] && return
  echo "$(date -u +%T) $tag"
  PYTHONPATH=~/bnbtune/$src:$REST python bench.py --dtypes $dts --kernels $k --ms $ms --out $D/$tag.csv "$@"
}
run A3 src auto fp16,bf16,fp32 $MS
run D src_d auto fp16,bf16,fp32 $MS
run A4 src auto fp16,bf16,fp32 $MS
run D_wmma src_d wmma fp16,bf16,fp32 $MS
run cap_edges src_d auto,dequant fp16,bf16 385,417,449,481
for r in 1 2 3; do
  run review_small_$r src_d auto,simt,wmma,dequant fp16,fp32 1,4,5 --shapes 14,16,25,26,30
  run review_cap_$r src_d auto,wmma,dequant fp16 448 --shapes 25
done
echo "$(date -u +%T) done"
