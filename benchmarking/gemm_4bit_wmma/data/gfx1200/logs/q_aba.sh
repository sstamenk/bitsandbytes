#!/bin/bash
# A/B/A of auto (A = inherited gfx1201 rules build in src, B = gfx1200 rules build in src_b), then extras.
D=~/bnbtune/data/aba; mkdir -p $D
REST=${PYTHONPATH#*:}   # overlay bootstrap etc. (pinrun put ~/bnbtune/src first)
MS=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,22,24,26,28,30,32,36,40,44,48,52,56,60,64,72,80,88,96,104,112,120,128,144,160,176,192,208,224,240,256,288,320,352,384,416,448,480,512
run() { # tag src kernels dtypes ms [extra]
  local tag=$1 src=$2 k=$3 dts=$4 ms=$5; shift 5
  [ -s $D/$tag.csv ] && return
  echo "$(date -u +%T) $tag"
  PYTHONPATH=~/bnbtune/$src:$REST python bench.py --dtypes $dts --kernels $k --ms $ms --out $D/$tag.csv "$@"
}
run A1 src auto fp16,bf16,fp32 $MS
run B  src_b auto fp16,bf16,fp32 $MS
run A2 src auto fp16,bf16,fp32 $MS
run B_wmma src_b wmma fp16,bf16,fp32 $MS
# row-tiles-first check on B (default 1 on gfx12)
BNB_WMMA_ROW_TILES_FIRST=0 run B_rtf0 src_b auto fp16,fp32 48,64,96,128,192,256,384,512
# Python cap: repeat auto/dequant at the top Ms (B), and blocksize 32
run cap_rep src_b auto,dequant fp16,bf16 320,352,384,416,448,480,512
run cap_bs32 src_b auto,dequant fp16,bf16 256,320,384,448,512 --blocksize 32
echo "$(date -u +%T) done"
