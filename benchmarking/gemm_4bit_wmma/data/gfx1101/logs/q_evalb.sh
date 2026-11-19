#!/bin/bash
# final eval on the B build (srcb, gfx1101 target): dense M, all kernels; sleep pre-roll scaled 0.1 (fastsite)
export BNB_SLEEP_SCALE=0.1
D=~/bnbtune/data/evalb; mkdir -p $D
MS=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,22,24,26,28,30,32,36,40,44,48,52,56,60,64,72,80,88,96,104,112,120,128,144,160,176,192,208,224,240,256,288,320,352,384,416,448,480,512
MSS=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,22,24,26,28,30,32,36,40,44,48,52,56,60,64
cd ~/bnbtune/srcb/benchmarking/gemm_4bit_wmma
export PYTHONPATH=~/bnbtune/fastsite:~/bnbtune/srcb
for dt in fp16 fp32 bf16; do
  for k in auto wmma dequant; do
    [ -s $D/${dt}_$k.csv ] || { echo "$(date -u +%T) $dt $k"; python bench.py --dtypes $dt --kernels $k --ms $MS --out $D/${dt}_$k.csv; }
  done
  [ -s $D/${dt}_simt.csv ] || { echo "$(date -u +%T) $dt simt"; python bench.py --dtypes $dt --kernels simt --ms $MSS --out $D/${dt}_simt.csv; }
done
echo "$(date -u +%T) done"
