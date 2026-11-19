#!/bin/bash
# final A/B/A of the default dispatch (auto): A = src (1b6a36a dev build), B = srcb (gfx1101 target), all dtypes;
# then forced fp32 SIMT/WMMA on B for the cutoff check. Sleep pre-roll scaled 0.1 (fastsite/sitecustomize.py).
export BNB_SLEEP_SCALE=0.1 PYTHONPATH=~/bnbtune/fastsite
D=~/bnbtune/data/aba; mkdir -p $D
MS=1,2,3,4,5,6,7,8,9,10,12,13,14,16,17,20,24,28,32,33,40,48,49,56,64,65,72,96,97,128,192,256,384,512
for v in a1:src b:srcb a2:src; do
  n=${v%%:*}; d=${v#*:}
  [ -s $D/aba_$n.csv ] || { echo "$(date -u +%T) $n $d"; (cd ~/bnbtune/$d/benchmarking/gemm_4bit_wmma && PYTHONPATH=~/bnbtune/fastsite:~/bnbtune/$d python bench.py --kernels auto --dtypes fp16,bf16,fp32 --ms $MS --out $D/aba_$n.csv); }
done
[ -s $D/b_fp32_simt_wmma.csv ] || { echo "$(date -u +%T) b fp32 simt/wmma"; (cd ~/bnbtune/srcb/benchmarking/gemm_4bit_wmma && PYTHONPATH=~/bnbtune/fastsite:~/bnbtune/srcb python bench.py --kernels simt,wmma --dtypes fp32 --ms 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,24,32 --out $D/b_fp32_simt_wmma.csv); }
echo "$(date -u +%T) done"
