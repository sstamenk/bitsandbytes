#!/bin/bash
# Final A/B/A/B, fp32 only, after reverting the fp32 M = 17-24 bucket to the gfx11 tiles (srcb rebuilt).
# gfx1102 target, dev build); per dtype A1 B1 A2 B2, dense M + bucket edges. Plus forced WMMA (B) for fp32 and SIMT
# (A; unchanged) at small M for the SIMT/WMMA choice. Runs inside job.sh (pinned, guarded).
export BNB_BENCH_SLEEP=5e6
D=~/bnbtune/data/aba2; mkdir -p $D
cd ~/bnbtune/src/benchmarking/gemm_4bit_wmma
MS=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,22,24,26,28,30,31,32,33,36,40,44,48,49,52,56,60,64,65,72,80,88,96,104,112,120,128,144,160,176,192,208,224,240,256,288,320,352,384,416,448,480,512
for dt in fp32; do
  for v in a1:src b1:srcb a2:src b2:srcb; do
    n=${v%%:*}; d=${v#*:}
    [ -s $D/${dt}_$n.csv ] || { echo "$(date -u +%T) $dt $n"; PYTHONPATH=~/bnbtune/$d python bench.py --kernels auto --dtypes $dt --ms $MS --out $D/${dt}_$n.csv; }
  done
done
[ -s $D/fp32_wmma_b.csv ] || { echo "$(date -u +%T) fp32 wmma b"; PYTHONPATH=~/bnbtune/srcb python bench.py --kernels wmma --dtypes fp32 --ms 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,24,28,32,40,48,56,64 --out $D/fp32_wmma_b.csv; }
echo "$(date -u +%T) done"
