#!/bin/bash
# First-in-process artifact check: llama3-70b gate/up (shape 0, the first shape of every bench.py worker) at M = 1-3,
# measured three times in a row in one worker, A (src) and B (srcb) builds, auto and simt; runs inside job.sh
export BNB_BENCH_SLEEP=5e6
D=~/bnbtune/data/art; mkdir -p $D
cd ~/bnbtune/src/benchmarking/gemm_4bit_wmma
for c in 1 2; do for v in a:src b:srcb; do n=${v%%:*}; d=${v#*:}
  PYTHONPATH=~/bnbtune/$d python bench.py --kernels auto,simt --dtypes fp16,bf16 --shapes 0 --ms 1,2,3,1,2,3,1,2,3 --out $D/art_${n}_c$c.csv
done; done
echo "$(date -u +%T) done"
