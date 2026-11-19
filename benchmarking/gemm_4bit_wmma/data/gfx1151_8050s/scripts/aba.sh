#!/bin/bash
# aba.sh <tag> <ms> <dtypes>: auto runs A (src), B (srcb), A (src) back to back (inside job.sh: pinned clocks).
tag=$1; ms=$2; dts=${3:-fp16}
for v in a1:src b:srcb a2:src; do
  n=${v%%:*}; d=${v#*:}
  echo "== $n $d $(date -u +%T)"
  (cd ~/bnbtune/$d/benchmarking/gemm_4bit_wmma && PYTHONPATH=~/bnbtune/$d python bench.py --kernels auto --dtypes $dts --ms $ms --out ~/bnbtune/data/${tag}_$n.csv)
done
