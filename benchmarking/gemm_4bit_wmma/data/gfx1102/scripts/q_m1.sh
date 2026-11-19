#!/bin/bash
# SIMT vs WMMA blocks (kernel forced per process): fp16/bf16 at M=1,2 and fp32 at M=3,4,5,6; 6 cycles, alternating order.
export BNB_BENCH_SLEEP=5e6  # ~50 ms idle before each timed batch (5e7 = 0.5 s dominated run time); launches queue in <5 ms
D=~/bnbtune/data/m1; mkdir -p $D
for c in 1 2 3 4 5 6; do
  if [ $((c % 2)) = 1 ]; then ks="simt,wmma"; else ks="wmma,simt"; fi
  for dt in fp16 bf16; do
    [ -s $D/${dt}_c$c.csv ] || python bench.py --dtypes $dt --kernels $ks --ms 1,2 --out $D/${dt}_c$c.csv
  done
  [ -s $D/fp32_c$c.csv ] || python bench.py --dtypes fp32 --kernels $ks --ms 3,4,5,6 --out $D/fp32_c$c.csv
  echo "$(date -u +%T) cycle $c"
done
