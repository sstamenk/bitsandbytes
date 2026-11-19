#!/bin/bash
# fp32 tile sweep (A dev build), M list trimmed for time; runs inside job.sh
export BNB_BENCH_SLEEP=5e6  # ~50 ms idle before each timed batch (5e7 = 0.5 s dominated run time); launches queue in <5 ms
D=~/bnbtune/data/sweep; mkdir -p $D
export BNB_GEMM4BIT_KERNEL=wmma
[ -e $D/sweep_fp32b.csv.done ] || { echo "$(date -u +%T) fp32"; python sweep.py --dtype fp32 --shapes $(seq -s, 0 39) --ms 4,8,12,16,24,32,48,64,128,256 --out $D/sweep_fp32.csv > $D/sweep_fp32.log && touch $D/sweep_fp32b.csv.done; }
echo "$(date -u +%T) done"
