#!/bin/bash
# fp16 tile sweep, second part (shapes 19-39; the first run was stopped there because the 0.5-s idle sleep per
# measurement dominated its run time); same M list and reps; runs inside job.sh
export BNB_BENCH_SLEEP=5e6  # ~50 ms idle before each timed batch (5e7 = 0.5 s dominated run time); launches queue in <5 ms
D=~/bnbtune/data/sweep; mkdir -p $D
export BNB_GEMM4BIT_KERNEL=wmma
[ -e $D/sweep_fp16b.csv.done ] || { echo "$(date -u +%T) fp16b"; python sweep.py --dtype fp16 --shapes $(seq -s, 19 39) --ms 1,2,4,8,12,16,20,24,28,32,40,48,56,64,80,96,112,128,160,192,256,384,512 --out $D/sweep_fp16b.csv > $D/sweep_fp16b.log && touch $D/sweep_fp16b.csv.done; }
echo "$(date -u +%T) done"
