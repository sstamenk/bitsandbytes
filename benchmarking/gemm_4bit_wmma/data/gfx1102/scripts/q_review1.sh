#!/bin/bash
# Astra round-1 requests, final dev build (srcb), in-process interleaved (alt2.py); inside job.sh.
export BNB_BENCH_SLEEP=5e6
export PYTHONPATH=~/bnbtune/srcb
D=~/bnbtune/data/review1; mkdir -p $D; R=/home/ubuntu/bnbtune
B=1073741824
run() { n=$1; shift; [ -s $D/$n.csv.done ] || { echo "$(date -u +%T) $n"; python alt2.py "$@" --out $D/$n.csv > $D/$n.log && touch $D/$n.csv.done; }; }
# R1: fp32 M = 4 cutoff: old = SIMT everywhere (gfx11), new = WMMA (final tile) iff N >= 8192
run r1_fp32_m4 --configs "old=simt,new=cut:3/$B+4/8192:$R/rulesB32.py" --dtype fp32 --ms 4 --cycles 5
# R2: M limit at blocksize 32: fused (final tiles = gfx11 tiles at M > 64) vs dequant + F.linear
for dt in fp16 bf16; do run r2_bs32_$dt --configs "fused=table:g11_f16,dequant=dequant" --dtype $dt --blocksize 32 --ms 448,512 --cycles 3; done
# R3: reuse checks: final vs gfx1101 vs gfx115x tables at M = 1, 8 (bf16/fp16); fp32 M = 17, 24 (gfx11 / final / gfx115x, + SIMT)
for dt in fp16 bf16; do run r3_reuse_$dt --configs "final=table:$R/rulesB.py,g1101=table:$R/rules1101.py,g115=table:g115_f16" --dtype $dt --ms 1,8 --cycles 5; done
run r3_fp32 --configs "g11=table:g11_f32,final=table:$R/rulesB32.py,g115=table:g115_f32,g1101=table:$R/rules1101_32.py,simt=simt" --dtype fp32 --ms 17,24 --cycles 5
echo "$(date -u +%T) done"
