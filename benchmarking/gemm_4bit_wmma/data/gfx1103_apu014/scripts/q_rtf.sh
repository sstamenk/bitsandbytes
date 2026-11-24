#!/bin/bash
# block order probe (dev build in src/), in-process interleaved: fp32 with BNB_WMMA_ROW_TILES_FIRST=0 (gfx110x default
# for fp32) and =1, gfx1100 tiles (A) and the 16-row tile (T); fp16 with =1 (default) and =0, gfx1100 tiles.
D=~/bnbtune/data/alt; mkdir -p $D
B=$HOME/bnbtune/bin
[ -s $D/rtf_fp32.csv ] || { echo "$(date -u +%T) fp32"; python $B/alt1103b.py --configs A0=table:g1100_f32@rtf0,A1=table:g1100_f32@rtf1,T0=table:$B/t16_f32.py@rtf0,T1=table:$B/t16_f32.py@rtf1 --dtype fp32 --ms 24,48,96,192 --shapes $(seq -s, 1 39),0 --cycles 3 --out $D/rtf_fp32.csv.tmp > $D/rtf_fp32.log 2>&1 && mv $D/rtf_fp32.csv.tmp $D/rtf_fp32.csv; }
[ -s $D/rtf_fp16.csv ] || { echo "$(date -u +%T) fp16"; python $B/alt1103b.py --configs A1=table:g1100_f16@rtf1,A0=table:g1100_f16@rtf0 --dtype fp16 --ms 24,48,96,192 --shapes $(seq -s, 1 39),0 --cycles 3 --out $D/rtf_fp16.csv.tmp > $D/rtf_fp16.log 2>&1 && mv $D/rtf_fp16.csv.tmp $D/rtf_fp16.csv; }
echo "$(date -u +%T) done"
