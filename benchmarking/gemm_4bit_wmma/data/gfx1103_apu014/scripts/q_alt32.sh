#!/bin/bash
# fp32 in-process interleaved (dev build in src/): forced SIMT, gfx1100 tiles (A), candidate B, gfx1150 tiles (D)
D=~/bnbtune/data/alt; mkdir -p $D
MS=1,2,3,4,5,6,8,9,12,16,17,20,24,32,33,40,48,49,64,80
[ -s $D/alt32.csv ] || { echo "$(date -u +%T) fp32"; python ~/bnbtune/bin/alt1103.py --configs S=simt,A=table:g1100_f32,B=table:$HOME/bnbtune/bin/rules32B.py,D=table:g1150_f32 --dtype fp32 --ms $MS --shapes $(seq -s, 1 39),0 --cycles 5 --out $D/alt32.csv.tmp > $D/alt32.log 2>&1 && mv $D/alt32.csv.tmp $D/alt32.csv; }
echo "$(date -u +%T) done"
