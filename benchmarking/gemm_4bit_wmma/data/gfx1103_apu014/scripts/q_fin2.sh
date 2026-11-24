#!/bin/bash
# q_fin2.sh <part>: final dev build (TREE=fin2 = final bf16/fp16 table cand/S7.py; fp32 table and cutoffs as fin1),
# in-process interleaved: A = the base dispatch through overrides (gfx1100 table), F = default dispatch of the build.
#   altF: fp16, bf16, 31 M values, 5 cycles; *_x: F against cand/S7.py through the override; *_m1: F against forced SIMT.
#   altB: blocksize 32 (fp16, bf16) and 128 (fp16), 3 cycles.
#   nest: double-quantized absmax (alt1103c.py --nested), fp16, bf16, 3 cycles; fp32 with the gfx1103 cutoffs.
#   fp32: A = gfx1103 cutoffs + gfx1100 tiles: fp32_lo (23 M values 1-160, 5 cycles), fp32_hi (256, 384, 512, 3 cycles),
#         fp32_x (F against cand/F1.py through the override), fp32_d (F against the gfx1150 fp32 table), bs32_fp32.
D=~/bnbtune/data/altF2; mkdir -p $D
B=$HOME/bnbtune/bin; SH=$(seq -s, 1 39),0
CUT=2/1073741824+4/3840
run() { # run <tag> <dtype> <ms> <cycles> <configs> [extra args]
  [ -s $D/$1.csv ] && return
  local tag=$1 dt=$2 ms=$3 cyc=$4 cfg=$5; shift 5
  local py=alt1103b.py; [[ " $* " == *" --nested "* ]] && py=alt1103c.py
  echo "$(date -u +%T) $tag"; python $B/$py --configs $cfg --dtype $dt --ms $ms --shapes $SH --cycles $cyc "$@" --out $D/$tag.csv.tmp > $D/$tag.log 2>&1 && mv $D/$tag.csv.tmp $D/$tag.csv
}
AF=A=table:g1100_f16,F=auto
case $1 in
  altF) for dt in fp16 bf16; do
      run $dt $dt 1,2,3,4,6,8,9,12,16,17,20,24,28,32,33,40,48,49,56,64,65,72,80,88,96,97,128,192,256,384,512 5 $AF
      run ${dt}_x $dt 1,8,16,24,32,48,64,96 3 F=auto,X=table:$B/cand/S7.py
      run ${dt}_m1 $dt 1,2 5 F=auto,S=simt
    done;;
  altB) run bs32_fp16 fp16 1,8,16,24,32,48,96 3 $AF --blocksize 32
    run bs32_bf16 bf16 1,8,16,24,32,48,96 3 $AF --blocksize 32
    run bs128_fp16 fp16 1,8,16,24,32,48,96 3 $AF --blocksize 128;;
  nest) run nest_fp16 fp16 1,8,16,24,32,48,64,96,128 3 $AF --nested
    run nest_bf16 bf16 1,8,16,24,32,48,64,96 3 $AF --nested
    run nest_fp32 fp32 4,8,16,48,80,128 3 A=cut:$CUT:g1100_f32,F=auto --nested;;
  fp32) run fp32_lo fp32 1,2,3,4,5,6,8,9,12,16,17,24,32,33,48,49,64,72,80,96,112,128,160 5 A=cut:$CUT:g1100_f32,F=auto
    run fp32_hi fp32 256,384,512 3 A=cut:$CUT:g1100_f32,F=auto
    run fp32_x fp32 4,8,16,48,80 3 F=auto,X=cut:$CUT:$B/cand/F1.py
    run fp32_d fp32 5,8,16,48,64,128 3 F=auto,D=cut:$CUT:g1150_f32
    run bs32_fp32 fp32 8,16,48,128 3 A=cut:$CUT:g1100_f32,F=auto --blocksize 32;;
esac
echo "$(date -u +%T) done"
