#!/bin/bash
# q_r.sh <part>: review round 1 settling measurements (alt1103b.py, in-process interleaved, all 40 shapes), one part
# per queue job so that every run starts with the level `high` rewritten. Output in ~/bnbtune/data/r1.
#   r1_fp16 / r1_bf16: blocksize 64, M = 49-96, A = gfx1100 table, F = default dispatch of the build, 10 cycles.
#   bs32cap_fp16 / bs32cap_bf16: blocksize 32, M = 384-512, F = default dispatch, Q = dequant + F.linear, 5 cycles.
#   fp32_afq: fp32, M = 256-512, A = gfx1103 cutoffs + gfx1100 tiles, F = default dispatch, Q = dequant, 5 cycles
#             (reset the GPU at its first dequant call, twice; replaced by the two runs below).
#   fp32_fq: fp32, M = 384-512, F = default dispatch, Q = dequant, 5 cycles (--syncwarm).
#   fp32_fqb: the same for shapes 0 and 1 only (shape 0 is the one the first fp32_fq run did not reach).
#   fp32_af: fp32, M = 256-512, A = gfx1103 cutoffs + gfx1100 tiles, F = default dispatch, 5 cycles.
#   abf_fp16 / abf_bf16: blocksize 64, M = 17-32, A = gfx1100 table, B = rulesB2.py, F = default dispatch, 5 cycles.
D=~/bnbtune/data/r1; mkdir -p $D
B=$HOME/bnbtune/bin; SH=$(seq -s, 1 39),0
CUT=2/1073741824+4/3840
run() { # run <tag> <dtype> <ms> <cycles> <configs> [extra args]
  [ -s $D/$1.csv ] && return
  local tag=$1 dt=$2 ms=$3 cyc=$4 cfg=$5; shift 5
  echo "$(date -u +%T) $tag"; python $B/alt1103b.py --configs $cfg --dtype $dt --ms $ms --shapes $SH --cycles $cyc "$@" --out $D/$tag.csv.tmp > $D/$tag.log 2>&1 && mv $D/$tag.csv.tmp $D/$tag.csv
}
case $1 in
  r1_fp16) run r1_fp16 fp16 49,56,64,65,72,80,88,96 10 A=table:g1100_f16,F=auto;;
  r1_bf16) run r1_bf16 bf16 49,56,64,65,72,80,88,96 10 A=table:g1100_f16,F=auto;;
  bs32cap_fp16) run cap_bs32_fp16 fp16 384,416,448,480,512 5 F=auto,Q=dequant --blocksize 32;;
  bs32cap_bf16) run cap_bs32_bf16 bf16 384,416,448,480,512 5 F=auto,Q=dequant --blocksize 32;;
  fp32_afq) run fp32_afq fp32 256,384,416,448,480,512 5 A=cut:$CUT:g1100_f32,F=auto,Q=dequant;;
  fp32_fq) run fp32_fq fp32 384,416,448,480,512 5 F=auto,Q=dequant --syncwarm;;
  fp32_fqb) SH=0,1; run fp32_fqb fp32 384,416,448,480,512 5 F=auto,Q=dequant --syncwarm;;
  fp32_af) run fp32_af fp32 256,384,416,448,480,512 5 A=cut:$CUT:g1100_f32,F=auto;;
  abf_fp16) run abf_fp16 fp16 17,20,24,28,32 5 A=table:g1100_f16,B=table:$B/rulesB2.py,F=auto;;
  abf_bf16) run abf_bf16 bf16 17,20,24,28,32 5 A=table:g1100_f16,B=table:$B/rulesB2.py,F=auto;;
esac
echo "$(date -u +%T) done; GPU resets since boot: $(sudo -n dmesg | grep -c 'GPU reset begin')"
