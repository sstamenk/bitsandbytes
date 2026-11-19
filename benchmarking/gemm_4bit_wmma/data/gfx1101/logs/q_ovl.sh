#!/bin/bash
# dequant+F.linear with the hipBLASLt fp32-fix overlay (ROCm/rocm-libraries#12465), all dtypes; run inside job.sh
D=~/bnbtune/data/ovl; mkdir -p $D
source ~/bnbtune/hipblaslt_fix/env.sh
MS=1,8,16,32,64,96,128,160,192,224,256,288,320,352,384,416,448,480,512
for dt in fp32 fp16 bf16; do
  [ -s $D/${dt}_dequant.csv ] || { echo "$(date -u +%T) $dt"; python bench.py --dtypes $dt --kernels dequant --ms $MS --out $D/${dt}_dequant.csv; }
done
echo "$(date -u +%T) done"
