#!/bin/bash
# fp32 M limit: fused (inherited fp32 table) vs dequant+F.linear with hipBLASLt (torch default) and rocBLAS, 3 cycles
D=~/bnbtune/data/cap; mkdir -p $D
MS=64,96,128,160,192,256,320,384,448,512
[ -s $D/cap_fp32_lt.csv.done ] || { echo "$(date -u +%T) fp32 hipblaslt"; python ~/bnbtune/fastsleep.py alt.py --tables A=rules/gfx11_f32.py --dequant --dtype fp32 --ms $MS --cycles 3 --out $D/cap_fp32_lt.csv > $D/cap_fp32_lt.log && touch $D/cap_fp32_lt.csv.done; }
[ -s $D/cap_fp32_rocblas.csv.done ] || { echo "$(date -u +%T) fp32 rocblas"; BNB_TORCH_BLAS=cublas python ~/bnbtune/fastsleep.py alt.py --tables A=rules/gfx11_f32.py --dequant --dtype fp32 --ms $MS --cycles 3 --out $D/cap_fp32_rocblas.csv > $D/cap_fp32_rocblas.log && touch $D/cap_fp32_rocblas.csv.done; }
echo "$(date -u +%T) done"
