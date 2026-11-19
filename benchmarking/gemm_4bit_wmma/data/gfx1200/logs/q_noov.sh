#!/bin/bash
D=~/bnbtune/data/noov; mkdir -p $D
echo "$(date -u +%T) dequant stock hipBLASLt"; echo "PYTHONPATH=$PYTHONPATH"
PYTHONPATH=~/bnbtune/src_b python bench.py --dtypes fp16,bf16 --kernels dequant --ms 256,384,416,448,480,512 --out $D/dequant_stockblas.csv
echo "$(date -u +%T) done"
