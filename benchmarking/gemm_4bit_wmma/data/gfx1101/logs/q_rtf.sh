#!/bin/bash
# block order check on the B build: row tiles first off/on, fp32 and fp16
D=~/bnbtune/data/rtf; mkdir -p $D; cd ~/bnbtune/srcb/benchmarking/gemm_4bit_wmma; export PYTHONPATH=~/bnbtune/srcb
for dt in fp32 fp16; do
  [ -s $D/rtf_$dt.csv.done ] || { echo "$(date -u +%T) $dt"; python ~/bnbtune/fastsleep.py rtf.py --dtype $dt --ms 24,48,64,128,256,512 --cycles 3 --out $D/rtf_$dt.csv && touch $D/rtf_$dt.csv.done; }
done
