#!/bin/bash
# q_probe.sh <tag> <ms> <configs> [shapes]: short fp32 run of alt1103b.py (1 cycle, default shape 1 = llama3-70b down)
# to find what resets the GPU at the start of `q_r.sh fp32_afq`. PROBE_ARGS: extra arguments (--syncwarm). Output in
# ~/bnbtune/data/probe.
D=~/bnbtune/data/probe; mkdir -p $D
python ~/bnbtune/bin/alt1103b.py --configs $3 --dtype fp32 --ms $2 --shapes ${4:-1} --cycles 1 $PROBE_ARGS --out $D/$1.csv > $D/$1.log 2>&1
echo "rc=$? $(date -u +%T) GPU resets since boot: $(sudo -n dmesg | grep -c 'GPU reset begin')"; cat $D/$1.log | tail -3
