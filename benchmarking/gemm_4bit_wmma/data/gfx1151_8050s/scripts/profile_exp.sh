#!/bin/bash
# profile_exp.sh <tag>: tile subset at M=24/96/384 on 8 shapes, platform_profile balanced -> performance -> balanced.
# Runs inside job.sh (perf level high); restores platform_profile on exit.
P=/sys/firmware/acpi/platform_profile
orig=$(cat $P)
trap 'echo $orig | sudo tee $P >/dev/null; echo "restored platform_profile $(cat $P)"' EXIT INT TERM
SH=1,4,6,9,10,12,14,16
CF="1,2,1,1,8,2;4,1,1,4,2,2;4,2,1,4,2,2;4,1,1,4,1,2;2,1,1,4,4,2;4,2,1,4,1,2"
for prof in balanced performance balanced; do
  echo $prof | sudo tee $P >/dev/null; sleep 20
  echo "== $prof $(cat $P) $(date -u +%T)"
  python sweep.py --shapes $SH --dtype fp16 --ms 24,96,384 --cfgs "$CF" --out ~/bnbtune/data/prof_${1}_$prof$([ -f ~/bnbtune/data/prof_${1}_$prof.csv ] && echo 2).csv
  python bench.py --kernels auto,dequant --dtypes fp16 --shapes $SH --ms 16,128,512 --out ~/bnbtune/data/profb_${1}_$prof$([ -f ~/bnbtune/data/profb_${1}_$prof.csv ] && echo 2).csv
done
