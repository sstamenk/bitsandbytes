#!/bin/bash
# pinrun.sh <log> <cmd...>: pin card1 (gfx1200) to high, run cmd under runs_guard, always restore auto.
lvl=/sys/class/drm/card1/device/power_dpm_force_performance_level
restore() { echo auto | sudo tee $lvl > /dev/null; }
trap restore EXIT INT TERM
echo high | sudo tee $lvl > /dev/null
echo "level=$(cat $lvl) start $(date -u +%T)"
cd ~/bnbtune/src/benchmarking/gemm_4bit_wmma
source ~/bnbtune/venv/bin/activate
[ -n "$USE_OVERLAY" ] && source ~/bnbtune/hipblaslt_fix/env.sh
export HIP_VISIBLE_DEVICES=0 PYTHONPATH=~/bnbtune/src${PYTHONPATH:+:$PYTHONPATH}
log=$1; shift
./runs_guard.sh "$log" "$@"; rc=$?
echo "end $(date -u +%T) rc=$rc level=$(cat $lvl)"
exit $rc
