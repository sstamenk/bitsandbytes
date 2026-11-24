#!/bin/bash
# job.sh <name> <command...>: one timing job at perf level high on card1 (Radeon 760M); auto restored on exit.
# Tree: ${TREE:-src}. Logs in ~/bnbtune/logs: <name>.meta (start/end, level, sclk/mclk tables), <name>.clk (every 10 s:
# sclk/mclk level, busy, edge temp, current sclk, PPT), <name>.kfd (every KFD process every 5 s; ours are python),
# <name>.log (+ runs_guard CLEAN/CONTAMINATED for foreign users).
. ~/bnbtune/env.sh
name=$1; shift
D=/sys/class/drm/card1/device; L=$D/power_dpm_force_performance_level
mkdir -p $T/logs $T/data
restore() { echo auto | sudo -n tee $L >/dev/null; echo "$(date -u +%FT%TZ) restored $(cat $L)" >> $T/logs/$name.meta; }
trap restore EXIT INT TERM
echo high | sudo -n tee $L >/dev/null
{ echo "$(date -u +%FT%TZ) start $name level=$(cat $L) tree=${TREE:-src}"; echo "cmd: $*"; echo "users: $(who | awk '{print $1}' | sort -u | tr '\n' ' ')"; cat $D/pp_dpm_sclk $D/pp_dpm_mclk; } > $T/logs/$name.meta
( while true; do echo "$(date +%T) sclk=$(grep '\*' $D/pp_dpm_sclk|tr -d '\n') mclk=$(grep '\*' $D/pp_dpm_mclk|tr -d '\n') busy=$(cat $D/gpu_busy_percent) temp=$(cat $D/hwmon/hwmon*/temp1_input 2>/dev/null|head -1) cur_sclk=$(cat $D/hwmon/hwmon*/freq1_input 2>/dev/null|head -1) ppt=$(cat $D/hwmon/hwmon*/power1_average $D/hwmon/hwmon*/power1_input 2>/dev/null|head -1) lvl=$(cat $L)"; sleep 10; done ) >> $T/logs/$name.clk 2>&1 &
mon=$!
( while true; do for p in $(ls /sys/class/kfd/kfd/proc 2>/dev/null); do echo "$(date +%T) $p $(ps -o user=,comm=,args= -p $p 2>/dev/null | cut -c1-150)"; done; sleep 5; done ) >> $T/logs/$name.kfd 2>&1 &
kmon=$!
cd $T/${TREE:-src}/benchmarking/gemm_4bit_wmma
export PYTHONPATH=$T/${TREE:-src}:$T/${TREE:-src}/benchmarking/gemm_4bit_wmma:$T/bin
export RULES_HDR=${RULES_HDR:-$T/${TREE:-src}/csrc/gemm_4bit_wmma_rules.cuh}
export BNB_BENCH_SLEEP_CYCLES=${BNB_BENCH_SLEEP_CYCLES:-2e6}
bash ./runs_guard.sh $T/logs/$name.log "$@"
rc=$?
kill $mon $kmon
echo "$(date -u +%FT%TZ) end rc=$rc" >> $T/logs/$name.meta
exit $rc
