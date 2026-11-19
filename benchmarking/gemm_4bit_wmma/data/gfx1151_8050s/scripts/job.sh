#!/bin/bash
# job.sh <name> <command...>: run a timing job at pinned clocks (high), restore auto on exit, guard for foreign GPU procs.
. ~/bnbtune/env.sh
name=$1; shift
mkdir -p ~/bnbtune/logs ~/bnbtune/data
L=/sys/class/drm/card1/device/power_dpm_force_performance_level
restore() { echo auto | sudo tee $L >/dev/null; echo "$(date -u +%FT%TZ) restored $(cat $L)" >> ~/bnbtune/logs/$name.meta; }
trap restore EXIT INT TERM
echo high | sudo tee $L >/dev/null
{ echo "$(date -u +%FT%TZ) start $name level=$(cat $L)"; echo "cmd: $*"; cat /sys/class/drm/card1/device/pp_dpm_sclk /sys/class/drm/card1/device/pp_dpm_mclk; } > ~/bnbtune/logs/$name.meta
# sample clocks/temps every 30 s
( while true; do echo "$(date +%T) sclk=$(grep '\*' /sys/class/drm/card1/device/pp_dpm_sclk|tr -d '\n') mclk=$(grep '\*' /sys/class/drm/card1/device/pp_dpm_mclk|tr -d '\n') gpu_busy=$(cat /sys/class/drm/card1/device/gpu_busy_percent) temp=$(cat /sys/class/drm/card1/device/hwmon/hwmon*/temp1_input 2>/dev/null|head -1)"; sleep 30; done ) >> ~/bnbtune/logs/$name.clk 2>&1 &
mon=$!
cd ~/bnbtune/src/benchmarking/gemm_4bit_wmma
export PYTHONPATH=~/bnbtune/src
bash ./runs_guard.sh ~/bnbtune/logs/$name.log "$@"
rc=$?
kill $mon
echo "$(date -u +%FT%TZ) end rc=$rc" >> ~/bnbtune/logs/$name.meta
exit $rc
