#!/bin/bash
# run8060.sh <name> <cmd...> on hb2-mb (8060S): pinned perf level high on card0, auto restored on exit, guarded.
. ~/develop/bnb-wmma/venv/bin/activate
export ROCM_PATH=$(rocm-sdk path --root)
name=$1; shift
L=/sys/class/drm/card0/device/power_dpm_force_performance_level
restore() { echo auto | sudo tee $L >/dev/null; echo "$(date -u +%FT%TZ) restored $(cat $L)" >> ~/bnbtune8050/logs/$name.meta; }
trap restore EXIT INT TERM
echo high | sudo tee $L >/dev/null
{ echo "$(date -u +%FT%TZ) start $name level=$(cat $L)"; echo "cmd: $*"; } > ~/bnbtune8050/logs/$name.meta
( while true; do echo "$(date +%T) sclk=$(grep '\*' /sys/class/drm/card0/device/pp_dpm_sclk|tr -d '\n') busy=$(cat /sys/class/drm/card0/device/gpu_busy_percent) $(sudo cat /sys/kernel/debug/dri/0/amdgpu_pm_info 2>/dev/null | grep -m1 'SCLK\|average SoC' | tr -s ' ' | tr -d '\n')"; sleep 30; done ) >> ~/bnbtune8050/logs/$name.clk 2>&1 &
mon=$!
cd ~/bnbtune8050/src/benchmarking/gemm_4bit_wmma
export PYTHONPATH=~/bnbtune8050/src
bash ./runs_guard.sh ~/bnbtune8050/logs/$name.log "$@"
rc=$?
kill $mon
echo "$(date -u +%FT%TZ) end rc=$rc" >> ~/bnbtune8050/logs/$name.meta
exit $rc
