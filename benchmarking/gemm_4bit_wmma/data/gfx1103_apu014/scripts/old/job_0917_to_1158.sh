#!/bin/bash
# job.sh <name> <command...>: one timing job at perf level high on card1 (Radeon 760M); auto restored on exit.
# Tree: ${TREE:-src}. Logs in ~/bnbtune/logs: <name>.meta (start/end, level, sclk/mclk tables), <name>.clk (every 10 s:
# sclk/mclk level, busy, edge temp, current sclk, PPT), <name>.kfd (every KFD process every 5 s; ours are python),
# <name>.log (+ runs_guard CLEAN/CONTAMINATED for foreign users). The level is rewritten until sclk shows the pinned
# state; on a GPU reset during the job (dmesg count) the job is stopped, .meta says so and the job returns 99 (its
# logs are renamed <name>_reset<time> when the job is started again).
. ~/bnbtune/env.sh
name=$1; shift
D=/sys/class/drm/card1/device; L=$D/power_dpm_force_performance_level
mkdir -p $T/logs $T/data
restore() { echo auto | sudo -n tee $L >/dev/null; echo "$(date -u +%FT%TZ) restored $(cat $L)" >> $T/logs/$name.meta; }
trap restore EXIT INT TERM
resets() { sudo -n dmesg | grep -c "GPU reset begin"; }
if grep -q "GPU RESET DURING JOB" $T/logs/$name.meta 2>/dev/null; then
  s=$(date +%H%M%S); for e in meta clk kfd log log.foreign; do [ -e $T/logs/$name.$e ] && mv $T/logs/$name.$e $T/logs/${name}_reset$s.$e; done
fi
r0=$(resets)
for i in 1 2 3 4 5 6 7 8 9 10; do
  echo high | sudo -n tee $L >/dev/null
  sleep 2
  [ "$(cat $L)" = high ] && grep -q '^2: .*\*' $D/pp_dpm_sclk && break
  echo auto | sudo -n tee $L >/dev/null
  sleep 5
done
{ echo "$(date -u +%FT%TZ) start $name level=$(cat $L) tree=${TREE:-src} gpu_resets=$r0 pin_tries=$i"; echo "cmd: $*"; echo "users: $(who | awk '{print $1}' | sort -u | tr '\n' ' ')"; cat $D/pp_dpm_sclk $D/pp_dpm_mclk; } > $T/logs/$name.meta
( while true; do echo "$(date +%T) sclk=$(grep '\*' $D/pp_dpm_sclk|tr -d '\n') mclk=$(grep '\*' $D/pp_dpm_mclk|tr -d '\n') busy=$(cat $D/gpu_busy_percent) temp=$(cat $D/hwmon/hwmon*/temp1_input 2>/dev/null|head -1) cur_sclk=$(cat $D/hwmon/hwmon*/freq1_input 2>/dev/null|head -1) ppt=$(cat $D/hwmon/hwmon*/power1_average $D/hwmon/hwmon*/power1_input 2>/dev/null|head -1) lvl=$(cat $L)"; sleep 10; done ) >> $T/logs/$name.clk 2>&1 &
mon=$!
( while true; do for p in $(ls /sys/class/kfd/kfd/proc 2>/dev/null); do echo "$(date +%T) $p $(ps -o user=,comm=,args= -p $p 2>/dev/null | cut -c1-150)"; done; sleep 5; done ) >> $T/logs/$name.kfd 2>&1 &
kmon=$!
cd $T/${TREE:-src}/benchmarking/gemm_4bit_wmma
export PYTHONPATH=$T/${TREE:-src}:$T/${TREE:-src}/benchmarking/gemm_4bit_wmma:$T/bin
export RULES_HDR=${RULES_HDR:-$T/${TREE:-src}/csrc/gemm_4bit_wmma_rules.cuh}
export BNB_BENCH_SLEEP_CYCLES=${BNB_BENCH_SLEEP_CYCLES:-2e6}
setsid bash ./runs_guard.sh $T/logs/$name.log "$@" &
cmd=$!
( while sleep 10; do
    [ "$(resets)" = "$r0" ] && continue
    echo "$(date -u +%FT%TZ) GPU reset: stopping the job" >> $T/logs/$name.meta
    sleep 20; kill -TERM -- -$cmd; sleep 10; kill -KILL -- -$cmd; break
  done ) 2>/dev/null &
wd=$!
wait $cmd
rc=$?
r1=$(resets)
kill $mon $kmon $wd 2>/dev/null
echo "$(date -u +%FT%TZ) end rc=$rc gpu_resets=$r1" >> $T/logs/$name.meta
[ "$r1" != "$r0" ] && { echo "$(date -u +%FT%TZ) GPU RESET DURING JOB: discard" >> $T/logs/$name.meta; sleep 60; rc=99; }
exit $rc
