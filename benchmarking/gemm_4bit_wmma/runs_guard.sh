#!/bin/bash
# runs_guard.sh <log> <command...>: run a timing job while logging any GPU (KFD) process not owned by $USER.
# Example: ./runs_guard.sh ev.log ./par_bench.sh ev "0 1 2 3" 0,1,2 wmma
# A long-lived idle foreign process (e.g. a vLLM server on other GPUs) can be allowed with
# GUARD_ALLOW_PIDS="<pids>" GUARD_IDLE_GPUS="<rocm-smi indices>": the run is still flagged if any other foreign process
# appears or those GPUs go above 10% use.
log=$1; shift
: > "$log.foreign"
( while true; do
    # KFD lists every process with a GPU context; rocm-smi --showpids is the fallback.
    { if [ -d /sys/class/kfd/kfd/proc ]; then ls /sys/class/kfd/kfd/proc
      else rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/ {print $1}'; fi; } | while read -r pid; do
      u=$(ps -o user= -p "$pid" 2>/dev/null)
      [ -n "$u" ] && [ "$u" != "$USER" ] && ! [[ " $GUARD_ALLOW_PIDS " == *" $pid "* ]] &&
        echo "$(date +%T) $pid $u $(ps -o comm= -p "$pid")" >> "$log.foreign"
    done
    # Allowed (idle) foreign processes must stay idle: rocm-smi GPU indices in GUARD_IDLE_GPUS above 10% use.
    for g in $GUARD_IDLE_GPUS; do
      use=$(rocm-smi -d "$g" --showuse 2>/dev/null | awk '/GPU use/ {print $NF}')
      [ -n "$use" ] && [ "$use" -gt 10 ] && echo "$(date +%T) rocm-smi GPU $g busy ${use}%" >> "$log.foreign"
    done
    sleep 5
  done ) &
mon=$!
"$@" > "$log" 2>&1
rc=$?
kill $mon
if [ -s "$log.foreign" ]; then echo "CONTAMINATED: foreign GPU processes seen (see $log.foreign)" | tee -a "$log"; else echo "CLEAN" | tee -a "$log"; fi
exit $rc
