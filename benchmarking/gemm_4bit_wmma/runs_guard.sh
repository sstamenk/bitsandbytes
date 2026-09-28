#!/bin/bash
# runs_guard.sh <log> <command...>: run a timing job while logging any GPU (KFD) process not owned by $USER.
# Example: ./runs_guard.sh ev.log ./par_bench.sh ev "0 1 2 3" 0,1,2 wmma
log=$1; shift
: > "$log.foreign"
( while true; do
    rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+/ {print $1}' | while read -r pid; do
      u=$(ps -o user= -p "$pid" 2>/dev/null)
      [ -n "$u" ] && [ "$u" != "$USER" ] && echo "$(date +%T) $pid $u $(ps -o comm= -p "$pid")" >> "$log.foreign"
    done
    sleep 5
  done ) &
mon=$!
"$@" > "$log" 2>&1
rc=$?
kill $mon
if [ -s "$log.foreign" ]; then echo "CONTAMINATED: foreign GPU processes seen (see $log.foreign)" | tee -a "$log"; else echo "CLEAN" | tee -a "$log"; fi
exit $rc
