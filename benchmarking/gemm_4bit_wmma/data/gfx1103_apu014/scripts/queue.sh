#!/bin/bash
# queue.sh <file>: run job lines (job.sh args, optionally prefixed by VAR=VAL env) one at a time; lines starting with #
# skipped; done lines appended to <file>.done. Re-reads the file after every job (append to extend).
# A job that returns 99 (GPU reset, see job.sh) is run again, at most twice; 98 (clock not pinned) stops the queue;
# any other nonzero status is logged in <file>.failed and the queue goes on.
. ~/bnbtune/env.sh
f=$1
while true; do
  line=$(grep -v '^#' "$f" | grep -v '^$' | grep -vxF -f <(cat "$f.done" 2>/dev/null) | head -1)
  [ -z "$line" ] && break
  echo "$(date -u +%FT%T) START $line" >> $T/logs/queue.log
  eval env $line
  rc=$?
  echo "$(date -u +%FT%T) END rc=$rc $line" >> $T/logs/queue.log
  [ $rc = 98 ] && { echo "$(date -u +%FT%T) QUEUE STOPPED: clock not pinned" >> $T/logs/queue.log; exit 98; }
  n=$(grep -cxF -- "$line" "$f.reset" 2>/dev/null)
  [ $rc = 99 ] && [ "${n:-0}" -lt 2 ] && { echo "$line" >> "$f.reset"; continue; }
  [ $rc != 0 ] && echo "rc=$rc $line" >> "$f.failed"
  echo "$line" >> "$f.done"
done
echo "$(date -u +%FT%T) QUEUE EMPTY" >> $T/logs/queue.log
