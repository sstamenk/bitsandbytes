#!/bin/bash
# queue.sh <file>: run job lines (job.sh args, optionally prefixed by VAR=VAL env) one at a time; lines starting with #
# skipped; done lines appended to <file>.done. Re-reads the file after every job (append to extend).
. ~/bnbtune/env.sh
f=$1
while true; do
  line=$(grep -v '^#' "$f" | grep -v '^$' | grep -vxF -f <(cat "$f.done" 2>/dev/null) | head -1)
  [ -z "$line" ] && break
  echo "$(date -u +%FT%T) START $line" >> $T/logs/queue.log
  eval env $line
  echo "$(date -u +%FT%T) END rc=$? $line" >> $T/logs/queue.log
  echo "$line" >> "$f.done"
done
echo "$(date -u +%FT%T) QUEUE EMPTY" >> $T/logs/queue.log
