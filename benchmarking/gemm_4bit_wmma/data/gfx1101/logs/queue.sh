#!/bin/bash
# queue.sh <file>: run job lines (job.sh args) one at a time; lines starting with # skipped; done lines appended to <file>.done
f=$1
while true; do
  line=$(grep -v '^#' "$f" | grep -vxF -f <(cat "$f.done" 2>/dev/null) | head -1)
  [ -z "$line" ] && break
  echo "$(date -u +%T) START $line" >> ~/bnbtune/logs/queue.log
  eval ~/bnbtune/job.sh $line
  echo "$(date -u +%T) END rc=$? $line" >> ~/bnbtune/logs/queue.log
  echo "$line" >> "$f.done"
done
echo "$(date -u +%T) QUEUE EMPTY" >> ~/bnbtune/logs/queue.log
