#!/bin/bash
# runq.sh <queuefile>: wait until no pytest of ours runs, then run the queue
while pgrep -u $USER -f "pytest tests" >/dev/null; do sleep 20; done
exec ~/bnbtune/queue.sh "$@"
