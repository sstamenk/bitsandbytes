#!/bin/bash
# q_kname.sh: launched kernel names of the base and final (relfin$KTAG) release builds (kname1103.py); runs inside job.sh.
. ~/bnbtune/env.sh
for d in rel relfin${KTAG:-1}; do (cd ~/bnbtune/$d && PYTHONPATH=$PWD python ~/bnbtune/bin/kname1103.py 2>&1 | grep -v USDT > ~/bnbtune/kname1103_$d.txt; cat ~/bnbtune/kname1103_$d.txt); done
echo "$(date -u +%T) done"
