#!/bin/bash
# correctness on the final release build (no timing)
. ~/bnbtune/env.sh; cd ~/bnbtune/srcrelb
{ echo "== kname (final release build srcrelb)"; PYTHONPATH=$PWD python ~/bnbtune/kname.py 2>&1 | grep -v USDT; echo "== tilecheck"; PYTHONPATH=$PWD python ~/bnbtune/tilecheck.py 2>&1 | grep -v -i "USDT\|warn"; } > ~/bnbtune/logs/correctness_final.log 2>&1
