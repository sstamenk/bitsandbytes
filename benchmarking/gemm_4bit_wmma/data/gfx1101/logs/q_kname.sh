#!/bin/bash
# kernel names / tiles on the A (src, 1b6a36a) and B (srcbrel, release) builds
for d in srcrel srcbrel; do echo "== $d"; (cd ~/bnbtune/$d && PYTHONPATH=~/bnbtune/$d python ~/bnbtune/kname.py 2>&1 | grep -v Warning); done
