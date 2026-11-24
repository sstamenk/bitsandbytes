#!/bin/bash
# q_build.sh <tag>: build the final code: ~/bnbtune/fin<tag> (-DBNB_WMMA_DEV=ON) and ~/bnbtune/relfin<tag> (release) =
# the base tree src/ with the files under ~/bnbtune/final_files<tag>/ (paths relative to the repository root) copied
# over; then the pytest subset and the launched kernel names on the release build. Runs inside job.sh so no timing
# job overlaps.
. ~/bnbtune/env.sh
tag=$1; F=~/bnbtune/final_files$tag
[ -f $F/csrc/gemm_4bit_wmma_rules.cuh ] || { echo "missing $F"; exit 1; }
for d in fin$tag relfin$tag; do
  rm -rf ~/bnbtune/$d; mkdir ~/bnbtune/$d && rsync -a --exclude build --exclude '*.so' --exclude __pycache__ ~/bnbtune/src/ ~/bnbtune/$d/ && rsync -a $F/ ~/bnbtune/$d/
done
(cd $F && find . -type f | sort | while read f; do md5sum $F/$f ~/bnbtune/src/$f ~/bnbtune/fin$tag/$f ~/bnbtune/relfin$tag/$f; done)
bash ~/bnbtune/build.sh fin$tag 1 || exit 1
bash ~/bnbtune/build.sh relfin$tag 0 || exit 1
bash ~/bnbtune/pytest.sh relfin$tag pytest_rel_final$tag.log; tail -2 ~/bnbtune/pytest_rel_final$tag.log
cd ~/bnbtune/relfin$tag && PYTHONPATH=$PWD python ~/bnbtune/bin/kname.py > ~/bnbtune/kname_rel_final$tag.txt 2>&1; cut -c1-90 ~/bnbtune/kname_rel_final$tag.txt
PYTHONPATH=$PWD python ~/bnbtune/bin/tilecheck1103.py > ~/bnbtune/tilecheck_rel_final$tag.txt 2>&1; echo "tilecheck rc=$?" >> ~/bnbtune/tilecheck_rel_final$tag.txt; tail -4 ~/bnbtune/tilecheck_rel_final$tag.txt
(cd ~/bnbtune/rel && PYTHONPATH=$PWD python ~/bnbtune/bin/tilecheck1103.py > ~/bnbtune/tilecheck_rel_base.txt 2>&1; echo "tilecheck rc=$?" >> ~/bnbtune/tilecheck_rel_base.txt; tail -4 ~/bnbtune/tilecheck_rel_base.txt)
echo "$(date -u +%T) done"
