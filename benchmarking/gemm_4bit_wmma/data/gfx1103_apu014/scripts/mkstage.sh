#!/bin/bash
# mkstage.sh <dest>: assemble the data directory (data/gfx1103_apu014) from live/ (rsync of the machine's ~/bnbtune),
# the local scripts and NOTES.md, then regenerate analysis/ with the staged scripts.
set -e
H=$(cd "$(dirname "$0")" && pwd); L=${LIVE:-$H/live}; D=$1
rm -rf $D/pinned $D/logs $D/scripts $D/analysis $D/rorlab35
mkdir -p $D/pinned $D/logs/data $D/scripts/cand $D/rorlab35
for d in sweep alt altV cap evalA evalA_fix evalH r1; do mkdir -p $D/pinned/$d; cp $L/data/$d/*.csv $D/pinned/$d/; done
# final build fin3: interleaved runs (altF3 on the machine) and the dense eval (evalF3); earlier candidates through
# their own builds: fin2 (cand/S7.py) -> alt_s7, fin1 (cand/S1.py) -> alt_s1
mkdir -p $D/pinned/altF $D/pinned/alt_s7 $D/pinned/alt_s1 $D/pinned/evalF
cp $L/data/altF3/*.csv $D/pinned/altF/
cp $L/data/altF2/*.csv $D/pinned/alt_s7/
cp $L/data/altF/*.csv $D/pinned/alt_s1/
cp $L/data/evalF3/*.csv $D/pinned/evalF/
# fp32 dense eval: from the final build if it was run there, else from fin2 (the same fp32 tables and cutoffs)
[ -s $D/pinned/evalF/fp32_auto.csv ] || cp $L/data/evalF2/fp32_auto.csv $L/data/evalF2/fp32_wmma.csv $D/pinned/evalF/
rm -f $D/pinned/*/*.raw.csv
# the 5-cycle repeats of the blocksize / double-quantized spot checks, if they were run
if ls $L/data/altF3s5/*.csv >/dev/null 2>&1; then mkdir -p $D/pinned/altF_5c; cp $L/data/altF3s5/*.csv $D/pinned/altF_5c/; fi
for d in sweep alt altV altF altF2 altF3 altF3s5 cap r1 probe; do [ -d $L/data/$d ] || continue; mkdir -p $D/logs/data/$d; cp $L/data/$d/*.log $D/logs/data/$d/; done; cp $L/data/sweep/*.done $D/logs/data/sweep/
cp $L/logs/* $D/logs/
for f in $L/q1.txt.reset $L/q1.txt.failed; do [ -e $f ] && cp $f $D/logs/; done
cp $L/q1.txt $L/q1.txt.done $L/build_*.log $L/pytest_rel_*.log $L/kname*.txt $L/tilecheck_*.txt $L/pip.log $D/logs/
cp $H/data/sweep/sweep_fp16.csv $D/rorlab35/sweep_fp16_5shapes.csv
python3 $H/audit_csv.py $L --live --mtimes > $D/logs/csv_mtimes.txt  # last write of every CSV on the machine
for f in job.sh queue.sh env.sh build.sh pytest.sh q_sweep.sh q_sweep2.sh q_sweep3.sh q_sweep4.sh q_sweep5.sh q_alt1.sh q_alt2.sh q_alt32.sh q_rtf.sh \
  q_eval.sh q_evalF.sh q_altF.sh q_altB.sh q_altV.sh q_nest.sh q_fin2.sh q_fin3.sh q_r.sh q_cap.sh q_build.sh q_kname.sh q_probe.sh sweep1103.py alt1103.py alt1103b.py alt1103c.py rules1103.py rules_eval.py fit1103.py fit2.py \
  cmp_rules.py alt_bucket.py alt2_an.py alt32_an.py analyze.py cut_eval.py clk_summary.py dedup.py kname.py kname1103.py tilecheck1103.py \
  final_an.py cap_an.py cap_loss.py r1_an.py hi_an.py regress.py tile_split.py audit_csv.py fill_notes.py notes_values.py cells.py small_rows_fp16.py bandwidth_m1.py pickdump.cpp pickcheck.py analysis.sh mkstage.sh \
  rulesB.py rulesC.py rulesB2.py rules32B.py t16_f32.py; do cp $H/$f $D/scripts/; done
mkdir -p $D/scripts/old; cp $H/scripts_old/* $D/scripts/old/  # the versions earlier jobs ran with
cp $H/cand/S1.py $H/cand/S3.py $H/cand/S7.py $H/cand/S8.py $H/cand/V2.py $H/cand/V3.py $H/cand/F1.py $D/scripts/cand/
python3 $H/fill_notes.py $H/stage_NOTES.md $D/NOTES.md
bash $D/scripts/analysis.sh $D
tail -1 $D/analysis/csv_provenance.txt; grep -q "^0 flagged" $D/analysis/csv_provenance.txt || { echo "PROVENANCE AUDIT FAILED"; exit 1; }
rm -rf $D/scripts/__pycache__
du -sh $D; find $D -type f | wc -l
