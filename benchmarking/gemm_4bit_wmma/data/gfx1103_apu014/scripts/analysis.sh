#!/bin/bash
# analysis.sh [root]: regenerate analysis/*.txt from the CSVs. root = this data directory (default: the parent of
# scripts/): pinned/{sweep,alt,altV,alt_s1,alt_s7,altF,r1,cap,evalA,evalA_fix,evalF,evalH}, logs/, rorlab35/. RULES_HDR = the rules header whose
# tables are scored (default: csrc/gemm_4bit_wmma_rules.cuh of the repository this directory is in).
H=$(cd "$(dirname "$0")" && pwd); R=$(cd "${1:-$H/..}" && pwd); O=$R/analysis; mkdir -p $O
export RULES_HDR=${RULES_HDR:-$R/../../../../csrc/gemm_4bit_wmma_rules.cuh}
S=$R/pinned/sweep; A=$R/pinned/alt; ROR=$R/rorlab35/sweep_fp16_5shapes.csv
cat1() { awk 'FNR==1 && NR>1 {next} {print}' "$@"; }
cat1 $S/sweep_fp16a.csv $S/sweep_fp16b.csv $S/sweep_fp16c.csv > $O/.fp16.csv
cat1 $S/sweep_bf16a.csv $S/sweep_bf16b.csv > $O/.bf16.csv
cat1 $S/sweep_fp32a.csv $S/sweep_fp32b.csv > $O/.fp32lo.csv
T16="g1100_f16 g1101_f16 g1102_f16 g1150_f16 g1151_f16 $H/rulesB2.py $H/cand/S1.py $H/cand/S3.py g1103_f16"
T32="g1100_f32 g1101_f32 g1102_f32 g1150_f32 g1151_f32 g1103_f32"
cd $H
python3 rules_eval.py $O/.fp16.csv $T16 --buckets 8,16,32,48,64,96 > $O/rules_vs_best_fp16.txt
python3 rules_eval.py $O/.bf16.csv $T16 --buckets 8,16,32,48,64,96 > $O/rules_vs_best_bf16.txt
{ echo "# M <= 48: sweeps fp32a + fp32b, all 40 shapes forced to WMMA"; python3 rules_eval.py $O/.fp32lo.csv $T32 --buckets 8,16,24,48
  echo "# M <= 48: only the shapes the gfx1103 cutoffs send to WMMA"; python3 rules_eval.py $O/.fp32lo.csv $T32 --fp32wmma --buckets 8,16,24,48
  echo "# M = 64, 128, 256: sweep fp32d (every tile, also under 32 rows)"; python3 rules_eval.py $S/sweep_fp32d.csv $T32
  echo "# M = 64, 128, 256: sweep fp32a (as sweep.py: only tiles with 32 rows or more at M >= 64)"; python3 rules_eval.py $S/sweep_fp32a.csv g1100_f32; } > $O/rules_vs_best_fp32.txt 2>&1
python3 rules_eval.py $ROR $T16 --buckets 8,16,32,48,64,96 > $O/rorlab35_rules_vs_best_fp16_5shapes.txt
python3 cmp_rules.py --apu $O/.fp16.csv --bf $O/.bf16.csv --ror $ROR $H/rulesB2.py $H/cand/S1.py $H/cand/S3.py g1103_f16 > $O/two_machines_fp16.txt
{ for ms in 1,2,4,8 12,16 20,24,28,32 40,48 56,64 80,96; do echo "#### M $ms"; python3 fit2.py $O/.fp16.csv $ROR --ms $ms --w 1,0.75,0.5 --leaves 2 | awk '!s[$0]++'; done; } > $O/two_machines_fit_fp16.txt
python3 small_rows_fp16.py $S/sweep_fp16c.csv $S/sweep_fp16e.csv > $O/fp16_m80_small_tiles.txt
python3 bandwidth_m1.py $S/sweep_fp16a.csv $S/sweep_bf16a.csv > $O/bandwidth_m1.txt
{ for dt in fp16 bf16; do echo "== alt1 $dt: A = gfx1100 table, B = rulesB.py, C = rulesC.py (B = rulesB2.py at M <= 64, C at M <= 48)"; python3 alt_bucket.py $A/alt1_$dt.csv A B,C 8,16,32,48,64,96; python3 alt2_an.py $A/alt1_$dt.csv A B C; done; } > $O/alt1.txt
{ for dt in fp16 bf16; do echo "== alt2 $dt: A = gfx1100 table, B = rulesB2.py (fit on this machine alone), S = cand/S1.py, T = cand/S3.py (S = the final table except at M = 17-32 and M = 65-96)"; python3 alt_bucket.py $A/alt2_$dt.csv A B,S,T 8,16,32,48,64,96; echo "-- S and T against B"; python3 alt_bucket.py $A/alt2_$dt.csv B S,T 8,16,32,48,64,96; echo "-- S against A with the final buckets"; python3 alt_bucket.py $A/alt2_$dt.csv A S 8,16,32,48,96; python3 alt2_an.py $A/alt2_$dt.csv A B S T; done; } > $O/alt2.txt
{ echo "== alt32 (fp32, every shape forced): S = SIMT, A = gfx1100 tiles, B = rules32B.py (= the final table at M <= 48), D = gfx1150 tiles"; python3 alt_bucket.py $A/alt32.csv A B,D 8,16,24,48; python3 alt2_an.py $A/alt32.csv A B D S
  echo "== dispatch (cutoff table + tiles) vs the per-shape oracle over SIMT and B"; python3 alt32_an.py $A/alt32.csv; } > $O/alt32.txt
{ echo "== block order, fp32: A0/A1 = gfx1100 tiles N-first / row-tiles-first, T0/T1 = tile 2,2,1,4,1,2"; python3 alt2_an.py $A/rtf_fp32.csv A0 A1 T0 T1
  echo "== block order, fp16: A1 = row-tiles-first (default), A0 = N-first, gfx1100 tiles"; python3 alt2_an.py $A/rtf_fp16.csv A1 A0; } > $O/block_order.txt
F=$R/pinned/altF; S7=$R/pinned/alt_s7; S1=$R/pinned/alt_s1; R1=$R/pinned/r1
pick() { [ -s $F/$1.csv ] && echo $F/$1.csv || echo $S7/$1.csv; }  # final build if the run exists there, else the S7 build
lab() { [ -s $F/$1.csv ] && echo "final build" || echo "build fin2 (cand/S7.py; this run is not affected by the M = 65-96 row unless it has M = 65-96)"; }
{ for dt in fp16 bf16; do echo "== final build (fin3), $dt: A = gfx1100 table (base dispatch), F = default dispatch of the final build"; python3 alt_bucket.py $F/$dt.csv A F 8,16,32,48,64,96; python3 alt2_an.py $F/$dt.csv A F
    echo "-- X = the final table (cand/S8.py) through the override (must match F)"; python3 alt_bucket.py $F/${dt}_x.csv F X 8,16,32,48,64,96
    echo "-- S = forced SIMT at M = 1, 2"; python3 alt2_an.py $F/${dt}_m1.csv F S; done
  echo "== fp32, $(lab fp32_lo): A = gfx1103 cutoffs + gfx1100 tiles (base dispatch), F = default dispatch"
  python3 alt_bucket.py $(pick fp32_lo) A F 2,4,8,16,24,48; python3 alt2_an.py $(pick fp32_lo) A F
  echo "-- M = 256-512, final build, 5 cycles (pinned/r1/fp32_af.csv)"; python3 alt_bucket.py $R1/fp32_af.csv A F 48; python3 alt2_an.py $R1/fp32_af.csv A F
  echo "-- M = 256, 384, 512, 3 cycles, $(lab fp32_hi)"; python3 alt2_an.py $(pick fp32_hi) A F
  echo "-- X = the final fp32 table (cand/F1.py) through the override (must match F), $(lab fp32_x)"; python3 alt2_an.py $(pick fp32_x) F X
  echo "-- D = gfx1103 cutoffs + gfx1150 fp32 tiles, $(lab fp32_d)"; python3 alt2_an.py $(pick fp32_d) F D; } > $O/final_interleaved.txt 2>&1
python3 regress.py $F/fp16.csv $F/bf16.csv $(pick fp32_lo) $R1/fp32_af.csv > $O/final_regressions.txt
{ python3 tile_split.py g1103_f16 $F/fp16.csv $F/bf16.csv
  if [ -s $R1/st_fp16.csv ]; then echo "== the two same-tile cells above 1.02 again (pinned/r1/st_*.csv: shapes gemma3-27b down and gemma3-4b gate/up, M = 28 and 56, 10 cycles)"
    for dt in fp16 bf16; do echo "-- $dt"; python3 cells.py $R1/st_$dt.csv A F; done; fi; } > $O/final_same_tile_cells.txt
{ echo "== other blocksizes and double-quantized absmax, 3 cycles (spot checks): A = base dispatch through overrides, F = default dispatch"
  for x in bs32_fp16 bs32_bf16 bs128_fp16 bs32_fp32 nest_fp16 nest_bf16 nest_fp32; do echo "-- $x, $(lab $x)"; python3 alt2_an.py $(pick $x) A F; done
  if [ -d $R/pinned/altF_5c ]; then echo "== the same on the final build with 5 cycles (pinned/altF_5c)"
    for x in bs32_fp16 bs32_bf16 bs128_fp16 bs32_fp32 nest_fp16 nest_bf16 nest_fp32; do [ -s $R/pinned/altF_5c/$x.csv ] && { echo "-- $x"; python3 alt2_an.py $R/pinned/altF_5c/$x.csv A F; }; done; fi; } > $O/final_blocksize_nested.txt
python3 r1_an.py $R1/r1_fp16.csv $R1/r1_bf16.csv > $O/r1_m49_96.txt
{ echo "== M = 17-32, blocksize 64, 5 cycles (build fin2, the final tile at these M): A = gfx1100 table, B = rulesB2.py, F = default dispatch"
  for dt in fp16 bf16; do echo "-- $dt"; python3 alt_bucket.py $R1/abf_$dt.csv A B,F 32; python3 alt_bucket.py $R1/abf_$dt.csv F B 32; done; } > $O/fit_vs_final_m17_32.txt
V=$R/pinned/altV
{ echo "== M = 17-32 at blocksize 32 (base build, 5 cycles): A = gfx1100 table, S = cand/S1.py, V2 = cand/V2.py, V3 = cand/V3.py (= the final bucket), B = rulesB2.py"
  for x in v32_fp16 v32_bf16; do echo "-- $x"; python3 alt_bucket.py $V/$x.csv A S,V2,V3,B 32; python3 alt2_an.py $V/$x.csv A S V2 V3 B; done
  echo "== the same at blocksize 64: A, S, V3"; for x in v64_fp16 v64_bf16; do echo "-- $x"; python3 alt_bucket.py $V/$x.csv A S,V3 32; python3 alt2_an.py $V/$x.csv A S V3; done
  echo "== bf16, blocksize 32, the other buckets: A, S"; python3 alt2_an.py $V/b32_bf16.csv A S
  echo "== double-quantized absmax, blocksize 64 (base build, 3 cycles): A, S, V3; fp32: A = gfx1103 cutoffs + gfx1100 tiles, F = the same cutoffs + cand/F1.py"
  for x in nest_fp16 nest_bf16; do echo "-- $x"; python3 alt2_an.py $V/$x.csv A S V3; done; echo "-- nest_fp32"; python3 alt2_an.py $V/nest_fp32.csv A F; } > $O/m17_32_blocksize_nested.txt
{ echo "== earlier candidate cand/S7.py (the final table with one tile for M = 49-96) through its own build (fin2): A = gfx1100 table, F = default dispatch of fin2"
  for dt in fp16 bf16; do echo "-- $dt"; python3 alt_bucket.py $S7/$dt.csv A F 8,16,32,48,64,96; done
  echo "== earlier candidate cand/S1.py through its own build (fin1): A = gfx1100 table, F = default dispatch of fin1"
  for dt in fp16 bf16; do echo "-- $dt"; python3 alt_bucket.py $S1/$dt.csv A F 8,16,32,48,96; done
  echo "-- blocksize 32, fp16 (3 cycles)"; python3 alt2_an.py $S1/bs32_fp16.csv A F; echo "-- blocksize 128, fp16"; python3 alt2_an.py $S1/bs128_fp16.csv A F
  echo "== fp32 on build fin1 (the same fp32 tables and cutoffs as the final build): A = gfx1103 cutoffs + gfx1100 tiles, F = default dispatch"
  echo "-- fp32_lo (rerun after the GPU reset, queue job altF2)"; python3 alt_bucket.py $S1/fp32_lo.csv A F 2,4,8,16,24,48
  echo "-- fp32_lo, the 13 shapes done before the GPU reset"; python3 alt_bucket.py $S1/fp32_lo_part13_before_gpu_reset.csv A F 2,4,8,16,24,48
  echo "-- fp32_hi, fp32_x, fp32_d: run after the GPU reset in the same job (the level high no longer held the clock at every sample)"
  python3 alt2_an.py $S1/fp32_hi.csv A F; python3 alt2_an.py $S1/fp32_x.csv F X; python3 alt2_an.py $S1/fp32_d.csv F D
  echo "-- bs32_fp32 (blocksize 32, 3 cycles)"; python3 alt2_an.py $S1/bs32_fp32.csv A F; } > $O/earlier_candidates.txt
{ echo "== blocksize 64: bf16/fp16 5 cycles, fp32 3 cycles (build fin1; the tiles at these M are the final ones)"; python3 cap_an.py $R/pinned/cap/cap_fp16.csv $R/pinned/cap/cap_bf16.csv $R/pinned/cap/cap_fp32.csv
  echo "== fp32, M = 384-512, 5 cycles (final build, pinned/r1/fp32_fq.csv)"; [ -s $R1/fp32_fq.csv ] && python3 cap_an.py $R1/fp32_fq.csv
  P39=$R1/fp32_fq_part39_before_gpu_reset.csv
  { cat $P39; grep '^llama3-70b gate/up,' $R1/fp32_fqb.csv; } > $O/.fq40.csv; { head -1 $P39; grep '^llama3-70b down,' $P39; } > $O/.fq_s1a.csv; { head -1 $P39; grep '^llama3-70b down,' $R1/fp32_fqb.csv; } > $O/.fq_s1b.csv
  [ -s $R1/fp32_fq256.csv ] && { echo "== fp32, M = 256, 5 cycles (final build, pinned/r1/fp32_fq256.csv)"; python3 cap_an.py $R1/fp32_fq256.csv; }
  echo "== fp32, the first run of it (ended by a GPU reset after 39 shapes) + llama3-70b gate/up from fp32_fqb.csv (a later job, shapes 0 and 1)"; python3 cap_an.py $O/.fq40.csv
  echo "-- the first run alone (39 shapes)"; python3 cap_an.py $P39
  echo "-- llama3-70b down: first run, then fp32_fqb"; python3 cap_an.py $O/.fq_s1a.csv $O/.fq_s1b.csv
  echo "== blocksize 32, M = 384-512, 5 cycles (build fin2, pinned/r1/cap_bs32_*.csv)"; python3 cap_an.py $R1/cap_bs32_fp16.csv $R1/cap_bs32_bf16.csv
  echo "== blocksize 32, M = 384 and 512, 3 cycles (build fin1)"; python3 cap_an.py $R/pinned/cap/cap_bs32_fp16.csv $R/pinned/cap/cap_bs32_bf16.csv; } > $O/cap.txt
python3 cap_loss.py $R1/cap_bs32_fp16.csv $R1/cap_bs32_bf16.csv $R/pinned/cap/cap_fp16.csv $R/pinned/cap/cap_bf16.csv > $O/cap_single_shape_losses.txt
[ -d $R/pinned/evalH ] && python3 hi_an.py $R/pinned/evalH > $O/beyond_512.txt
python3 final_an.py $R/pinned $R/pinned > $O/dispatch_base_final.txt
E=$R/pinned/evalA
cat1 $E/fp32_simt.csv $E/fp32_wmma.csv > $O/.f32sw.csv; python3 cut_eval.py $O/.f32sw.csv > $O/fp32_cutoffs_base_tiles.txt
python3 clk_summary.py $R/logs/*.clk > $O/clocks.txt
{ python3 audit_csv.py $R; echo "== queue jobs that did not return 0"; grep -E "END rc=[1-9]" $R/logs/queue.log | cut -c1-140; } > $O/csv_provenance.txt
for f in $R/logs/*.kfd; do echo "$(basename $f): $(awk '{print $3}' $f | sort | uniq -c | tr '\n' ' ')"; done > $O/gpu_processes.txt
rm -f $O/.*.csv
