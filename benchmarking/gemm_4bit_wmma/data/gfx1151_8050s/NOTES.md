# gfx1151 Radeon 8050S (32 CUs): does the gfx115x tuning (fitted on the 40-CU 8060S) fit?

Result: the inherited gfx115x rules are kept unchanged (no code change). The fp32 cutoffs and fp32 tiles (only
M = 24 swept) need no change; the Python M limits stay at 512, for fp32 as an explicit aggregate-time tradeoff (most
shapes lose to dequant+BLAS above M = 256). The 8050S prefers other bf16/fp16 tiles at the measured M >= 17: an
8050S-fitted table B cuts total time by 2.8-9.3% (fp16) and 1.3-12.2% (bf16), but costs the 8060S 3-40% (fp16) and
2-15% (bf16), re-measured on hb2-mb. No tested shared candidate (J2, J3) stayed within ~2-3% of each device's best
per M bucket and recovered B's single-shape losses, and a CU-count based target split was dropped (review: the WGP
count is not a robust SKU identifier). Details below.

## Machines
- 8050S: CS-WSPG-APU005 (OrchestrAI Playground, maas-mkm-cs/4sdfc8), AMD "MAPLE" board, Ryzen AI MAX PRO 390,
  Radeon 8050S gfx1151, 32 CUs (HIP multiProcessorCount 16), max sclk 2799 MHz, 2 MB L2, 32 MB MALL, 32 GB
  LPDDR5X-8000 256-bit = 256 GB/s peak, 512 MB carve-out, platform_profile balanced. Ubuntu 24.04, kernel
  7.0.0-43-generic.
- 8060S (reference, data/gfx1151 and the hb2-mb re-measurement): HB2-MB, Radeon 8060S gfx1151, 40 CUs
  (multiProcessorCount 20), max sclk 2900 MHz, 128 GB LPDDR5X-8000 (256 GB/s), 512 MB carve-out, kernel 6.18.
- Both: TheRock nightly 10.2.0a20260928, torch 2.14.0+rocm10.2.0a20260928, amd-torch-device-gfx1151 2.14.0, stock
  hipBLASLt (16x2048x2048 fp32 F.linear 0.09 ms on the 8050S). bitsandbytes 1b6a36a, -DBNB_ROCM_ARCH=gfx1151, Unix
  Makefiles; -DBNB_WMMA_DEV=ON for timing (tile override), release build for the tests.

## Clocks / method
Every timing job: perf level `high` (restored to `auto` by an EXIT trap), one job at a time from a queue, wrapped in
runs_guard.sh (all CLEAN), sclk/temperature sampled every 30 s (`logs/*.clk`, `*.meta`). NF4, blocksize 64, 40 shapes,
weights streamed from DRAM (bench.py / sweep.py methodology).
- 8050S under `high`: 2.4-2.8 GHz in the 30-s samples (`logs/*.clk`). Unarchived interactive observations during
  ev_fp32 (transcribed here): a 20-s probe of pp_dpm_sclk / hwmon power1_input every 0.2 s showed mostly 2.76-2.78
  GHz at 55-56 W and intervals of 2.15-2.18 GHz at 71-72 W, and amdgpu_pm_info read 2524 MHz at 65 W "average SoC
  including CPU". The 8060S showed 2900 MHz in every active 30-s sample of its runs. Hypothesis, not isolated: a
  package power limit around 70 W on this board is part of why the fused kernel is 1.3-1.7x slower on the 8050S at
  compute-bound M (more than the 1.30x CU x clock ratio).
- Block-to-block drift on the 8050S: separate runs of the same build differed by up to 5-15% in total at M >= 96
  (aba1 a1 vs a2, profexp first pass). Within a run it is stable (`auto` vs forced `wmma`, same dispatch: within 1%
  total, 2-5% per shape). Tile decisions therefore rest on the in-process interleaved comparison (alt.py): the same
  weight copies and activations, configurations alternating in forward/reverse order (B always in the middle).
  Repetitions: 8050S fp16 5 cycles for shapes 0-22 and 3 for shapes 23-39 (totals = sum of per-shape medians; cycle
  ranges use the 3 full-suite cycles only); 8050S bf16 and bf16 cap check 2 cycles; 8060S fp16 5, bf16 3 cycles.
  Spread of the full-suite per-cycle total ratios: <= 0.8% on the 8050S fp16, up to 2.2% on the 8060S (B/A at
  M = 512). This is a spread, not a confidence interval; residual order effects are possible.

## Files (pinned/)
| file | contents |
|---|---|
| dense/ev_fp16.csv | auto, wmma, dequant fp16, 56 M (1-512), inherited rules |
| dense/ev_fp32.csv | auto, wmma, dequant fp32, 26 M (1-512) |
| dense/ev_simt.csv | simt fp16 + fp32, M = 1-10, 12, 14, 16, 24, 32 |
| dense/ev_bf16.csv | auto, dequant bf16, 9 M |
| sweep_fp16.csv | tile sweep fp16: M = 4, 8 (shapes 0-13) and 16, 24, 48, 96, 192, 384 (all 40) = part1 (stopped during shape 13 to save time) merged with part2 (shapes 13-39, replaces shape 13) |
| sweep_fp32.csv | tile sweep fp32, M = 24 |
| ab/aba1_*, aba2_*, final_* | block A/B/A runs of `auto` (A = gfx115x table; B = 8050S table, aba: srcb build, final: a CU-split build, since dropped). Drift-confounded, superseded by alt/ |
| alt/alt_fp16.csv | interleaved A / B / J2 tiles, fp16, 10 M, 40 shapes (part1: shapes 0-22, 5 cycles; part2: 23-39, 3 cycles) |
| alt/alt_bf16.csv | interleaved A / B / J2, bf16, M = 24, 48, 96, 128, 256, 2 cycles |
| alt/alt_cap_bf16.csv | interleaved A / B / J2 / dequant+F.linear, bf16, M = 384, 512, 2 cycles |
| alt/alt60_fp16.csv, alt60_bf16.csv | the same interleaved fp16 (10 M, 5 cycles) / bf16 (8 M, 3 cycles) comparison on the 8060S (hb2-mb, logs/hb2-mb/) |
| profexp/ | 6 tiles x 8 tall-K shapes x M = 24/96/384 and auto/dequant, platform_profile balanced -> performance -> balanced |
Tables: A = wmma_rules_gfx115_f16 (shipped), B = 8050S fit (scripts/rulesB.py), J2 = joint fit over both devices'
sweeps with each device's time normalized by its own per-shape-best total (scripts/rulesJ2.py), J3 = J2 with A's
M = 17-24 bucket. logs/: job logs, guard verdicts, clocks, queue timeline, pytest logs, kernel traces.
scripts/: the queue/job wrappers and every analysis script used (derive_cfg3_safe.py = derive_cfg3.py plus a guard
against merging buckets that share no swept tile).

## Correctness (release build, 1b6a36a)
`pytest tests/test_functional.py tests/test_linear4bit.py -k "(4bit or Linear4bit or gemv) and not fsdp"`:
2423 passed, 22 failed, all fp32 test_gemv_4bit (the separate GEMV kernel; this is the base code, not caused by any
change here), no fp16/bf16 failure. The dropped CU-split build gave the identical 22 failures. Kernel trace (rocprofv3):
gemm_4bit_wmma<11, ...> for fp16/bf16 at M = 1/16/128 and fp32 at M = 16/128, gemm_4bit_simt<float> at fp32 M = 1.

## Findings
1. Default dispatch vs per-shape oracle (min over the measured kernels; SIMT only up to M = 32, bf16 only auto and
   dequant), total over the 40 shapes: fp16 within 0.6% at M = 1-416, 1.010-1.021 at 448-512; fp32 within 1.6% at
   M <= 2 and 6-32, 1.062-1.065 at M = 3-4 (8060S: 1.063 at M = 3), 1.03 at 48-128, 1.04-1.05 at 384-512.
2. 8050S vs 8060S (same kernel, shape, M; 8060S data/gfx1151/pinned/dense/eval_final.csv): memory-bound region
   equal (fp16 auto total 0.98-1.00 at M <= 16; single shapes 0.73-1.18, the spread of single-pass data); fp16
   dequant+BLAS 0.92-1.08; fused fp16 1.09x at M = 32, 1.28x at 64, 1.54x at 128, 1.60-1.66x at 192-512 (~20 vs
   ~32 TFLOP/s at M = 512); fused fp32 1.26-1.31x at M = 1-8 and 1.5-2.0x above; fp32 dequant+BLAS 0.57-0.71x in
   total (the slow large-N fp32 hipBLASLt GEMMs take 250 vs 509 ms on llama3-70b gate/up at M = 384).
3. fp32 SIMT/WMMA cutoffs: the gfx115x table is within 0.6% of the best single N cutoff at every M up to 32 except
   M = 3 (all SIMT, 1.036x vs N >= 28672; the 8060S's data has the same 1.026x gap). Kept.
4. fp32 tiles: only M = 24 swept: gfx115x table 1.028 vs best swept tile, best fit 1.026. Kept (other M unswept).
5. Python M limits, all kept at 512:
   - fp16 (inherited tiles): fused/dequant total 0.71 at M = 256, 0.83 at 384, 0.93 at 512, ahead on 36, 31, 23/40.
   - bf16 (interleaved, inherited tiles, M = 384 and 512 only, 2 cycles): 0.82 at M = 384 (34/40), 0.93 at 512
     (25/40; each order separately 0.928 / 0.924; worst shape 1.24x). The single ev_bf16 run had 1.01 at 512 (19/40),
     consistent with drift or the harness difference (separate processes there).
   - fp32: total 0.36-0.39 at M = 192-512, but fused is ahead on only 12-13/40 shapes at M >= 320 (25 at 192, 23 at
     256): an aggregate-time tradeoff dominated by 12 large-N shapes where fp32 dequant+BLAS is 3-5x slower. The 8060S
     wins on 25/40 at 384-512. Not changed (a lower cap would cost ~2.6x in total on those shapes).
6. bf16/fp16 tiles. Sweep: the gfx115x table is 1.03-1.06 off the best swept tile at M = 4-16, 1.175 at 24, 1.095 at
   48, 1.098 at 96, 1.050 at 192, 1.069 at 384 (8060S on its own pinned sweep: 1.050, 1.050, 1.034, 1.014, 1.016 at
   24-384). At M = 4-8 both units pick the same tiles; from M = 16 they differ: for tall-K shapes at M = 17-24 the
   8060S's ksplit-8 tile {1,2,1,1,8,2} is 1.21-1.48x off on the 8050S. The platform profile does not change the
   ranking (profexp). Interleaved timing, total over 40 shapes, ratio to A (lower = faster):

   | M | 8050S fp16 B | J2 | 8060S fp16 B | J2 | 8050S bf16 B | J2 | 8060S bf16 B | J2 |
   |---|---|---|---|---|---|---|---|---|
   | 17 | 0.916 | 0.906 | 1.093 | 1.074 | | | 1.052 | 1.037 |
   | 24 | 0.917 | 0.905 | 1.091 | 1.080 | 0.918 | 0.906 | 1.058 | 1.043 |
   | 40 | 0.955 | 0.973 | 1.034 | 0.993 | | | 1.027 | 0.989 |
   | 48 | 0.956 | 0.972 | 1.035 | 0.994 | 0.956 | 0.971 | 1.030 | 0.987 |
   | 96 | 0.924 | 0.967 | 1.108 | 1.013 | 0.987 | 0.997 | 1.030 | 1.029 |
   | 128 | 0.907 | 0.964 | 1.058 | 0.989 | 0.983 | 0.993 | 1.024 | 1.019 |
   | 192 | 0.972 | 0.993 | 1.404 | 1.028 | | | 1.147 | 1.024 |
   | 256 | 0.926 | 0.984 | 1.275 | 1.020 | 0.878 | 0.988 | 1.057 | 1.023 |
   | 384 | 0.948 | 0.974 | 1.139 | 1.007 | | | | |
   | 512 | 0.934 | 0.977 | 1.077 | 0.982 | | | | |

   B has repeatable single-shape losses on the 8050S fp16: phi3-mini down 1.35x at M = 40-48, 7-8b k/v 1.45x at 128,
   gemma4-26b-moe gate_up 1.29x at 96; J2 removes the last two but keeps phi3-mini down at 1.34-1.35x (M = 40-48) and
   adds gemma4-26b-moe gate_up 1.34x at M = 17-24.
   Decision: no change. B would need a separate target (it costs the 8060S up to 1.40x); a CU-count selector
   (multiProcessorCount <= 16, tried as a commit and dropped) is not a robust SKU identifier (WGP mode, CU masks,
   the 16-CU 8040S). No tested shared candidate met the criterion (within ~2-3% of each device's best per M bucket
   and recovering the listed regressions): J2 is 2-6% off B on the 8050S fp16 at M >= 40 (12% at bf16 M = 256) and
   7.4-8.0% slower than A on the 8060S fp16 at M = 17-24; J3 (J2 with A's M = 17-24 bucket) is 4.6-6.3% off B on the
   8050S at M = 96/128/256/512 and keeps the phi3-mini down loss. J3's aggregate fp16 time reduction on the 8050S is
   2.16% over the 10 measured M values (sample-weighted; 3.3-3.6% at M = 96/128), at 0.14% in aggregate and up to
   2.8% (M = 192) on the 8060S. This does not show that no shared table could work; none of the tested ones did.
7. Speedup of the default dispatch over main (SIMT M <= 8, else dequant+F.linear), total, inherited rules: fp16 M = 1
   1.15x, 4 1.93x, 8 3.22x, 16 6.29x, 32 4.68x, 64 2.96x, 128 2.04x, 256 1.40x, 384 1.20x, 512 1.07x; fp32 M = 4
   1.14x, 8 1.94x, 16 7.84x, 32 4.61x, 64 2.84x, 128 2.90x, 256 2.77x, 512 2.58x.

The tile-ranking difference reproduces on these two systems under `high` (the 8060S re-measured with the same
interleaved harness confirms its old sweep). Possible contributors: CU count, clocks / power policy, and other
platform differences (kernel 7.0 vs 6.18, memory size); switching platform_profile balanced/performance on the 8050S
changed neither the ranking nor the timings.

Plots: plots/gfx1151_8050s (256 GB/s peak), from dense/ev_fp16, ev_fp32, ev_simt (inherited rules; no bf16 wmma/simt
runs, so no bf16 per-shape plot).
