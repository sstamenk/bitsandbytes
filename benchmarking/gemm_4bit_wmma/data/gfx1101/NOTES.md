# gfx1101 (Radeon RX 7700 XT): measurements behind the gfx1101 tuning target

Code commit: `[ROCm] WMMA 4-bit GEMM: own tuning target for gfx1101` (WMMA_TARGET_GFX1101 = 111, own bf16/fp16 tile
table at M <= 64 and fp32 table at M <= 16, gfx110x SIMT/WMMA choice and block order; Python M limits bf16/fp16 448,
fp32 512). Before it, gfx1101 used the gfx1100 (W7900) rules and limits unmeasured.

## Machine and environment

- CS-UCICD-DT28 (OrchestrAI Playground, maas-mkm-cs/7cdcap), Ryzen 9 9950X3D, 64 GB, Ubuntu 24.04, kernel 7.0.0-43.
- Radeon RX 7700 XT (PCI 0x747e), gfx1101: 54 CUs (HIP multiProcessorCount 27), 12 GB GDDR6, 192-bit, 18 Gbps
  (mclk 1124 MHz) = 432 GB/s peak, 48 MB Infinity Cache, 4 MB L2. The 9950X3D's gfx1036 iGPU is HIP device 1: every
  run used `HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0` (gcnArchName checked: gfx1101), the clock level was set on
  `card2` (the dGPU).
- TheRock nightly 10.2.0a20260928: `rocm[libraries,devel,device-gfx1101]==10.2.0a20260928`, torch
  2.14.0+rocm10.2.0a20260928, `amd-torch-device-gfx1101` 2.14.0+rocm10.2.0a20260928 (HIP 7.17.26385), Python 3.12.
  Stock hipBLASLt (torch default BLAS on this build).
- Builds (Unix Makefiles, `-DBNB_ROCM_ARCH=gfx1101`): A = 1b6a36a, B = the gfx1101 commit; `-DBNB_WMMA_DEV=ON` for
  every timing build, release builds for the tests.
- Clocks: `power_dpm_force_performance_level=high` on card2 for every timing job, `auto` restored by a trap (logged
  in `logs/*.meta`). `high` is not a fixed clock: in the 30-s samples (`logs/*.clk`) sclk was 2028-2348 MHz while
  busy (median 2339, 5th percentile 2157), mclk always 1124 MHz, temperature 43-61 C.
- Every timing job ran serially from one queue (`logs/queue.sh`, `logs/q1.txt`, `logs/queue.log`) under
  `runs_guard.sh`; all 19 guarded jobs (timing jobs plus the overlay build, kernel-name and test jobs) report CLEAN,
  i.e. the periodic (5 s) check saw no GPU (KFD) process of another user; it does not rule out same-user activity
  or activity without a KFD process (none of ours ran in parallel: one serial queue). Builds and the correctness tests ran outside timing windows.

## Correctness

Release builds, `pytest tests/test_functional.py tests/test_linear4bit.py -k "(4bit or Linear4bit or gemv) and not
fsdp"`: A (1b6a36a) and the final code both 2423 passed / 22 failed, the identical 22 fp32 `test_gemv_4bit` cases
(separate GEMV kernel, also failing on A, i.e. pre-existing; root cause not investigated; no bf16/fp16 failure)
(`logs/pytest_rel.log`, `logs/pytest_final3.log` = final commit; `pytest_final.log`/`_final2.log` are the same suite on
earlier revisions of the M-limit and comment edits, same result). Profiler (`logs/kname_A_B.log`,
`logs/kname_final3.txt`): the final build launches `gemm_4bit_wmma<11, ...>` with the gfx1101 tiles where the tables
differ (fp16 M = 1/16/40, bf16 M = 40, fp32 M = 8/16 at 4096 x 4096), the gfx1100 tile at fp16 M = 128, and
`gemm_4bit_simt<float>` at fp32 M = 1. Relative error vs an fp32 reference: 3.3e-4 fp16, 2.6e-3 bf16, 1.1e-6 fp32.

## Files

| file | contents |
|---|---|
| `pinned/eval_inherited.csv` | A build: `auto`, `wmma`, `dequant` (56 M values 1-512) and `simt` (M <= 64), all dtypes, harness default sleep pre-roll. |
| `pinned/eval_final.csv` | B build, same kernels and M, sleep pre-roll scaled 0.1 (see below). Input of the plots. |
| `pinned/sweep_fp16.csv`, `sweep_fp32.csv` | Tile sweeps of every candidate (all 40 shapes; fp16 19 M values 1-384, fp32 16 M values 4-512). |
| `pinned/fp32_simt_wmma.csv`, `_final.csv` | Forced fp32 SIMT and WMMA, A build (M <= 64) and B build (M <= 32). |
| `pinned/alt/alt1_*.csv` | Interleaved A table vs candidate B2, fp16 and bf16, 17 M values 1-64, 5 cycles. |
| `pinned/alt/alt3_*.csv` | Interleaved A vs B2 vs B3 (the final bf16/fp16 table), M = 17-32, 5 cycles. |
| `pinned/alt/alt2_fp32.csv` | Interleaved A fp32 table vs C32 (the final fp32 table), M = 5-16, 5 cycles. |
| `pinned/ab/aba_*.csv` | Final A/B/A of `auto` with the two builds, all dtypes, 34 M values 1-512. |
| `pinned/cap/cap*_{fp16,bf16}.csv` | M limit: fused vs dequant + `F.linear` interleaved, M = 352-512 (21 values), 3 cycles. |
| `pinned/cap/cap_fp32_{lt,rocblas}.csv` | The same for fp32, M = 64-512, torch default BLAS (hipBLASLt) and rocBLAS. |
| `pinned/cap/cap_bs32.csv` | B build, blocksize 32, `auto` vs `dequant`, bf16/fp16, M = 384-512. |
| `pinned/ovl/*_dequant.csv` | dequant + `F.linear` with the ROCm/rocm-libraries#12465 hipBLASLt overlay (no effect here). |
| `pinned/r1/` | Astra round-1 checks: fp32 M = 128 paired repeats of two shapes (A/B builds) and the blocksize-32 cap run (`alt.py --blocksize 32`). |
| `pinned/rtf/rtf_*.csv` | B build, `BNB_WMMA_ROW_TILES_FIRST=0` vs `1` interleaved, fp32 and fp16, 3 cycles. |
| `analysis/` | Outputs of the scripts below (dispatch vs oracle, rule fits, interleaved comparisons, caps). |
| `scripts/` | `alt.py` (interleaved table comparison via BNB_WMMA_CFG), `rtf.py`, `fastsleep.py`, `fastsite/`, rule tables `rules/*.py` (gfx11_* = inherited, B2/B3/C32 candidates), analysis scripts. |
| `logs/` | Queue scripts (`q_*.sh`), per-job guard logs, clock samples, metadata, pytest logs, `blas_probe.txt`. |

Commands are in `logs/q_*.sh` (run by `logs/job.sh` from `logs/queue.sh`), e.g.

```bash
python bench.py --dtypes $dt --kernels auto|wmma|dequant --ms <56 values> ; python bench.py --kernels simt --ms <M <= 64>
BNB_GEMM4BIT_KERNEL=wmma python fastsleep.py sweep.py --dtype fp16 --shapes 0..39 --ms <19 values>
python derive_cfg3_safe.py sweep_fp16.csv --score total --merge 0 --feature auto    # scripts/fitbucket.sh per bucket
python fastsleep.py alt.py --tables A=rules/gfx11_f16.py,B3=rules/B3.py --dtype fp16 --ms ... --cycles 5
python ab_compare.py ab/aba_a1.csv ab/aba_b.csv ab/aba_a2.csv auto
```

Sleep pre-roll: `bench.timed_rotation` holds the GPU with `torch.cuda._sleep(5e7)` before queueing the timed
launches; on gfx1101 that is ~0.5 s per measurement (most of the run time: a 56-M eval of one kernel took 20 min).
Sweeps, interleaved runs, the A/B/A and the final eval scale it by 0.1 (`fastsleep.py`, `fastsite/sitecustomize.py`);
the timed window is unchanged. The A1 run (A build, scaled) against the inherited eval (A build, default, ~3 h
earlier): totals within approximately 0.61% at M <= 16, approximately 0.3-3.4% slower at M >= 128 (compute-bound sizes; per-shape p1-p99
0.99-1.05). Cause not isolated (pre-roll length, i.e. GPU power/thermal state, or session drift). Every decision
below compares configurations measured in the same run.

Repetitions: each bench.py value is the median of `min(max(40, copies), 200)` back-to-back timed calls (copies = the
number of weight copies rotated to exceed 384 MB); sweep.py, alt.py and rtf.py use `min(max(20, 2*copies//3), 120)`
calls per value. Interleaved runs repeat each configuration 5 cycles (tile tables) or 3 cycles (caps, block order) in
alternating order and use the median per shape; eval, sweeps and the build A/B/A are one pass per build.

## Findings

### Inherited rules (A build, `analysis/dispatch_vs_oracle_inherited.txt`)

Default dispatch vs the per-shape oracle min(simt, wmma, dequant) of the same run, total over the 40 shapes:
within 1.2% at M = 2-384 for every dtype except fp32 M = 4-5 (1.027-1.029) and M = 9-10 (1.018-1.030); bf16/fp16
M = 1 1.035-1.037 (WMMA 1.7-2.2% slower than SIMT in total); above the gfx110x caps 1.05-1.22 (bf16/fp16 M = 416-512)
and 3.2-5.1 (fp32 M = 160-512). The oracle only contains this build's own tiles: the sweep shows the tiles were the
problem (5% off the best swept tile at M <= 16, 9-11% at M = 20-64 for fp16, 8-10% fp32 M <= 16 on WMMA shapes;
`analysis/rules_vs_best_*`). The gfx115x table (an alternative to reuse) is worse at M >= 12 (1.06-1.27) but only
1.2-1.6% behind the new table at M <= 8 in the fp16 sweep (not validated interleaved or in bf16; the new M <= 8 bucket
is, so it was kept).

### 1. bf16/fp16 tiles: new table at M <= 64 (B3), gfx1100 tiles above

Fitted per bucket (M <= 8, 9-16, 17-32, 33-48, 49-64) on the fp16 sweep (`analysis/derive_buckets.txt`), then
edited for single-shape regressions: narrow-N leaves (N < 1280) at M = 9-16 and 49-64 keep deeper split-K (small k/v
and MoE gate_up were 1.16-1.38x slower with the plain fit); the M = 33-48 leaf for N >= 28672 (one shape gains, one
loses) was dropped; at M = 17-32 a K/N < 2/5 leaf (wide gate/up layers) replaces a narrow-N leaf (B2 -> B3). Swept-tile
totals vs the best tile per shape: M <= 8 1.010-1.012, 9-16 1.016-1.018, 17-32 1.017-1.020, 33-48 1.042-1.043, 49-64
1.022 (inherited 1.046-1.111). M = 65-512 kept: inherited within 0.4-2.6% of the best tile, refits gain < 2%.

Validation, all bf16 and fp16 separately (no prescale path below M = 96, so the two behave the same here):
- Interleaved in one process (`alt.py`, 5 cycles, alternating order, sum of per-shape medians), A table vs B2 at 17 M
  values including unswept bucket edges (3, 5, 9, 17, 33, 49, 56), and A vs B2 vs B3 at M = 17-32: B3/A = 0.960-0.969
  at M <= 16, 0.936-0.938 at 17-32, 0.938-0.940 at 33-48, 0.932-0.933 at 49-64; full-suite per-cycle ratio ranges
  within approximately 0.68 percentage points (bf16 M = 48). Over the sampled M of each changed bucket, the summed
  per-shape medians drop 3.5% (M <= 8 and 9-16), 6.3% (17-32), 6.1% (33-48) and 6.7% (49-64), fp16 and bf16 alike.
  B3 vs B2 at 17-32: 0.984-0.991.
- A/B/A of `auto` with the two builds (`analysis/final_aba.txt`, A1/A2 totals within approximately 0.53%): B 1.032-1.041x faster at
  M = 1-16, 1.063-1.073x at M = 17-64; M = 65-512 0.995-1.002 (unchanged tiles).
- Single shapes slower than with A by > 10% in the changed buckets (A/B/A): small GQA k/v (512 x 2048) 1.23-1.24x at M = 17-24 (14 -> 17 us)
  and 1.12x at 25-32; qwen3-30b-moe expert gate_up (1536 x 2048) 1.12-1.13x at M = 9-16; 7-8b k/v and
  qwen3-coder-480b expert down ~1.10x at M <= 8. Trade-off kept for B3 at M = 17-32: with B2 the small GQA k/v stays at
  A's time but llama3-8b and gemma3-12b gate/up are 1.17-1.19x slower (147 -> 172 us each); B3 is 1.1-1.6% faster in
  total. Three leaves per bucket do not allow both with the leaves tried (not a proof that no rule can).
- Excursions outside the changed buckets (A/B/A, fp32 M = 128): qwen3-14b o/q square 1.235x (approximately 1053 ->
  1300 us) and gemma3-4b gate/up 1.266x (1038 -> 1314 us) in the B pass. Both builds launch the same tile there
  (`gemm_4bit_wmma<11, float, 4,1,1,4,1,1, ...>`, `logs/kname_fp32m128_*.txt`); three paired passes A/B, B/A, A/B
  (`pinned/r1/fp32m128_*.csv`) give 1029.6-1030.4 us and 1019.4-1023.1 us for both builds, so these were transient
  single-shape timing excursions of that pass, not a rule effect. Build totals are stable to ~0.5%, single shapes
  below ~70 us or in one pass are not (see also the 1.0-1.11 worst-shape values of unchanged buckets).
- bf16/fp16 at M = 1: with the new M <= 8 tiles WMMA is 1.3-1.5% faster than SIMT in total (final eval), so the
  gfx110x choice (WMMA at every M) stays.
- Block order (`analysis/row_tiles_first.txt`, B build): fp16 row tiles first is 1.08-1.28x faster at M = 48-512
  (neutral at 24); fp32 neutral (0.995-1.001). gfx1101 keeps the gfx110x order (fp16 rows first, fp32 N first).

### 2. fp32 tiles: new table at M <= 16 (C32)

Fitted on the sweep restricted to the shapes the gfx110x cutoffs send to WMMA; a narrow-N leaf (N < 2816 at
M = 13-16) recovers 7-8b k/v and MoE shapes (1.23-1.48x with the plain fit). Interleaved A vs C32 (`alt2_fp32`, 5
cycles): 0.932-0.936 at M = 5-8 and 0.910-0.915 at M = 9-16 in 40-shape total (SIMT shapes added from the eval).
A/B/A: 1.067-1.071x at M = 5-8, 1.091-1.097x at M = 9-16; worst single shape 1.13-1.14x (7-8b k/v, M = 13-16, 52 ->
59 us). M = 17-24 kept (refit 0.2%), M = 25-48 kept (refit 3.0% at M = 32 but 1.9% at M = 48: below the bar for the
bucket).

### 3. fp32 SIMT/WMMA cutoffs: gfx110x table kept

Forced SIMT and WMMA on the B build (`analysis/fp32_cutoffs_final_tiles.txt`, `fit_fp32_cutoff_final.txt`): the
gfx110x table is within 1% of the per-shape faster kernel at every M <= 32 except M = 4 (1.031; best single cutoff
1.010, i.e. 2.1%, below the bar). The gfx1103/gfx115x/gfx12 tables are 0.1-0.7% worse in aggregate over M <= 32.

### 4. Python M limits

- fp32: 512 (gfx110x: 128). With the torch default BLAS (hipBLASLt) fp32 `F.linear` runs at ~1.4 TFLOP/s on this
  build (fp16 23-49 TFLOP/s, fp32 rocBLAS 6.6-7.5 TFLOP/s; `logs/blas_probe.txt`), so dequant + `F.linear` is 3-5x
  slower than the fused kernel in total at M = 64-512 and loses on all 40 shapes (`cap_fp32_lt`, interleaved). The
  #12465 hipBLASLt overlay does not remove the fp32 deficit: separate-run fp32 per-M totals with/without it
  differ by -0.5..+1.1% (`pinned/ovl`; single shapes and fp16 totals up to 5% apart, not attributed to the overlay
  from separate runs); its host-side cost is not the issue here. Even against rocBLAS (`cap_fp32_rocblas`) fused stays ahead in total through M = 512 (0.43-0.99;
  majority of shapes except 0.96 at 384 with 23/40 and 0.994 at 512 with 16/40). So fp32 512 is supported for the
  stock build; against rocBLAS it is a total-time-only near-tie at M = 512 (0.6%, minority of shapes), and a
  materially faster fp32 BLAS (e.g. a fixed hipBLASLt) would need the cap re-measured.
- bf16/fp16: 448 (gfx110x: 384). Fused time steps every 32 rows, dequant + BLAS around M = 400/416, so the crossover
  is jagged (`analysis/cap.txt`, interleaved, 3 cycles, 21 M values 352-512):
  - M = 385-400: near-tie (fused/dequant 0.987-1.004, 19-23 of 40 shapes ahead).
  - M = 401-448: fused 1.09-1.18x faster in total, ahead on 25-37 shapes (majority and total). The single-pass final
    eval has 21/40 (fp16) and 27/40 (bf16) at M = 448.
  - M = 449-480: 1.03-1.05x faster in total, ahead on 20-26 shapes (fp16 20/40 at 480).
  - M = 481-512: 0.97-0.99 in total but ahead on 16-20 shapes only (minority).
  - Blocksize 32: single pass (`cap_bs32`, B build) and interleaved (`pinned/r1/cap_bs32_alt_*`, 3 cycles,
    `analysis/cap_bs32_alt.txt`): at M = 416 fused/dequant 0.933 (fp16, 29/40) / 0.998 (bf16, 25/40); at M = 448
    1.004 (fp16, 14/40) / 1.070 (bf16, 8/40); at M = 480 1.064 / 1.130 (3-6/40). Single pass: bf16 already 1.05 at
    M = 384 (7/40), i.e. the gfx110x cap 384 also loses for bf16 at blocksize 32.
  Decision: 448, a conservative cap shared by both dtypes for blocksize 64, where total time and most shapes favour
  the fused kernel in the interleaved runs (fp16 0.907, 28/40; bf16 0.920, 27/40 at M = 448). Per dtype it is not the
  largest such cap: bf16 still wins 24-26/40 at M = 449-480 (0.967-0.975) and fp16 20-24/40 (0.950-0.960; 20/40 at
  480). 449-480 (3-5% in total, thin majority) is left to dequant + BLAS; cost 3-5% in total there versus a 480 cap.
  With blocksize 32, 448 is a compromise: even (fp16) and 1.07x slower (bf16) at M = 448.

### Final dispatch (B build, `analysis/dispatch_vs_oracle_final.txt`, per-shape oracle of the same run)

Within 1.8% of the oracle at every M up to 448 for every dtype except fp32 M = 4 (1.033, SIMT-vs-WMMA cutoff);
bf16/fp16 M = 1 1.010-1.012; bf16/fp16 at the measured M = 480/512 approximately 1.044-1.069 (above the cap; all
claims on the measured M grid). Speedup of the default dispatch over upstream
main (main = SIMT for M <= 8, dequant + F.linear above), total, final eval:
fp16 M = 1 1.02x, 2 1.21x, 4 1.78x, 8 3.08x, 16 7.28x, 32 5.83x, 64 3.23x, 128 1.99x, 256 1.29x, 384 1.13x, 448 1.10x;
bf16 within 0.15x of fp16; fp32 M = 8 1.82x, 16 5.59x, 32 4.39x, 64 3.69x, 128 4.85x, 256 3.22x, 384 5.12x, 512 3.16x
(fp32 M <= 4 stays SIMT as on main). These absolute speedups are from one run and carry the session-level drift noted
above (~1-3% at M >= 128).

## Notes

- `derive_cfg3.py` crashes when greedy merging joins M buckets without a common swept tile; the fits used the local
  copy `scripts/derive_cfg3_safe.py` from the gfx1200 round (never merges such buckets). The branch script is
  unchanged.
- `alt.py` forces each table's tile through BNB_WMMA_CFG with KCH lowered to 1 when K % 128 != 0. That is the
  launcher's behaviour for the tile, but it is not prescale-aware for fp16 at M >= 96 in general; here the changed
  buckets stop at M = 64, and in the cap runs the only K % 128 != 0 shapes (N = 2816, K = 704/2112) already get
  KCH_MAX = 1 from the inherited large-M table, so no comparison is affected.
- The 22 fp32 `test_gemv_4bit` failures are pre-existing (A build) and not investigated here.
- The table was calibrated on the 54-CU RX 7700 XT only; the RX 7800 XT (60 CUs, 256-bit, 64 MB) is also gfx1101 and
  is unmeasured.
