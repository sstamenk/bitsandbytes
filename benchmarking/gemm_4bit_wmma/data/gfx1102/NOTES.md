# gfx1102 (Radeon RX 7600 XT) measurements

## Machine and versions

- OrchestrAI Playground CS-UCICD-DT199 (Markham, maas-mkm-cs/4egwn4), Ubuntu 24.04, kernel 7.0.0-43-generic,
  Threadripper PRO 3975WX, 512 GB. The RX 7600 XT is the only AMD GPU (card2; card1 is the BMC's ASPEED VGA).
- Radeon RX 7600 XT (Navi 33 XT, gfx1102, PCI 1002:7480 rev c0, Gigabyte 1458:2418, VBIOS 113-APM7076OC-102):
  32 CUs (HIP multiProcessorCount 16), 16 GB GDDR6 128-bit at mclk 1124 MHz (18 Gbps) = 288 GB/s peak, 32 MB
  Infinity Cache, 2 MB L2, PCIe 4.0 x8. (The inventory tag says vram_8gb; the card has 16 GB, `mem_info_vram_total`.)
- TheRock nightly `rocm[libraries,devel,device-gfx1102]==10.2.0a20260928`, `torch 2.14.0+rocm10.2.0a20260928`,
  `amd-torch-device-gfx1102` (same nightly), triton 3.8.0, Python 3.12.3, cmake 3.31.10, uv 0.13.9.
- Builds: `cmake -G "Unix Makefiles" -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH=gfx1102 [-DBNB_WMMA_DEV=ON]`.
  A = 1b6a36a (gfx1102 on the gfx1100 rules), B = the gfx1102 target commit. Dev builds for timing, release builds for
  pytest and the correctness checks.
- Clocks: `power_dpm_force_performance_level=high` on card2 for every timing job, `auto` restored on exit
  (`scripts/job.sh`). `high` is a performance level, not fixed clocks: in the 30-s samples with the GPU busy
  (`logs/*.clk`, per job in `analysis/clock_ranges.txt`) sclk medians are 2571-2587 MHz for the eval, sweeps, alt and
  cap jobs but 2477-2478 MHz for the fp32 A/B/A/B jobs (`aba2`, `aba3`: 2303-2590 MHz, none in 2582-2589), 2539 for
  `aba`, 2395 for `extra` and 1883 for `m1b` (12 samples, 1697-2586 MHz); mclk 1124 MHz in every sample; temperature
  at most 61 C. The paired/interleaved comparisons are the evidence for decisions for that reason. Every timing job
  ran under `runs_guard.sh` (`logs/*.log` end in CLEAN, `.foreign` empty). The guard polls KFD every 5 s and only
  sees other users' processes; same-user overlap was excluded by running everything from one serial queue (`scripts/queue.sh`,
  `scripts/queues.txt`, `logs/queue.log`); builds and pytest ran between the guarded timing windows. Exceptions,
  all discarded: two aborted jobs at 13:08-13:09 and 17:29 (a stray dequant worker and a 2-s probe overlapped the
  start of a sweep / the first `m1` run; those outputs were deleted and the jobs rerun as `sweep2` and `m1b`).
- Harness change: `bench.timed_rotation` idles the GPU with `torch.cuda._sleep(5e7)` before each timed batch so the
  launches queue up; on this GPU that is 0.5 s per measurement, which dominated run time (20 min per kernel and
  dtype). From the sweeps on, jobs set `BNB_BENCH_SLEEP=5e6` (~0.05 s; a launch costs ~11 us, so 120 launches queue
  in ~1.3 ms). The patch (`scripts/bench_py_sleep_patch.txt`) only changes the idle time before the timed window. The
  first sweep part (shapes 0-18, long sleep) and the second (shapes 19-39, short sleep) overlap on shape 19
  (gemma3-4b gate/up, M = 1-20): median ratio of the 412 common (M, tile) times 1.0006.

## Files (NF4, blocksize 64, 40 shapes)

| file | contents |
|---|---|
| `pinned/dense/eval_inherited.csv` | A build: `auto`, `wmma` (fp16, fp32), `dequant` at 56 M values 1-512, `simt` at M <= 64; fp16/bf16/fp32. bf16 `wmma` was not run (identical to `auto`: WMMA at every M). Long idle sleep. |
| `pinned/dense/eval_final.csv` | Final: `auto` = mean of `ab/*_b1.csv` and `*_b2.csv` (61 M values); `wmma` = that `auto` for bf16/fp16, fp32 forced WMMA from `ab/fp32_wmma_b.csv` (M <= 64) and the A build above M = 24 (same tiles); `simt`, `dequant` from `eval_inherited.csv` (unchanged paths). Built by `scripts/final_an.py`. |
| `pinned/sweep_fp16.csv` | Tile sweep (`sweep.py`, every candidate), 23 M values 1-512. Shapes 0-18 from the first run, 19-39 from the rerun (see above). |
| `pinned/sweep_fp32.csv` | fp32 tile sweep, M = 4, 8, 12, 16, 24, 32, 48, 64, 128, 256 (trimmed for time). |
| `pinned/fp32_simt_wmma.csv` | Forced fp32 SIMT and WMMA, M <= 64 (A build). |
| `pinned/m1/` | SIMT vs WMMA, forced per worker process, A build: fp16/bf16 at M = 1-2, fp32 at M = 3-6, 6 cycles alternating order. |
| `pinned/alt/alt1_{fp16,bf16}.csv` | In-process interleaved tile tables (`scripts/alt.py`, `BNB_WMMA_CFG`): A = gfx1100 table, B = gfx1102 candidate (`scripts/rulesB.py` = final), 19 M values 1-65 incl. bucket edges, 5 cycles alternating order. |
| `pinned/alt/alt32.csv` | The same for fp32: A, B32 (M = 9-24 refitted), C32 (B32 + M <= 8), 13 M values 4-32. |
| `pinned/alt/cap_{fp16,bf16}.csv` | Fused (WMMA) vs dequant + `F.linear` in one process, M = 352-512, 3 cycles. |
| `pinned/ab/{fp16,bf16}_{a1,b1,a2,b2}.csv` | Final A/B/A/B of `auto`, 61 M values (dense + 17, 31, 33, 49, 65). The B binary had the draft fp32 table (identical bf16/fp16 tables). |
| `pinned/ab/fp32_{a1,b1,a2,b2}.csv`, `fp32_wmma_b.csv` | The same for fp32 with the final build (rebuilt; `aba3`); forced WMMA of the final build, M <= 64. |
| `pinned/ab/draft_fp32_*.csv` | fp32 A/B/A/B of the first B binary: the same fp32 tables as final (`aba`). |
| `pinned/ab/reverted_fp32_*.csv` | fp32 A/B/A/B with M = 17-24 reverted to the gfx11 tiles (`aba2`; see the fp32 tile section). |
| `pinned/ab/simt_c*.csv`, `autob_c*.csv` | SIMT (A) and final `auto` at M = 1-3, fp16/bf16: three repeated process pairs, always SIMT first. |
| `pinned/review1/` | Review round 1 (`scripts/alt2.py`, in-process, kernel and tile chosen per call): `r1_fp32_m4` fp32 M = 4 cutoff old/new (5 cycles); `r2_bs32_*` blocksize-32 fused vs dequant at M = 448/512 (3 cycles); `r3_reuse_*` final vs gfx1101 vs gfx115x tables at M = 1/8 (5 cycles); `r3_fp32` fp32 M = 17/24 gfx11 / final / gfx115x / gfx1101 tiles and SIMT (5 cycles). |
| `pinned/extra/` | Block order (`BNB_WMMA_ROW_TILES_FIRST=0/1`, final dev build, fp16/fp32, M = 48-512, 2 alternating process cycles) and a blocksize-32 `bench.py` run (auto, dequant, M = 352-512). |
| `pinned/artifact/` | First-in-process check (below). |
| `pinned/dense/dequant_*_hipblaslt_overlay.csv` | dequant + `F.linear` with the hipBLASLt #12465 overlay. |
| `analysis/` | Outputs of the scripts below (rules vs best tile, fits, alt/A-B summaries, cutoffs, oracle). |
| `logs/` | Guard logs, clocks (`.clk`), job metadata (`.meta`), queue log, pytest logs, `correctness_final.log`, `blascheck.out` (BLAS backend and fp32/fp16 `F.linear` speed, stock and with the overlay). |
| `scripts/` | Queue/job scripts, `alt.py`, rule tables (`rules11.py`, `rulesB.py`, `rulesB32.py`, `rulesC32.py`), analysis scripts (as run, then reformatted by pre-commit: ruff fixes, typos-safe names; behaviour unchanged). The analysis scripts expect the work-directory layout they ran in (`data/eval/<dtype>_<kernel>.csv`, `data/aba*/`, `data/alt/`, ...), i.e. the per-run CSVs before they were merged/renamed here; `final_an.py` writes `data/eval_final.csv`. |

## Findings

"Total" is the sum over the 40 shapes at one M; "oracle" the per-shape minimum over simt, wmma and dequant.

### Correctness

`pytest tests/test_functional.py tests/test_linear4bit.py -k "(4bit or Linear4bit or gemv) and not fsdp"`, release
builds: A build 2445 passed, 0 failed (`logs/pytest_rel.log`); final build 2445 passed, 0 failed
(`logs/pytest_final.log`; again on the exact final commit after a docstring-only edit, `logs/pytest_final3.log`). No fp32 `test_gemv_4bit` failure on this card (other archs had 22), and the 16 GB card runs
`test_4bit_quant_large` (gfx1200, 8 GB: OOM). Profiler (`scripts/kname.py`, final release build, 4096 x 4096):
`gemm_4bit_wmma<11, half, 1, 1, 1, 1, 4, 2, ...>` at M = 1, `<..., 2, 1, 1, 4, 1, 2, ...>` at M = 16, the prescaled
`<..., 4, 2, 1, 4, 1, 2, 4, true>` at M = 128, bf16 the same tiles, fp32 SIMT at M = 1 and the new fp32 M <= 16 tile
`<..., 2, 2, 1, 4, 1, 2, ...>` at M = 16. `scripts/tilecheck.py` (final release build, default dispatch vs a float64
reference; 10 (N, K) pairs covering every leaf incl. N < 1024, K % 128 != 0 and K >= 14336; 20 M values 1-512;
blocksize 32/64/128; nested and not; three dtypes; a logged numerical check: it asserts finiteness, no tolerance): max
relative error 3.9e-4 (fp16), 3.2e-3 (bf16), 8.7e-7 (fp32), all finite (`logs/correctness_final.log`). All 40 shapes fit in 16 GB for every kernel (largest: fp32 dequant of
28672 x 8192 = 940 MB); none skipped.

### Tile rules: own target WMMA_TARGET_GFX1102

bf16/fp16, against the best swept tile (`analysis/rules_vs_best_fp16.txt`, total):

| M | 1-8 | 12-16 | 20-32 | 40-48 | 56-64 | 80-96 | 112-128 | 160-512 |
|---|---|---|---|---|---|---|---|---|
| gfx1100 table | 1.097-1.127 | 1.133-1.137 | 1.184-1.188 | 1.097-1.099 | 1.082-1.087 | 1.034-1.040 | 1.019 | 1.004-1.007 |
| gfx115x table | 1.045-1.081 | 1.096-1.098 | 1.188-1.306 | 1.230-1.233 | 1.075-1.076 | 1.127-1.129 | 1.146-1.147 | 1.068-1.174 |
| gfx1102 (final) | 1.019-1.025 | 1.020-1.021 | 1.036-1.037 | 1.052-1.053 | 1.023-1.024 | kept | kept | kept |

Buckets M = 65-96 and above keep the gfx1100 tiles: by the sweep a joint refit would gain 1.5% (80-96) and under 1%
above. These are sweep estimates and not prescale-complete: the sweep has no KCH = 2 candidates for K % 128 != 0 and
`rules_eval.py` lowers such rule tiles to KCH = 1, while at fp16 M >= 96 the launcher picks the prescaled kernel from
KCH_MAX (2 of 40 shapes affected); the changed buckets (M <= 64) have no prescaling. The gfx115x
(8060S) table is no closer than gfx1100's; the gfx12 tables use 16-row tiles and do not apply. Fits:
`analysis/derive_fp16_buckets.txt` (one joint fit per bucket, leaves on N or K). The M <= 8 bucket uses K leaves
(`K < 3840`, `K < 14336`; on N the worst shape was 2.96x the best tile). The N < 1024 leaves at M = 17-48 were added
for small GQA k/v (512 x 2048), which the fitted bucket put on a tile 1.40-1.45x (M = 20-32) and 1.25x (40-48) slower
than its own best.

gfx1101 reuse (sibling tune/gfx1101, 16b858b): its fp16 table looked up in the gfx1102 sweep is 1.02-1.05x the final
table at M = 1-8, 1.15x at 12-16, 1.13-1.14x at 20-32, 1.06x at 40-64 (computed by the reviewer from
`pinned/sweep_fp16.csv`), so the whole table is not reusable. In-process at M = 1 and 8 (`analysis/review1.txt`,
R3, 5 cycles) the gfx1101 table is 1.017-1.022x and the gfx115x table 1.020-1.025x slower than the final M <= 8
bucket in total, in fp16 and bf16. A gfx1102 table is needed for the other buckets anyway, so copying another chip's
M <= 8 row would cost ~2% without removing a table; the fitted row is kept. (Either row would still be 5-7% faster
than the gfx1100 row.) gfx1103 has no tile table of its own (gfx1100 tiles).

In-process interleaved (`analysis/alt1_*.txt`, 5 cycles, cycle-total ranges within approximately 0.9%), B/A total:
fp16 0.928-0.937 (M = 1-8), 0.896-0.903 (9-16), 0.874-0.880 (17-32), 0.957-0.962 (33-48), 0.939-0.942 (49-64), 1.000
(65); bf16 0.927-0.937, 0.896-0.903, 0.875-0.879, 0.958-0.963, 0.941-0.943, 1.000.

Builds A/B/A/B of the default dispatch (`analysis/aba.txt`; A2/A1 and B2/B1 within approximately 0.72% at every M; fp32 from the final build, `aba3`), B/A total,
llama3-70b gate/up excluded at M <= 2 only where noted below:

| | M = 1-8 | 9-16 | 17-32 | 33-48 | 49-64 | 65-512 |
|---|---|---|---|---|---|---|
| fp16 | 0.924-0.938 | 0.895-0.903 | 0.869-0.875 | 0.953-0.962 | 0.944-0.946 | 0.999-1.002 |
| bf16 | 0.924-0.936 | 0.895-0.905 | 0.870-0.875 | 0.951-0.963 | 0.943-0.947 | 0.996-1.002 |
| fp32 | 0.998-1.001, M = 4: 0.944 | 0.900-0.907 | 0.970-0.971 (17-24), 1.000-1.001 (25-32) | 1.000-1.001 | 1.000 | 0.997-1.002 |

Every changed bucket gains at least ~3% in total in each dtype (fp32 M = 17-24 the smallest, 2.9-3.0%). Single shapes slower than with the gfx1100 table
(final A/B, bf16 and fp16 alike): qwen3-14b o/q (5120 x 5120) 1.17x at M = 8 (67.9 -> 79.4 us in the sweep), small
GQA k/v 1.12-1.14x at M = 9-16 and 1.13-1.15x at 49-64 (tens of microseconds), gemma3-4b gate/up 1.11-1.13x at
M = 17-32, llama3-8b o/q 1.12-1.13x at M = 33-48, gemma4-26b dense gate_up 1.09-1.10x at M = 1-6. No shape is more
than 1.2x slower from the tile rules (the M-limit change has larger single-shape losses, listed there). fp32: gemma3-4b gate/up 1.07-1.08x at M = 9-16; the sweep's small GQA k/v regression (1.62x at
M = 9-16 with forced WMMA) never runs, because fp32 N < 1024 stays on SIMT through M = 64.

fp32 tiles (`analysis/rules_vs_best_fp32.txt`, `analysis/alt32.txt`): the gfx11 table was 11.7-11.9% off the best
swept tile at M = 12-16 (refit 0.9%), 4.6% at M = 24, 5.0% at M <= 8, 2.1-3.2% at 32-48, within 0.2% at 64-256.
Refitted M = 9-16 and 17-24. 17-24 is the marginal one: in-process 3.0-3.1% faster over the shapes that dispatch to
WMMA (`alt32`), 3.0-3.2% in total in the build A/B/A/B (`analysis/aba.txt`, draft section, all 40 shapes) and 2.9-3.0%
(B/A 0.970-0.971) in the final A/B/A/B (`aba3`). (On the way it was reverted once: a first evaluation left llama3-70b
gate/up out at every M instead of only M <= 2 and read 2.4%; `ab/reverted_fp32_*` is that build, B/A 0.999-1.001 at
17-24 as expected.) Review R3 (`analysis/review1.txt`, in-process, 5 cycles, N < 1024 on SIMT for every table as
dispatched): at M = 17/24 the final bucket is 3.1% faster than the gfx11 tiles (B/A 0.969, no shape slower than
1.00x); the gfx115x fp32 bucket would be 0.7% slower than the final one, i.e. only 2.4% better than the gfx11 tiles
(below the bar) and 1.13x slower on 7-8b k/v; the gfx1101 fp32 bucket there is the gfx11 one. Kept: an accepted
~3% gain (2.9-3.2% in four measurements). M <= 8 (C32): 2.9-3.3% at M = 5-8 but 0.1% at M = 4 (WMMA shapes only): kept on the gfx11 tiles.

Row tiles first: kept as on gfx11 (bf16/fp16 row tiles first, fp32 N first); the sweeps and fits used that order.
Checked with the final build (`analysis/extra.txt`, `BNB_WMMA_ROW_TILES_FIRST`, 2 alternating process cycles, M = 48-512):
N-first is 1.08-1.42x slower for fp16 (1.08-1.09 at M = 64-128, 1.16-1.18 at 192-512) and within 1% for fp32
(0.999-1.009), so the gfx11 choice stands.

### SIMT/WMMA choice

fp32 (`analysis/fp32_cutoffs.txt`): with the gfx11 cutoffs the dispatch was 6.4-6.8% above the per-shape faster kernel
at M = 4 (SIMT for every N there); the gfx1102 table switches N >= 8192 to WMMA at M = 4 (0.4%), measured on the
inherited eval, on 6 alternating m1 cycles (gfx11 1.064, new 1.004) and with the final tiles (1.065 -> 1.004).
In-process (review R1, `analysis/review1.txt`, same allocations, 5 cycles alternating, warm-up pass): new/old 0.942 in
total (cycles 0.940-0.945), faster on 22 shapes, worst 1.03x (qwen3-14b gate/up). Other
rows unchanged: within 0.9% of the oracle at every M <= 64 with the final tiles (largest M = 12; a full refit gains
at most 0.8%). The gfx1103 table (WMMA from M = 3) is ~1.18x off at M = 3 (m1 medians and final tiles; the inherited eval reads
1.447 because its forced-WMMA llama3-70b gate/up at M = 3 was a startup-contaminated 3726 us vs 1166 us in the final
run), the gfx115x and gfx12 tables 1.05-1.07 at M = 4.

bf16/fp16 at M = 1 stays WMMA. With the gfx1100 tiles SIMT was 2-3% faster in total at M = 1 (eval: SIMT 1.022, WMMA
1.056 of the oracle); with the gfx1102 tiles WMMA is ~5% faster than SIMT (`analysis/simt_vs_wmma_m1_final.txt`,
three repeated SIMT-then-auto process pairs: SIMT/WMMA 1.049, WMMA 1.003 of the oracle, SIMT ahead on 10/39 shapes; at
M = 2 SIMT is 1.25-1.27x).

### First-in-process artifact

The first one to five measurements of a bench worker process (always llama3-70b gate/up at M = 1-3, the first shape)
are sometimes ~1.44 ms instead of 0.52-0.63 ms (`pinned/artifact/`: in 7 of 8 auto workers, and 0.91-0.97 ms in 2 of 8
SIMT workers;
the later repeats of the same (shape, M) in the same process are normal). It hit both builds and both kernels and is not
a property of a tile. Cause not isolated (hypothesis: a power/clock state at process start). It is in the raw data
(eval bf16 M = 1, `ab/bf16_*` M = 1-2, `m1/`, `eval_final.csv`); the dispatch comparisons above and in
`analysis/dispatch_vs_oracle.txt` leave llama3-70b gate/up out at M = 1-2 (39 shapes);
`analysis/aba_with_first_shape_artifact.txt` has the unfiltered A/B.

### Python M limits

bf16/fp16 (gfx1100: 384) -> 512. Fused vs dequant + `F.linear` total: 0.71-0.84 at M = 385-512 with 24-36 of 40
shapes ahead (`analysis/cap_*.txt`, in-process, 3 cycles: fp16 0.714/0.706/0.744/0.791/0.836 at M =
385/416/448/480/512, 36/36/31/28/24 shapes ahead; bf16 0.717/0.710/0.749/0.797/0.838, 33/34/32/31/25); the eval and the
final A/B agree (0.705-0.838). At blocksize 64 majority of shapes and total time agree at every M up to 512. With the 384 cap the default dispatch
was 1.26-1.42x off the oracle at M = 416-512.

Single-shape losses of the cap change (fused / dequant, per-shape cycle medians, `pinned/alt/cap_*`), > 20%: none at
M = 448; M = 480 fp16 gemma4-26b-moe expert down 1.25x, qwen3-30b-moe expert down 1.22x; bf16 qwen3-30b-moe expert
down 1.29x, gemma4-26b-moe expert down 1.28x, gemma4-26b dense down 1.23x; M = 512 fp16 gemma4-26b-moe expert down
1.38x (104 -> 143 us), gemma4-26b-moe expert gate_up 1.30x, qwen3-30b-moe expert down 1.26x, gemma4-26b dense down
1.24x, qwen3.5-122b-moe expert gate_up 1.23x, qwen3.5-35b-moe expert gate_up 1.23x; bf16 qwen3-30b-moe expert down
1.44x (75 -> 109 us), qwen3.5-35b-moe expert down 1.37x, 7-8b k/v 1.30x, gemma4-26b-moe expert down 1.30x,
gemma4-26b-moe expert gate_up 1.25x, gemma4-26b dense down 1.24x.

Blocksize 32 (review R2, `analysis/review1.txt`, in-process, 3 cycles): fused / dequant 0.820 (fp16) and 0.824 (bf16)
at M = 448, ahead on 29 / 28 shapes; at M = 512 0.921 / 0.924 in total but ahead on only 21 / 19 shapes (bf16 a
minority), with up to 12 shapes losing > 20% (worst qwen3.5-122b-moe expert gate_up 1.55x fp16, qwen3-30b-moe expert
down 1.52x bf16). A `bench.py` run (`analysis/extra.txt`) agrees (0.919-0.920, 19-21 at M = 512). So at blocksize 32
the cap at M ~480-512 is a total-time tradeoff (7.6-8% less total time), not a majority win.

A regional fallback (dequant + F.linear for small weights, N x K < 4-13 M, above M = 448) would recover these losers
but gains at most 0.8% in total at M = 448-512 for either blocksize (fused 1.002-1.026 of the per-shape oracle ->
1.000-1.019), so the single M limit is kept, as on the other targets.

fp32 (gfx1100: 128) -> 512. Fused is 2.4-2.7x faster in total at M = 129-512 and ahead on all 40 shapes. With this
nightly's default BLAS configuration the fp32 `F.linear` runs at ~1.8-1.9 TFLOPS (512 x 8192 x 8192,
`logs/blascheck.out`; fp16 ~22-27 TFLOPS). torch does not use hipBLASLt on gfx1102 ("Attempting to use hipBLASLt on an
unsupported architecture! Overriding blas backend to hipblas"): requesting `cublaslt` gives the same backend and speed,
so the GEMM runs through hipBLAS/rocBLAS and no faster backend was available to test. Consequently the
ROCm/rocm-libraries#12465 hipBLASLt overlay (built from the same pin, `scripts/ovl.sh`) changes nothing here (dequant
fp32 0.996-1.002x at every M, fp16/bf16 0.992-1.006x). The fp32 limit is evidence for this nightly's default BLAS
only; if a later ROCm speeds up fp32 GEMM on gfx1102 it should be re-measured.

### Default dispatch vs per-shape oracle, total (`analysis/dispatch_vs_oracle.txt`)

Inherited (A build, gfx1100 limits): fp16/bf16 within 0.4% at M = 2-384 except M = 1 (1.043), then 1.20-1.42 at
M = 416-512 (cap); fp32 within 0.7% except M = 4 (1.068) and 2.42-2.65 at M = 144-512 (cap). Final: fp16/bf16 within
0.7% at every M (M = 1: 1.003); fp32 within 1.1% at every M (M = 4: 1.007; M > 128 compares final `auto` with the A
build's `wmma`, a cross-run difference of ~1%). The final numbers are a composite (`eval_final.csv`: final `auto` from
the A/B runs, SIMT and dequant from the older eval with the long idle sleep), so sub-percent gaps are within cross-run
differences; the within-run tile and cap comparisons above are the stronger evidence. The oracle uses each build's own
WMMA tiles; against the best swept tile see the tables above.

### Speedup vs upstream main (SIMT for M <= 8, else dequant + F.linear), total, final

| M | 1 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 384 | 512 |
|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | 1.05 | 1.84 | 3.00 | 7.96 | 6.12 | 3.82 | 2.27 | 1.72 | 1.37 | 1.19 |
| bf16 | 1.05 | 1.80 | 2.91 | 8.11 | 6.23 | 3.89 | 2.28 | 1.72 | 1.37 | 1.19 |
| fp32 | 1.00 | 1.06 | 1.79 | 6.79 | 3.66 | 2.94 | 2.70 | 2.52 | 2.44 | 2.40 |

### Compared with the other 32-CU and small parts

gfx1200 (RX 9060 XT, 32 CUs, 320 GB/s, RDNA4) also needed its own tile tables at small/medium M and also moved its
bf16/fp16 limit from 384 to 512; its kernels (16-row tiles) and rules do not transfer to gfx11. The gfx1103 (760M)
fp32 cutoffs and limits fit worse here (fp32 M = 3 ~1.18x; its 384 limit would leave fp32 2.4x slower at M = 385-512).
The gfx115x (8060S/8050S) tables are 5-30% off on gfx1102. The gfx1101 (RX 7700 XT, 54 CUs) tables are 1.02-1.15x
the final ones at M <= 64 (see the tile section).

## Commands

```bash
# timing jobs: scripts/job.sh <name> <cmd> (pins `high`, logs clocks, runs_guard.sh), queued by scripts/queue.sh
python bench.py --dtypes $dt --kernels auto|wmma|dequant --ms <56 M values 1-512>   # scripts/q_eval.sh
python bench.py --dtypes $dt --kernels simt --ms <M <= 64>
BNB_GEMM4BIT_KERNEL=wmma python sweep.py --dtype fp16|fp32 --shapes 0-39 --ms ...     # q_sweep*.sh
python scripts/alt.py --tables A=g11_f16,B=rulesB.py --dtype fp16|bf16 --ms ... --cycles 5   # q_alt1.sh, q_alt32.sh, q_cap.sh
python scripts/alt2.py --configs "old=simt,new=cut:3/1073741824+4/8192:rulesB32.py" --dtype fp32 --ms 4 --cycles 5  # q_review1.sh
PYTHONPATH=<A|B tree> python bench.py --kernels auto --dtypes $dt --ms <61 M values>        # q_aba.sh, q_aba2.sh, q_aba3.sh
python scripts/derive_cfg3_safe.py sweep.csv --score total --merge -1 --feature auto    # per-M; scripts/fitb.sh per bucket
python scripts/rules_eval.py sweep.csv g11_f16 rulesB.py; python scripts/aba_an.py ab fp16; python scripts/final_an.py --excl
PLOT_FILES=data/gfx1102/pinned/dense/eval_final.csv PLOT_PEAK_GBS=288 PLOT_GPU="Radeon RX 7600 XT" PLOT_ARCH=gfx1102 \
  PLOT_CLOCKS="pinned clocks (perf level high)" PLOT_OUT=plots/gfx1102 python plots_kernels.py
```

`derive_cfg3.py`'s `--merge` compares a time-weighted joint total ratio with a count-weighted mean of the separate
ones, so with `--score total` it merged M = 20-512 into one bucket at a 5-12% loss; the fits here use no merging and
one joint fit per hand-chosen bucket (`scripts/fitb.sh`), with the local `derive_cfg3_safe.py` (never merges buckets
without a common swept tile). The branch script is unchanged.

Timeline (UTC, machine clock): pytest A 09:26-09:29; eval 09:29-13:08; overlay builds (no timing) 13:09-13:20 (aborted
clone) and 17:29-17:31; fp16 sweep 13:20-17:20 and 17:31-17:57; m1b 17:57-18:05; overlay dequant 18:05-18:22; alt1
18:22-18:37; cap 18:37-18:42; fp32 sweep 18:42-18:59; alt32 19:00-19:09; builds 19:10-19:12; A/B/A/B 19:12-20:11;
rebuild 20:12-20:14; reverted fp32 A/B/A/B 20:14-20:45; pytest 20:45-20:48; correctness checks 20:49-20:51; artifact
check 20:51-20:53; final rebuild 20:55-20:58; final fp32 A/B/A/B 20:58-21:29; pytest final 21:29-21:32; correctness
checks final 21:32-21:33; block order + blocksize-32 bench 21:37-21:49; BLAS probe 21:50 (no timing job running);
review R1-R3 21:51-21:58.
