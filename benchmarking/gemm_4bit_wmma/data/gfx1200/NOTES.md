# gfx1200 (Radeon RX 9060 XT) measurements

## Machine and versions

- OrchestrAI Playground HYD-ORAI-DT02 (Hyderabad), Ubuntu 24.04, kernel 7.0.0-43-generic, Ryzen 9 9900X (its gfx1036
  iGPU is HIP device 1; every run used `HIP_VISIBLE_DEVICES=0`, the RX 9060 XT).
- Radeon RX 9060 XT (Navi 44, gfx1200): 32 CUs, 8 GB GDDR6 128-bit, mclk 1258 MHz (20 Gbps) = 320 GB/s peak,
  32 MB Infinity Cache, sclk up to 2780 MHz, VBIOS 113-APM107741-102, PCIe 5.0 x16.
- TheRock nightly `rocm[libraries,devel,device-gfx1200]==10.2.0a20260928`, `torch 2.14.0+rocm10.2.0a20260928`,
  `amd-torch-device-gfx1200` (same nightly), triton 3.8.0, Python 3.12.3.
- hipBLASLt overlay (`../../hipblaslt_fix_overlay.sh <dir> gfx1200`): rocm-libraries 459a4ece + ROCm/rocm-libraries#12465
  at 9df96b1e. Without it a 16x2048x2048 fp32 `F.linear` took 16.8 ms (with it 0.17 ms), as on gfx1201. Every bench
  run used it. It does not change fp16/bf16 dequant + `F.linear` (`dense/dequant_stockblas.csv`, stock hipBLASLt:
  0.996-1.001x in total at M = 256-512).
- Builds: `cmake -G "Unix Makefiles" -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH=gfx1200 [-DBNB_WMMA_DEV=ON]`. A = the
  branch base (gfx1200 on the gfx1201 rules), "draft" = a first version of the gfx1200 rules (M <= 16 bf16/fp16 bucket
  with a third leaf for N >= 28672, fp32 M <= 16 refitted), final = commit "own tuning target for gfx1200".
- Clocks: `power_dpm_force_performance_level=high` on card1 for every timing run, `auto` restored on exit
  (`logs/pinrun.sh`). Every timing job ran under `runs_guard.sh` (`logs/*.guard.log`, all CLEAN, `.foreign` empty).
  The guard only sees other users' processes; same-user overlap was excluded by running everything from serial queue
  scripts (`logs/q_*.sh`) and starting builds and correctness runs only between guarded windows (timeline below).

## Files (NF4, blocksize 64 unless noted, 40 shapes)

| file | contents |
|---|---|
| `pinned/dense/eval_inherited.csv` | A build: `auto`, `wmma`, `dequant` at 56 M values 1-512, `simt` at M <= 64; fp16/bf16/fp32. |
| `pinned/dense/eval_final.csv` | Final build `auto` (= `ab/final_B.csv`, 61 M values incl. 17, 31, 97, 103, 127) and `wmma`; `simt` and `dequant` copied from `eval_inherited.csv` (A build; neither depends on the tile rules, the SIMT/WMMA choice is unchanged). |
| `pinned/sweep_fp16.csv`, `pinned/sweep_fp32.csv` | Tile sweeps (`sweep.py`, every candidate tile), fp16 at 23 M values 1-384, fp32 at 19 M values 4-512. |
| `pinned/fp32_simt_wmma.csv` | Forced fp32 SIMT and WMMA, M <= 64 (A build), input of `fit_fp32_cutoff.py`. |
| `pinned/ab/final_A1.csv`, `final_B.csv`, `final_A2.csv` | A/B/A of `auto`, all dtypes, 61 M values: A build vs final build. |
| `pinned/ab/draft_A1.csv`, `draft_B.csv`, `draft_A2.csv` | The same against the draft build (56 M values). |
| `pinned/ab/leaf_B1.csv`, `leaf_C.csv`, `leaf_B2.csv` | Draft (three-leaf M <= 16 bucket) vs final two-leaf bucket, bf16/fp16 M = 2-16. |
| `pinned/ab/draft_B_row_tiles_first0.csv` | Draft build with `BNB_WMMA_ROW_TILES_FIRST=0`, fp16/fp32, M = 48-512 (vs `draft_B.csv`). |
| `pinned/dense/cap_repeat.csv`, `cap_blocksize32.csv`, `cap_edges.csv` | `auto` and `dequant`, fp16/bf16, for the Python M limit: repeat at M = 320-512 and blocksize 32 at M = 256-512 (draft build, same tiles as final there), M = 385/417/449/481 (final). |
| `pinned/dense/dequant_stockblas.csv` | fp16/bf16 dequant with the stock hipBLASLt, M = 256-512. |
| `pinned/dense/repeats/review_*.csv` | Three repeats of `auto/simt/wmma/dequant` on 5 small or tall shapes at M = 1, 4, 5 and small GQA k/v at M = 448 (final build). |
| `logs/` | pytest logs (`pytest_rel.log` A build, `pytest_final.log` draft, `pytest_final2.log` final), `correctness_final.log` (profiler kernel names, fp32 BLAS check, float64 check; scripts `sanity.py`, `tilecheck.py`), guard logs, queue scripts. |

Commands (in `benchmarking/gemm_4bit_wmma`, see `logs/q_*.sh` for the exact ones):

```bash
MS=1,...,16,18,...,32,36,...,64,72,...,128,144,...,256,288,...,512        # 56 values, see eval_inherited.csv
python bench.py --dtypes $dt --kernels auto|wmma|dequant --ms $MS
python bench.py --dtypes $dt --kernels simt --ms <M <= 64>
BNB_GEMM4BIT_KERNEL=wmma python sweep.py --dtype fp16 --shapes $(seq -s, 0 39) --ms <23 values>
python derive_cfg3.py pinned/sweep_fp16.csv --score total --merge 0.003   # see the derive_cfg3.py note below
python fit_fp32_cutoff.py pinned/fp32_simt_wmma.csv
python ab_compare.py pinned/ab/final_A1.csv pinned/ab/final_B.csv pinned/ab/final_A2.csv auto
```

Timeline (UTC, machine clock): guarded timing windows eval 23:11-23:39, sweeps 23:39-00:11, draft A/B/A + extras
00:14-00:42, leaf B/C/B 00:50-00:52, stock-BLAS dequant 00:53-00:54, final A/B/A + repeats 00:56-01:25. Outside them:
hipBLASLt overlay build (done 23:09), A-build pytest 23:01-23:10, builds 00:13, 00:42, 00:49-00:50, 00:56 (each
finished before the next window), draft pytest and float64 check 00:42-00:48, final correctness 01:26-01:28.

## Findings

"Total" is the sum over the 40 shapes at one M.

### Correctness

`pytest tests/test_functional.py tests/test_linear4bit.py -k "(4bit or Linear4bit or gemv) and not fsdp"`, release
builds (no `BNB_WMMA_DEV`), A build and final build (and the draft): 2417 passed, 28 failed each time, the same
cases. 22 are fp32 `test_gemv_4bit` (pre-existing: also failing on the A build; the test runs the separate GEMV kernel
and is unseeded; the 1151/1201 notes attribute these to the torch fp32 reference, not re-diagnosed here; no fp16/bf16
failure), 6 are `test_4bit_quant_large[*-fp32]`, which allocate an 8 GiB tensor on this 8 GB card (OOM). The profiler
(`logs/sanity.py`, fp16 and fp32, M = 1/8/64, 4096 x 4096) shows `gemm_4bit_wmma<12, ...>` at M = 8 and 64 and
`gemm_4bit_simt` at M = 1. `logs/tilecheck.py` (final build, default dispatch vs
`A.double() @ dequantize_4bit(W).double().T`; (N, K) = (512, 2048), (2816, 704), (4096, 1536), (5376, 2048),
(10240, 1024), (28672, 1024), (36864, 512); 17 M values 1-512; blocksize 32/64/128 where K % blocksize == 0; nested
and not; fp16/bf16/fp32): largest relative error ~4.3e-4 (fp16), ~2.9e-3 (bf16), ~1.7e-6 (fp32), all finite
(`logs/correctness_final.log`). Every benchmark shape fits in 8 GB for every kernel (largest fp32 dequant
28672 x 8192 = 940 MB); no benchmark shape was skipped.

### Tile rules: own target (WMMA_TARGET_GFX1200)

Against the best swept tile per shape (total), gfx1201 tables -> final gfx1200 tables:

| | M <= 16 | 20-32 | 40-96 | 104-128 | 144-384 (512) |
|---|---|---|---|---|---|
| fp16 | 1.054-1.058 -> 1.016-1.020 | 1.073-1.079 -> 1.038-1.040 | 1.023-1.027 (kept) | 1.130-1.156 -> 1.037-1.039 | 1.005-1.031 (kept) |
| fp32 | 1.025-1.027 (kept) | 1.041-1.043 -> 1.005-1.006 | 1.005-1.006 (kept) | 1.001 (kept) | 1.000-1.001 (kept) |

Kept buckets: a refit gains under 1.6% (fp16 M = 96: 1.5%); fp32 M <= 16 gains ~2% (A/B/A of the draft: 1.007-1.021x at
M = 6-16), below the 3% bar, so it stays on the gfx1201 tiles. The draft's third M <= 16 leaf (N >= 28672) rested on
two shapes with opposite preferences (llama3-70b gate/up 5% faster, gemma2-27b gate/up 1% slower) and gained 0.4% in
total (`ab/leaf_*`, B/C/B: 1.004x, B1/B2 within 0.12%); the final bucket has two leaves (N < 10240, else
`{1,1,1,4,4,2}`). At M <= 16 the 32-column split-K-8 tile stays fastest for N < 10240 (gfx1201: N < 3840).

A/B/A of `auto` (final_*), total, A1/A2 within 0.72% at every M (largest at fp16 M = 127); unchanged buckets
within 0.33% (fp16 M = 224: 0.997x):

| | M = 1 | 2-16 | 17-32 | 33-96 | 97-128 | 129-512 |
|---|---|---|---|---|---|---|
| fp16 | 1.000 | 1.039-1.045x | 1.037-1.041x | 0.998-1.002x | 1.087-1.121x | 0.997-1.000x |
| bf16 | 1.000 | 1.039-1.044x | 1.036-1.040x | 0.998-1.001x | 1.093-1.120x | 0.998-1.001x |
| fp32 | 1.000 | 0.998-1.002x | 1.034-1.036x | 0.999-1.001x | 0.999-1.000x | 0.999-1.000x |

M = 17, 31, 97, 103 and 127 (bucket edges between swept M values) are in this run. Single-shape regressions:
small GQA k/v (512 x 2048) at bf16/fp16 M = 17-32, 1.19-1.20x (12.6 -> 15 us; a small-N leaf would recover it, geomean
-0.4%, total -0.05%, below `derive_cfg3.py`'s 1% split threshold); fp32 qwen3-235b expert down (4096 x 1536) at
M = 17-32, 1.14-1.17x; qwen3.5-122b expert down (3072 x 1024) at M = 97-128, ~1.06x (38 -> 40 us).

Per-shape timings of small kernels (< ~70 us) are sometimes bimodal (repeats in `dense/repeats/`: fp32 SIMT on
qwen3-30b expert gate_up at M = 1 16.4/22.9/23.0 us, on qwen3.5-35b expert gate_up at M = 4 27.2/18.2/27.4 us, small
GQA WMMA at M = 448 64.4/64.2/48.1 us; bf16 qwen3.5-122b expert down at M = 97: 60.8 us in final_B vs 40.5 us at
M = 103). Single-shape extremes below are therefore approximate; totals over the 40 shapes repeat within 0.72%.

Row tiles first stays right on gfx1200: `BNB_WMMA_ROW_TILES_FIRST=0` is 1.13-1.98x slower in total (fp16, M = 48-512)
and 1.03-1.05x (fp32).

`derive_cfg3.py` (unchanged in this branch) crashes when greedy merging joins M buckets that share no swept tile
(the candidate filter drops 16-row tiles from M >= 64 and wide-M tiles below); a local copy that never merges such
buckets was used for the fits.

### SIMT/WMMA choice: gfx1201 values kept

fp32 cutoffs: refitting the table on the gfx1200 forced SIMT/WMMA timings (`fit_fp32_cutoff.py`, same measurements
for both tables) gains at most 0.25% in total at any M <= 32 (M = 15); at M = 5 the fit picks the same N >= 8192.
The default dispatch is 1.9-2.0% above the per-shape oracle at fp32 M = 5 (the rule itself 1.1% on the forced
timings, the rest is run-to-run difference between the auto and the forced runs). The narrow shapes stay on SIMT
through M = 31 (N < 1024), up to 1.4x slower than WMMA on 7-8b k/v (1024 x 4096) at M = 15, on tens of microseconds.
bf16/fp16 at M = 1 stays SIMT: in total over the per-shape faster kernel SIMT is 0.1-0.3% off and WMMA 5-6% (WMMA wins
the small MoE expert shapes, e.g. qwen3-30b expert gate_up 15.8 vs 24 us); at M = 2 WMMA is 1.3% off and SIMT 7-8%.

### Python M limits

fp32: fused ahead of dequant + `F.linear` in total at every M up to 512 (0.67x at M = 512, all 40 shapes ahead
there); single MoE shapes lose at M = 128-256 (qwen3.5-35b expert down up to 1.16x at M = 192): 512, as gfx1201.
bf16/fp16 (gfx1201: 384): total fused/dequant 0.71-0.75 at M = 416-512 (final build), fused ahead on 26-31 of 40
shapes; 0.72-0.78 and 25-30 at M = 385/417/449/481; repeat (draft build) 0.71-0.75, 25-31; blocksize 32 0.71-0.74,
24-31 (fp16 M = 512: 24, worst 1.48x at M = 448). The losing shapes are the small MoE expert / small square / k/v
shapes (up to ~1.4-1.5x, tens of microseconds). gfx1201 kept 384 because only 20-22 of 40 shapes won above it in fp16; here a
clear majority does at every M up to 512. With the 384 cap the default dispatch was 1.35-1.42x slower than the oracle
at M = 416-512. New limit: 512 for every dtype (`_gemm_4bit_custom_max_m` stops at 512 anyway).

### Default dispatch vs per-shape oracle (min of simt, wmma, dequant), total

Final (`eval_final.csv`; simt/dequant from the A build), on the 56-M grid of `eval_inherited.csv` (M = 17, 31, 97, 103
and 127 have only `auto`/`wmma`): within 1.4% at every M and dtype (M = 2: fp16 1.013, bf16
1.014) except fp32 M = 5 (1.020); before (`eval_inherited.csv`): within 0.6% except M = 2 bf16/fp16 1.019-1.020, fp32
M = 5 1.019 and bf16/fp16 M = 416-512 1.35-1.42 (cap). The oracle uses each build's own WMMA tiles; against the best
swept tile see the table above. Worst single shape: fp16 M = 1, qwen3-30b expert gate_up, auto (SIMT) 24.5 vs WMMA
15.8 us.

### Speedup vs upstream main (SIMT for M <= 8, else dequant + F.linear), total (worst shape)

| M | fp16 | bf16 | fp32 |
|---|---|---|---|
| 1 | 1.00x (0.93x) | 1.00x (1.00x) | 1.00x (0.99x) |
| 2 | 1.07x (0.90x) | 1.05x (0.91x) | 1.00x (0.95x) |
| 4 | 1.67x (1.08x) | 1.64x (1.08x) | 1.00x (0.94x) |
| 8 | 2.74x (1.38x) | 2.70x (1.37x) | 1.52x (0.96x) |
| 16 | 7.12x (2.52x) | 7.13x (2.56x) | 6.92x (2.98x) |
| 32 | 6.00x (2.00x) | 6.01x (2.31x) | 4.56x (2.07x) |
| 64 | 3.93x (1.46x) | 3.94x (1.71x) | 2.66x (1.03x) |
| 128 | 2.55x (1.13x) | 2.57x (1.31x) | 2.02x (0.99x) |
| 256 | 1.66x (0.93x) | 1.66x (1.01x) | 1.69x (0.91x) |
| 384 | 1.46x (0.81x) | 1.46x (0.88x) | 1.57x (1.20x) |
| 512 | 1.36x (0.72x) | 1.36x (0.78x) | 1.50x (1.23x) |

"main" is reconstructed from this run (this branch's SIMT kernel at M <= 8, the fixed hipBLASLt dequant + `F.linear`
above), not a build of main. Worst-shape values at M <= 4 are within the small-kernel noise above.
