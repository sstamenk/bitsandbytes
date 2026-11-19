# gfx1150 (Radeon 890M) measurements

## Machine and software

- OrchestrAI Playground machine MNB-UCICD-DT219 (maas-mkm-mnb/4dhf3y): Ryzen AI 9 HX 370, Radeon 890M (gfx1150,
  16 CUs / 8 WGPs, 2 MB L2, no Infinity Cache), 32 GB LPDDR5X-7500 on a 128-bit bus: peak 120 GB/s. 2 GB VRAM
  carve-out, 14.5 GB GTT (torch reports 14.5 GB). The large layers stream at 87-101 GB/s at M = 1 (73-84% of peak).
- Ubuntu 24.04, kernel 7.0.0-43-generic, amdgpu DKMS 7.2.5 (installed by the session's previous user, with TDR/debugfs
  patches and `mes_log_enable=1`; not changed here).
- TheRock nightly 10.2.0a20260928: `rocm[libraries,devel,device-gfx1150]==10.2.0a20260928`,
  `torch==2.14.0+rocm10.2.0a20260928`, `amd-torch-device-gfx1150==2.14.0+rocm10.2.0a20260928`, Python 3.12.16 (uv-managed;
  the system Python lacks the headers `torch.compile` needs).
- Built with `cmake -B build -G "Unix Makefiles" -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH=gfx1150 -DBNB_WMMA_DEV=ON
  -DCMAKE_HIP_COMPILER_ROCM_ROOT=$(rocm-sdk path --root)`; the correctness runs without `-DBNB_WMMA_DEV`.

## Clocks and noise

Every timing run wrote `high` to `/sys/class/drm/card1/device/power_dpm_force_performance_level` (restored to `auto` by
an EXIT trap), ran alone through a sequential job queue, and was wrapped in `runs_guard.sh`: all CLEAN, i.e. no GPU
process of another user appeared (the guard does not see the same user's processes; the queue ran one job at a time).
Under load mclk sat at 937 MHz (LPDDR5X-7500) and fclk at 1960 MHz. On this APU `high` is not a hard pin: sclk, sampled
every 30 s, had a median of 2.81 GHz but dropped to 2.33-2.5 GHz in about 10% of the samples (power limit, ~38 W SoC,
70-78 C), more often in fp32 runs (in the fp32 A/B/A, 5-6 of 10 samples per run were below 2.7 GHz).

- The rule changes are validated by `pinned/paired/` (`scripts/paired.py`): one process per dtype, inherited and final
  dispatch selected per call (`BNB_GEMM4BIT_KERNEL`, `BNB_WMMA_CFG`; the launcher applies the same KCH lowering and fp16
  prescale decision as the default dispatch), 5 cycles per (shape, M) alternating which variant runs first, median per
  shape over the cycles, summed over the 40 shapes. The per-cycle old/new total ratios stay within approximately 0.2-1.5%
  of each other at every M.
- The earlier step-by-step A/B/A runs (`pinned/ab/`) were sequential: A1, B, A2 as separate `bench.py` processes, one
  median per (shape, M) each. A1 and A2 agree within approximately 0.2-1.4% in total per M; single small shapes vary by up
  to approximately 10%. They document how the tables were developed; the paired runs are the validation of the final tables.
- Across sessions the totals drift at compute-bound M: on the 34 shapes whose fp32 tile is the same in both,
  `final/final_fp32.csv` is approximately 2.0-3.6% slower in total than `dense/base_fp32.csv` at M = 96-512. A plausible
  cause (not isolated) is the 0.5 s idle between timed batches in the base runs (below), which lets the chip cool.
  Cross-session comparisons (`final/*` vs `dense/base_*`) are therefore only indicative.

## Harness change

`bench.timed_rotation` holds the GPU with `torch.cuda._sleep(5e7)` before each timed batch. On RDNA3.5 that is 500 ms
(100 MHz counter), so a full 40-shape tile sweep spent most of its time sleeping (~8 min per shape). Enqueueing 120 calls
takes ~39 ms here, so the sweeps from shape 7 on, the A/B/A runs, the bf16 runs and the final evaluation use
`BNB_BENCH_SLEEP_CYCLES=1.5e7` (150 ms, new env var, default unchanged). Check: shape 7 (qwen3-14b o/q square, all 14 M
and 799 tiles) measured with both settings (`pinned/sweep_fp16_sleepcheck.csv` vs `pinned/sweep_fp16.csv`): median
ratio 0.998, p5-p95 0.973-1.018, the same best tile at every M. The `dense/base_*` fp16/fp32 runs used the default.

## Data (`pinned/`, NF4, blocksize 64, 40 shapes)

| file | contents |
|---|---|
| `paired/paired_{fp16,bf16,fp32}_*.csv`, `paired/paired_summary.txt` | **Validation of the final tables**: inherited vs final dispatch, interleaved in one process (see Clocks and noise). Rows per cycle, order and variant (kernel, tile). |
| `paired/cap_{fp16,bf16,fp32}.csv` | Final fused path (`auto`) vs dequant + `F.linear`, interleaved the same way, M = 384-512. |
| `dense/base_fp16.csv`, `dense/base_fp32.csv` | PR head (inherited gfx115x rules): `auto`, `wmma`, `dequant` at 56 M = 1-512. |
| `dense/base_simt.csv` | SIMT fp16/fp32 at M = 1-64 (32 values); SIMT does not depend on the tile rules. |
| `base_bf16.csv`, `base_simt_bf16.csv` | PR head bf16: `auto`, `wmma`, `dequant` at 18 M = 1-512; SIMT at M = 1-16. |
| `final/final_fp16.csv`, `final/final_bf16.csv` | Dense runs (56 M, `auto`, `wmma`; bf16 also `dequant`) of an intermediate table (step F + k_lt(1024), see below); `final/finalH_{fp16,bf16}.csv` re-measures the final tables at M = 52-256, the only M where they differ. |
| `final/final_fp32.csv` | Dense fp32 run (56 M) of the round-1 fp32 tables; the final fp32 tables equal it at M = 9-64 and the inherited ones (`dense/base_fp32.csv`) elsewhere. |
| `eval_final.csv` | Final tables assembled from the above (fp16/bf16: `final_*` with M = 52-256 from `finalH_*`; fp32: `final_fp32` at M = 9-64, `dense/base_fp32` elsewhere; dequant/SIMT from the base runs, bf16 dequant from `final_bf16`), used for the plots. Mixed sessions. |
| `final/dispatch_analysis_final.txt`, `final/new_vs_inherited.txt` | Analyses of the dense final-code data (cross-session, indicative). |
| `dense/dispatch_analysis_inherited.txt` | auto vs per-shape oracle, fused vs dequant, vs main, fp32 cutoff tables, for the inherited rules. |
| `sweep_fp16.csv` | Tile sweep fp16, M = 1, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512 (shapes 0-6 with the default sleep, 7-39 with 1.5e7; no duplicate keys). |
| `sweep_fp16_sleepcheck.csv` | Shape 7 with the default sleep (see Harness change). |
| `sweep_fp32.csv` | Tile sweep fp32, M = 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256 (sweep.py's candidates: no 16-row tiles at M >= 64). |
| `sweep_fp32_mt16.csv` | fp32 16-row tiles at M = 64, 96, 128, 256 (explicit `--cfgs`: the kch = 2 rows of K % 128 != 0 shapes ran as kch = 1 and are duplicates). `scripts/fp32_fit_input.py sweep_fp32.csv sweep_fp32_mt16.csv` rebuilds the fp32 fit input (drops those 48 duplicates). |
| `review/sweep_bf16_80_96.csv`, `review/sweep_fp16_80.csv` | bf16 at M = 80, 96 and fp16 at M = 80, all candidates (M = 65-96 joint refit). |
| `order/sweep_fp32_{rtf,nfirst}.csv` | fp32 at M = 64, 128, 256 with the same 20 tiles, row-tiles-first vs N-first. The raw runs also forced `4,1,1,4,1,2`, which does not fit in LDS for fp32: the launcher printed "not available" and launched nothing (~2.4 us rows); those 120 rows per file were removed. |
| `order/sweep_fp16_nfirst.csv` | fp16 N-first at M = 128, 256, 384, all candidates (compare with `sweep_fp16.csv`). |
| `ab/` | Sequential A/B/A runs of the development steps (`ab_compare.txt`). |
| `rules_vs_sweep.txt` | Inherited gfx115x, gfx110x and the final gfx1150 tables against the sweeps. |
| `pytest_final.txt`, `kernel_names_final.txt` | Correctness run and profiler kernel names of the final code. |

Analysis scripts in `scripts/`: `paired.py`, `paired_summary.py`, `rules_eval.py` (prescale-aware: an fp16 kch = 2 tile on
a K % 128 != 0 shape at prescale-eligible M counts as missing, the automatic sweep candidates only have its kch = 1
sibling; `sweep.py --cfgs` could measure it), `leaf_search.py` (NOT prescale-aware: it treats kch = 2 as kch = 1 for
K % 128 != 0, so its scores for such shapes at M >= 96 are approximate), `analyze.py`, `cutoff_tables.py`,
`order_cmp.py`, `new_vs_old.py`, `fp32_fit_input.py`. Exact job lines (command, env) are in `jobs.txt`.

## Findings

### Tile rules: new target `WMMA_TARGET_GFX1150`

Against the per-shape best swept tile (total time over the 40 shapes, `rules_vs_sweep.txt`):

| fp16 M | 1 | 4 | 8 | 16 | 24 | 32 | 48 | 64 | 96 | 128 | 192 | 256 | 384 | 512 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gfx115x (inherited) | 1.037 | 1.049 | 1.046 | 1.065 | 1.166 | 1.114 | 1.151 | 1.059 | 1.112 | 1.142 | 1.366 | 1.436 | 1.198 | 1.214 |
| gfx110x/gfx1103 | 1.062 | 1.056 | 1.070 | 1.108 | 1.120 | 1.114 | 1.153 | 1.070 | 1.044 | 1.011 | 1.008 | 1.011 | 1.014 | 1.019 |
| gfx1150 (final) | 1.005 | 1.019 | 1.014 | 1.032 | 1.018 | 1.022 | 1.035 | 1.018 | 1.022 | 1.004 | 1.003 | 1.002 | 1.000 | 1.000 |

| fp32 M | 4 | 8 | 12 | 16 | 24 | 32 | 48 | 64 | 96 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gfx115x (inherited) | 1.035 | 1.036 | 1.137 | 1.136 | 1.030 | 1.144 | 1.086 | 1.094 | 1.004 | 1.004 | 1.003 |
| gfx110x/gfx1103 | 1.064 | 1.070 | 1.172 | 1.171 | 1.096 | 1.025 | 1.058 | 1.117 | 1.124 | 1.131 | 1.145 |
| gfx1150 (final) | 1.035 | 1.036 | 1.021 | 1.020 | 1.008 | 1.004 | 1.001 | 1.007 | 1.004 | 1.004 | 1.003 |

(Inherited fp16 at M >= 96 leaves out 2 (shape, M) per M whose prescaled kch = 2 tile the sweep lacks.) gfx110x
(gfx1103's tiles) is within approximately 2% at fp16 M >= 128 but 4-16% off below, and 2.5-17% off for fp32, so no existing target's
rules fit; the inherited rules lose most on tall-K layers at M > 128 (e.g. llama3-70b down, qwen3-32b down: 4.9-5.4 TFLOPS
vs 9-10 on the other large layers), likely because the 890M has no Infinity Cache (hypothesis).

How the final tables were made:
1. `derive_cfg3.py --feature auto --merge 0.003` on the fp16 sweep and on the fp32 sweep + 16-row supplement.
2. fp16 M = 65-96 refitted on fp16 and bf16 at M = 80 and 96 together (`review/`): the first fit came from fp16 at M = 96,
   where fp16 prescaling applies, and was approximately 1.4-1.5x slower than the inherited rules on four bf16 shapes.
3. One extra narrow-N leaf at fp16 M = 17-32 (`n_lt(2048)`, 5 shapes), 33-48 (`n_lt(1024)`, only small GQA k/v in this
   corpus), 49-64 (`n_lt(2048)`, plus the `k_lt(2560)` leaf tile changed to `{4,1,1,4,2,2}`, which recovers
   qwen3.5-122b-moe expert down) and fp32 M = 9-16 (`n_lt(2048)`), for the MoE expert / GQA k/v layers the large-layer
   tiles slowed down (`scripts/leaf_search.py`: one leaf, kept only with a geomean gain and no total-time loss on the sweep).
4. Reverted to the inherited choices where the gain was below approximately 3% per bucket: fp32 tiles at M <= 8 and
   M > 64 (approximately 0.1-1.2% in sequential A/B/A) and the fp32 SIMT/WMMA cutoffs (a refit gained 0.3-0.9% and cost
   qwen3-30b-moe expert down 1.18x at M = 4). Also dropped: short-K leaves at fp16 M = 65-256 (per-shape tradeoffs up to
   1.20x, totals within noise).

Steps 1-3 were developed with sequential A/B/A runs (`ab/ab_compare.txt`): round 1 vs inherited fp16/bf16
1.03-1.43x and fp32 1.00-1.14x in total; each later step changed totals by less than approximately 1% (by design: they target
small layers) and the targeted shapes by 5-32% (geomean).

### Validation of the final tables (paired, interleaved, `paired/paired_summary.txt`)

Inherited vs final default dispatch, sum over the 40 shapes of the per-shape medians of 5 alternating cycles, old/new
(> 1: final faster); the per-cycle range of that ratio; and how many of the 40 shapes are more than 5% / 10% slower:

```
dtype    M  old/new  geomean  per-cycle old/new  >5%  >10%  worst new/old          best new/old
bf16    1  1.032   1.052    1.031-1.033       1    0  1.10 gemma4-26b-moe e  0.77 small square
bf16    4  1.030   1.054    1.029-1.031       1    0  1.09 gemma4-26b-moe e  0.76 qwen3-30b-moe ex
bf16    8  1.034   1.058    1.034-1.034       1    0  1.09 gemma4-26b-moe e  0.70 gemma4-26b-moe e
bf16   12  1.032   1.034    1.030-1.032       3    2  1.15 qwen3-235b-moe e  0.79 gemma4-26b-moe e
bf16   16  1.033   1.036    1.030-1.036       3    2  1.16 small square      0.79 gemma4-26b-moe e
bf16   20  1.149   1.122    1.147-1.151       1    0  1.05 small square      0.67 gemma4-26b-moe e
bf16   24  1.147   1.120    1.144-1.156       1    0  1.05 small square      0.68 gemma4-26b-moe e
bf16   32  1.089   1.151    1.087-1.093       2    0  1.07 small GQA k/v     0.59 qwen3-235b-moe e
bf16   36  1.118   1.119    1.112-1.121       2    1  1.17 gemma4-26b-moe e  0.74 qwen3-30b-moe ex
bf16   40  1.118   1.120    1.115-1.121       2    2  1.18 gemma4-26b-moe e  0.74 qwen3-30b-moe ex
bf16   48  1.113   1.112    1.110-1.117       3    2  1.16 gemma4-26b-moe e  0.74 qwen3-30b-moe ex
bf16   56  1.046   1.112    1.041-1.049       1    0  1.05 qwen3-14b o/q sq  0.63 qwen3.5-122b-moe
bf16   64  1.046   1.110    1.039-1.050       2    1  1.14 qwen3.5-35b-moe   0.63 qwen3.5-122b-moe
bf16   72  1.086   1.077    1.085-1.088       5    2  1.28 qwen3-30b-moe ex  0.71 qwen3.5-35b-moe
bf16   80  1.096   1.081    1.083-1.102       4    1  1.26 qwen3-30b-moe ex  0.71 qwen3.5-35b-moe
bf16   88  1.114   1.085    1.110-1.119       3    2  1.24 qwen3-30b-moe ex  0.71 qwen3.5-35b-moe
bf16   96  1.111   1.084    1.095-1.116       3    2  1.22 qwen3-30b-moe ex  0.72 qwen3.5-35b-moe
bf16  112  1.205   1.133    1.198-1.210       4    2  1.23 qwen3.5-35b-moe   0.66 llama3-70b down
bf16  128  1.259   1.153    1.253-1.262       4    2  1.20 qwen3.5-35b-moe   0.59 llama3-70b down
bf16  160  1.337   1.166    1.330-1.343       2    1  1.23 qwen3.5-35b-moe   0.52 llama3-70b down
bf16  192  1.361   1.202    1.357-1.374       2    0  1.08 qwen3.5-35b-moe   0.56 qwen3-32b down
bf16  224  1.396   1.231    1.388-1.403       2    2  1.11 qwen3.5-35b-moe   0.52 qwen3-32b down
bf16  256  1.405   1.295    1.402-1.406       0    0  1.03 qwen3.5-35b-moe   0.52 qwen3-32b down
bf16  288  1.239   1.066    1.233-1.247       0    0  1.00 llama3-70b gate/  0.51 gemma3-27b down
bf16  384  1.199   1.059    1.195-1.202       0    0  1.00 llama3-70b gate/  0.54 gemma3-27b down
bf16  448  1.209   1.059    1.206-1.211       0    0  1.00 llama3-70b gate/  0.56 gemma3-27b down
bf16  512  1.209   1.059    1.206-1.209       0    0  1.00 llama3-70b gate/  0.56 gemma3-27b down
fp16    1  1.032   1.056    1.031-1.033       1    0  1.10 gemma4-26b-moe e  0.73 gemma4-26b-moe e
fp16    4  1.030   1.055    1.029-1.031       1    0  1.09 gemma4-26b-moe e  0.75 gemma4-26b-moe e
fp16    8  1.034   1.058    1.033-1.034       1    0  1.09 gemma4-26b-moe e  0.71 gemma4-26b-moe e
fp16   12  1.031   1.033    1.030-1.032       4    2  1.16 small square      0.78 gemma4-26b-moe e
fp16   16  1.033   1.037    1.032-1.036       3    2  1.15 small square      0.79 gemma4-26b-moe e
fp16   20  1.149   1.121    1.147-1.152       0    0  1.02 small square      0.67 gemma4-26b-moe e
fp16   24  1.151   1.122    1.148-1.160       0    0  1.04 small square      0.68 gemma4-26b-moe e
fp16   32  1.090   1.151    1.089-1.094       2    0  1.07 qwen3.5-122b-moe  0.58 qwen3-235b-moe e
fp16   36  1.114   1.119    1.113-1.115       2    1  1.17 gemma4-26b-moe e  0.73 qwen3-30b-moe ex
fp16   40  1.114   1.117    1.111-1.116       2    2  1.15 gemma4-26b-moe e  0.74 qwen3-30b-moe ex
fp16   48  1.118   1.116    1.112-1.119       2    2  1.16 gemma4-26b-moe e  0.74 qwen3-30b-moe ex
fp16   56  1.050   1.110    1.046-1.050       1    1  1.14 qwen3.5-35b-moe   0.63 qwen3.5-122b-moe
fp16   64  1.040   1.109    1.036-1.045       2    1  1.14 qwen3.5-35b-moe   0.62 qwen3.5-122b-moe
fp16   72  1.090   1.084    1.088-1.094       3    2  1.27 qwen3-30b-moe ex  0.71 qwen3.5-35b-moe
fp16   80  1.086   1.082    1.084-1.090       5    1  1.26 qwen3-30b-moe ex  0.71 qwen3.5-35b-moe
fp16   88  1.110   1.086    1.098-1.123       4    2  1.24 qwen3-30b-moe ex  0.71 qwen3.5-35b-moe
fp16   96  1.091   1.099    1.090-1.094       1    1  1.13 gemma4-26b-moe e  0.69 qwen3.5-35b-moe
fp16  112  1.130   1.088    1.129-1.135       3    3  1.28 qwen3-30b-moe ex  0.81 qwen3.5-35b-moe
fp16  128  1.133   1.092    1.129-1.148       3    3  1.18 qwen3.5-35b-moe   0.80 qwen3.5-35b-moe
fp16  160  1.324   1.181    1.314-1.332       0    0  1.03 qwen3.5-35b-moe   0.56 llama3-70b down
fp16  192  1.376   1.195    1.367-1.380       2    2  1.19 qwen3.5-35b-moe   0.55 llama3-70b down
fp16  224  1.408   1.238    1.400-1.416       0    0  1.05 qwen3-30b-moe ex  0.53 qwen3-32b down
fp16  256  1.436   1.298    1.425-1.445       1    0  1.07 qwen3-30b-moe ex  0.53 qwen3-32b down
fp16  288  1.243   1.062    1.235-1.246       0    0  1.02 gemma4-26b dense  0.52 qwen3-32b down
fp16  384  1.195   1.053    1.192-1.198       1    0  1.05 gemma4-26b dense  0.55 gemma3-27b down
fp16  448  1.215   1.055    1.213-1.218       0    0  1.04 gemma4-26b dense  0.56 qwen3-32b down
fp16  512  1.211   1.056    1.210-1.211       0    0  1.03 gemma4-26b dense  0.57 gemma3-27b down
fp32    4  1.000   1.000    1.000-1.000       0    0  1.00 llama3-70b gate/  1.00 llama3-70b gate/
fp32    5  1.000   1.000    1.000-1.000       0    0  1.00 llama3-70b gate/  1.00 llama3-70b gate/
fp32    6  1.000   1.000    1.000-1.000       0    0  1.00 llama3-70b gate/  1.00 llama3-70b gate/
fp32    8  1.000   1.000    1.000-1.000       0    0  1.00 llama3-70b gate/  1.00 llama3-70b gate/
fp32    9  1.112   1.130    1.107-1.116       1    0  1.05 qwen3-coder-480b  0.65 phi3-mini/llama3
fp32   12  1.114   1.130    1.114-1.117       1    0  1.06 qwen3-coder-480b  0.65 gemma3-4b down
fp32   16  1.114   1.130    1.113-1.116       0    0  1.04 qwen3-coder-480b  0.65 gemma3-4b down
fp32   20  1.028   1.039    1.027-1.028       5    4  1.20 gemma4-26b-moe e  0.74 qwen3.5-122b-moe
fp32   24  1.023   1.038    1.021-1.028       5    4  1.20 gemma4-26b-moe e  0.74 qwen3.5-122b-moe
fp32   28  1.151   1.171    1.150-1.152       0    0  1.00 llama3-70b gate/  0.62 qwen3-30b-moe ex
fp32   32  1.139   1.167    1.139-1.142       0    0  1.00 llama3-70b gate/  0.62 qwen3-30b-moe ex
fp32   40  1.083   1.054    1.082-1.083       2    0  1.07 gemma4-26b-moe e  0.84 phi3-mini/llama3
fp32   48  1.085   1.057    1.084-1.086       0    0  1.04 gemma4-26b-moe e  0.84 llama3-70b down
fp32   56  1.089   1.058    1.088-1.091       2    0  1.10 gemma4-26b dense  0.83 llama3-70b down
fp32   64  1.087   1.058    1.085-1.088       2    0  1.09 gemma4-26b dense  0.84 qwen3-coder-480b
fp32   96  1.000   1.000    1.000-1.000       0    0  1.00 llama3-70b gate/  1.00 llama3-70b gate/
fp32  256  1.000   1.000    1.000-1.000       0    0  1.00 llama3-70b gate/  1.00 llama3-70b gate/
```

Every fp16/bf16 bucket gains at least approximately 3% in total (smallest: M <= 16, approximately 3.0-3.4%). fp32 changes only M = 9-64 (M <= 8 and
>= 65 pick the same kernels as the inherited rules, ratio 1.000 by construction): 11.1-11.4% at M = 9-16 (`{16}`
bucket); the `{64}` bucket (M = 17-64) gains 8.3-15% at M = 28-64 but only 2.3-2.8% at M = 20-24.

Single shapes still more than approximately 20% slower than with the inherited rules (paired data): qwen3-30b-moe expert
down (N = 2048, K = 768) at fp16/bf16 M = 72-112 (approximately 1.22-1.28x); qwen3.5-35b-moe expert down (N = 2048,
K = 512) at bf16 M = 112-160 (approximately 1.20-1.23x); gemma4-26b-moe expert down at fp32 M = 24 (1.20x). Short-K leaves at M = 65-256 fixed
some of these but cost other MoE down projections up to 1.20x in the same buckets, so they were dropped; within the rule
structure (three leaves, one feature each) these losses are the price of the 8-26% bucket gains on the same M. In the same
buckets the best shapes run at 0.59-0.80x of their inherited time (M = 80-128).

### fp32 SIMT/WMMA cutoffs: gfx115x table kept

`{3, INT_MAX}, {5, 2560}, {7, 1024}` with the gfx115x tiles (which gfx1150 keeps through M = 8): within 0.1-0.9% of the
per-shape faster kernel in total at M = 3-7 (`dense/dispatch_analysis_inherited.txt`). The gfx1103 table is 1.13x slower
at M = 3, the gfx110x one 1.11x at M = 4. A refit with the round-1 M <= 8 tiles (`{3, INT_MAX}, {4, 1024}`) gained only
approximately 0.3-0.9% (`ab/abG_*`) and was not kept.

### Block order: row-tiles-first kept

fp16 N-first (`order/sweep_fp16_nfirst.csv`) is approximately 1.12x slower best-vs-best at M = 128-256 (1.01x at 384);
fp32 N-first is 0.4-0.8% slower at M = 64-256 (a tie), unlike gfx1103 (approximately 1.08x the other way).

### Python M limit: 512 kept (gfx115x branch), per dtype

Interleaved final fused path vs dequant + `F.linear` (5 alternating cycles, `paired/cap_*.csv`), dequant / fused in
total time at M = 384, 416, 448, 480, 512, and how many of the 40 shapes the fused path wins:

| dtype | 384 | 416 | 448 | 480 | 512 |
|---|---|---|---|---|---|
| fp16 | 1.68x, 40/40 | 1.66x, 39/40 | 1.65x, 39/40 | 1.58x, 39/40 | 1.48x, 39/40 |
| bf16 | 1.70x, 40/40 | 1.68x, 39/40 | 1.69x, 38/40 | 1.61x, 38/40 | 1.48x, 37/40 |
| fp32 | 3.11x, 40/40* | 3.09x, 40/40* | 3.13x, 40/40* | 3.14x, 40/40* | 3.22x, 40/40 |

\* fp32 was measured interleaved at M = 512 only (machine time); the starred values are the single-pass dense runs
(`final/dispatch_analysis_final.txt`, which gives 3.17x at M = 512).

512 is supported by both the majority of shapes and the total time in every dtype (the losing shapes are the Gemma 4 MoE
experts, up to approximately 1.2x); it is the largest M measured, not a crossover. The dense single-pass runs agree
(`final/dispatch_analysis_final.txt`). gfx1103's cap (384) would give away approximately 1.5-1.7x at M = 416-512.

### N = 2816-3071 leaves

fp16 M > 64 and fp32 M = 9-64 have an `N < 3072` leaf after `N < 2816` that only the two Gemma 4 down projections
(N = 2816, K = 704 and 2112, both K % 128 != 0) reach. Measured improvements on those shapes, not a rule validated for
that N range.

### Speedup vs main

Main: SIMT for M <= 8, dequant + `F.linear` above. Final tables (dense runs, indicative): fp16 1.16x in total at M = 1,
2.4x at M = 8, 7.9x at M = 16, 2.6x at M = 128, 2.0x at M = 256, 1.5x at M = 512; bf16 similar; fp32 1.0x at M <= 3,
1.9x at M = 8, approximately 8x at M = 9-16, 3.1x at M = 512.

## Correctness

- PR head (no dev overrides): `pytest tests/test_functional.py tests/test_linear4bit.py -k "(4bit or Linear4bit or gemv)
  and not fsdp"`: 2445 passed, 0 failed (no fp32 `test_gemv_4bit` failures with this toolchain). Profiler: fp16 M = 1/16/128
  run `gemm_4bit_wmma<11, __half, ...>`, fp32 M = 1 runs `gemm_4bit_simt<float, 1>`.
- Final code (this branch's csrc, built without `BNB_WMMA_DEV`, `BNB_WMMA_DEV:BOOL=OFF`; only comments in
  `gemm_4bit_wmma_rules.cuh` changed after this run): the same subset, 2445 passed, 0 failed (`pinned/pytest_final.txt`). An intermediate table (step F + k_lt(1024)) also passed 2445/0.
  `pinned/kernel_names_final.txt`: fp16/bf16 M = 16 run the gfx1150 tile `<2, 1, 1, 4, 4, 2>` (gfx115x picked
  `<2, 2, 1, 1, 4, 2>`), fp32 M = 1 SIMT, M = 16/128 WMMA.
