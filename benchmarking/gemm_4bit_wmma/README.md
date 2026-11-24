# AMD WMMA 4-bit GEMM: benchmarks, data and plots

Benchmark harness and measurements behind the WMMA kernel for the fused 4-bit GEMM on RDNA3, RDNA3.5 and RDNA4
(`csrc/gemm_4bit_wmma.cu`). The plots are indexed in [plots/README.md](plots/README.md).

## Setup

All numbers were taken with a ROCm nightly (TheRock) venv, e.g. for gfx1100:

```bash
uv venv -p 3.12 venv && source venv/bin/activate
IDX="--index-url https://nightly.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match"
uv pip install $IDX "rocm[libraries,devel,device-gfx1100]==10.2.0a20260928" torch amd-torch-device-gfx1100
rocm-sdk init
export ROCM_PATH=$(rocm-sdk path --root)
cmake -B build -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH=gfx1100 -DBNB_WMMA_DEV=ON && cmake --build build -j
pip install -e .
```

`-DBNB_WMMA_DEV=ON` enables the tuning overrides the scripts use (`BNB_GEMM4BIT_KERNEL=simt|wmma` forces a kernel,
`BNB_WMMA_CFG` a tile). Release builds don't contain them.

Pin the GPU clocks while measuring (`amd-smi set -l HIGH`, or write `high` to
`/sys/class/drm/cardN/device/power_dpm_force_performance_level` on APUs) and restore `auto` afterwards. At the default
level, timings were bimodal by ~10% from DVFS, which is enough to pick wrong tiles.

fp32 dequant + `F.linear` on the ROCm 10.2 nightlies used here hits a slow hipBLASLt fp32 path on gfx1201. The R9700
fp32 baseline was measured with hipBLASLt rebuilt with ROCm/rocm-libraries#12465 (`hipblaslt_fix_overlay.sh`).

## Scripts

| script | |
|---|---|
| `shapes.py` | The 40 LLM linear-layer shapes (N, K). |
| `bench.py` | Times `simt`, `wmma`, `dequant` (dequant + `F.linear`) and `auto` (default C++ dispatch) per shape, dtype and M; weights are streamed from DRAM (a buffer larger than L2 is rotated between iterations). |
| `sweep.py`, `sweep_table.py` | Times every candidate WMMA tile per shape and M. |
| `derive_cfg3.py` | Fits the per-M-bucket tile rules (up to three N leaves) from a sweep. |
| `fit_fp32_cutoff.py` | Fits the fp32 SIMT/WMMA N cutoffs from forced SIMT and WMMA runs. |
| `ab_compare.py` | Compares two `auto` runs (A/B/A) shape by shape. |
| `plots_kernels.py` | Kernel latency and achieved bandwidth vs the bandwidth limit (median and per-shape plots). |
| `plots_vs_main.py` | Speedup of the default dispatch over upstream main on all four GPUs. |
| `par_bench.sh`, `par_sweep.sh`, `runs_guard.sh` | Run bench/sweep on several GPUs in parallel; `runs_guard.sh` logs any other GPU process seen during a run. |
| `hipblaslt_fix_overlay.sh` | Builds hipBLASLt with the gfx1201 fp32 fix for the fp32 dequant + BLAS baseline. |

## Data

`data/<arch>/`, NF4, blocksize 64, 40 shapes, all at pinned clocks unless noted. Columns: `shape,N,K,M,dtype,kernel,us`
(bench) or `shape,N,K,M,cfg,us` (sweeps, one dtype per file).

| file | contents |
|---|---|
| `pinned/eval_final.csv` | Final code: `simt`, `wmma`, `dequant` and `auto` for M = 1-384/512. gfx1151: `pinned/dense/eval_final.csv`. gfx1201: `pinned/eval_final.csv` uses stock hipBLASLt, `pinned/final/eval_final_fixedblas.csv` the fixed one. |
| `pinned/dense/`, gfx1201 `pinned/final/` | Dense M scans (up to 56 values of M) of `auto`, `wmma`, `dequant` and `simt`, used to check the M limits. |
| `pinned/sweep_*.csv` | Tile sweeps the gfx1100 and gfx1201 rules were fitted from (gfx1151: `sweep_*_gapM.csv`, extra M values). |
| `sweep_*.csv` | Earlier tile sweeps at default clocks (gfx1151 rules start from these). |
| `pinned/fp32_simt_wmma.csv` | Forced fp32 SIMT and WMMA runs the fp32 cutoff tables were fitted from. |

gfx1103 (Radeon 760M) was measured on two machines: `data/gfx1103/` (one DDR5-5600 SO-DIMM; no sweeps; its fp32
cutoffs come from it) and `data/gfx1103_apu014/` (128-bit LPDDR5X-7500; tile rules and M limits; sweeps, interleaved
runs and notes in `NOTES.md`).
gfx1150 (Radeon 890M) has its own tile rules and fp32 cutoffs; its data, A/B/A runs and notes are in `data/gfx1150/`
(`NOTES.md`), the validation of the final tables in `data/gfx1150/pinned/paired/`, and `data/gfx1150/pinned/eval_final.csv`
(final tables, assembled from several sessions) for the plots.
