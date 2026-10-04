# AMD WMMA 4-bit GEMM: benchmark and tuning harness

Harness, data and notes behind the RDNA3/RDNA4 WMMA kernel for the fused 4-bit GEMM
(`csrc/gemm_4bit_wmma.cu`). Use it to re-tune the tile rules and to enable more
architectures.

Dispatch with `K % 64 == 0` and blocksize >= 32:

| arch | dtype | kernel choice |
|---|---|---|
| RDNA3 (gfx110x) | fp16 / bf16 | WMMA for every M the Python side routes to the custom kernel (M <= 256) |
| RDNA3 (gfx110x) | fp32 | SIMT for M <= 5, WMMA for M = 6-128 |
| RDNA4 (gfx1201) | fp16 / bf16 | SIMT for M = 1, WMMA for M = 2-256 |
| RDNA4 (gfx1201) | fp32 | SIMT for M <= 4; from M = 5, WMMA when N is large enough for its M (SIMT below N = 8192 at M = 5, 4096 at M = 6, 2048 at M = 7-9, 1408 at M = 10-15, 1024 at M = 16-31), otherwise SIMT; fused through M = 512 |

fp32 is emulated with 3-way split bf16 operands (6 WMMAs per tile) and is fp32-accurate.
Every other architecture keeps SIMT plus dequant+`F.linear`, unchanged.

## Setup

The numbers were taken with a ROCm nightly (TheRock) venv:

```bash
uv venv -p 3.12 venv-rocm-nightly && source venv-rocm-nightly/bin/activate
IDX="--index-url https://nightly.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match"
uv pip install $IDX "rocm[libraries,devel,device-gfx1100]==10.2.0a20260928" torch amd-torch-device-gfx1100
rocm-sdk init
```

The per-arch device wheel (`amd-torch-device-gfx1100`) is required, otherwise torch fails with
"invalid kernel file".

For RDNA4 use `device-gfx1201` / `amd-torch-device-gfx1201` and `-DBNB_ROCM_ARCH=gfx1201`.

Build with the dev overrides enabled. `BNB_WMMA_DEV` is off by default, and release builds do
not contain the overrides.

```bash
export ROCM_PATH=$(rocm-sdk path --root)
cmake -B build -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH=gfx1100 -DBNB_WMMA_DEV=ON \
      -DCMAKE_HIP_COMPILER_ROCM_ROOT=$ROCM_PATH
cmake --build build -j
export PYTHONPATH=$PWD   # use the in-tree library, no editable install needed
```

Dev-only environment variables (these need `-DBNB_WMMA_DEV=ON`):

| variable | effect |
|---|---|
| `BNB_GEMM4BIT_KERNEL=simt\|wmma` | Forces the fused kernel in the C dispatch. The Python side still chooses fused vs dequant+`F.linear`. |
| `BNB_WMMA_CFG=wm,wn,warps_m,warps_n,ksplit,kch` | Forces a WMMA tile. It is re-read on every call. The allowed tiles are the `BNB_WMMA_DEV_TRY` table in the launcher. |

## Methodology

- **Shapes.** 40 LLM linear-layer shapes (`shapes.py`), the same set as PR #1979 (SIMT kernel). NF4, blocksize 64, not nested.
- **Weights from DRAM.** Each call reads its weights from DRAM, as in layer-by-layer decoding. The weights rotate over at least 384 MB of copies, 4x the 96 MB Infinity Cache on the W7900.
  - Don't flush L2 by writing a buffer: the dirty Infinity Cache lines inflate the time of small kernels a lot.
- **Launch overhead.** A `torch.cuda._sleep` kernel holds the GPU until every launch is queued, so CPU launch latency never falls inside the timed window.
- **Per-call timing.** The result is the median of the per-call event deltas.
- **Real quantized weights.** Always time real NF4-quantized weights. Uniform or uniformly random bytes change the bank-conflict pattern of the LDS lookup table and gave misleading results: a prefetch change looked 1.2x faster in a synthetic harness but was neutral on real weights.
- **One process per GPU.** `par_bench.sh` and `par_sweep.sh` shard shapes round-robin across GPUs, one process per GPU, and merge the CSVs.
- **Clock mode.** Never compare timings across clock modes. `power_dpm_force_performance_level=profile_standard`, which is needed for rocprofv3 SQ counters, pins sclk at about 970 MHz and roughly halves WMMA throughput. It also moves the crossovers. All data here was taken at default (auto) clocks.
- **Device order.** HIP device order can differ from amd-smi order. Check with `rocminfo` before pinning clocks on a specific GPU.
- **Exclusive GPUs.** Other users' jobs on a shared machine invalidate timings, also on GPUs you are not using (their host and PCIe traffic still interferes). `amd-smi process` does not show other users' processes; `rocm-smi --showpids` does. On gfx1201 a vLLM job on the other GPUs inflated small-kernel times by up to 10x, while two clean runs 25 minutes apart agreed within ±2% (p5-p95). `runs_guard.sh` wraps a timing job, logs any foreign GPU process during the run, and reports CLEAN or CONTAMINATED.

## Scripts

| script | purpose |
|---|---|
| `bench.py` | Times SIMT / WMMA / dequant+BLAS for the chosen dtypes, M values and shapes, or `auto` (the default C dispatch). Each fused kernel runs in its own subprocess. |
| `par_bench.sh <tag> "<gpus>" <shape idx list> <kernels> [bench args]` | Runs `bench.py` sharded across GPUs. |
| `sweep.py` | Times every candidate tile per (shape, M) in one process through `BNB_WMMA_CFG`. |
| `par_sweep.sh <tag> "<gpus>" <shape idx list> [sweep args]` | Runs `sweep.py` sharded across GPUs. |
| `derive_cfg.py sweep.csv` | For each M, finds the best single tile and the best 2-way split on a shape feature, scored by geomean slowdown vs the per-shape best. |
| `best_vs_base.py sweep.csv bench.csv` | Per-shape speedup of the best swept tile vs SIMT and dequant. |
| `summarize.py bench.csv` | Median latency per M and kernel, plus a count of wins. |
| `plots.py` | Charts and a markdown table comparing the new default dispatch to the previous one. Set `PLOT_FILES`, `PLOT_OUT` and `PLOT_CLOCKS`. |
| `check.py` | fp16/bf16 correctness sweep over odd shapes, blocksizes, nested stats and bias (`BNB_GEMM4BIT_KERNEL` picks the kernel). |
| `check_tiles.py` | Correctness of every tile in the dev table (the sweep candidates) for fp16/bf16/fp32, blocksizes 32/64/256 and nested stats, against a float64 reference. |
| `sweep_table.py` | The dev tile table and the LDS budget check shared by `sweep.py` and `check_tiles.py` (rows per WMMA follow the device: 8 on gfx11, 16 on gfx12). |
| `runs_guard.sh <log> <cmd...>` | Runs a timing job and flags it CONTAMINATED if any other user's GPU process appeared during the run. `GUARD_ALLOW_PIDS` / `GUARD_IDLE_GPUS` tolerate a known idle process (e.g. a vLLM server parked on other GPUs) as long as it stays idle. |
| `hipblaslt_fix_overlay.sh <dir>` | Builds the active nightly's exact hipBLASLt host libraries plus the ROCM-31245 fix and writes `<dir>/env.sh` to load them (see Known issues). Needed for a meaningful fp32 dequant+BLAS baseline on ROCm 10.2 nightlies from 0913. |
| `experiments/wmma_gfx12_layout_probe.hip` | Maps the gfx12 WMMA operand and accumulator layouts with one-hot inputs. |
| `check_f32.py`, `f32_err_dist.py` | fp32 accuracy of WMMA (split bf16) vs SIMT vs torch, against a float64 reference. |

Example: re-sweep fp16 on 8 GPUs, derive rules, then re-evaluate.

```bash
./par_sweep.sh sw_fp16 "0 1 2 3 4 5 6 7" $(seq -s, 0 39) --dtype fp16 --ms 1,4,8,12,16,24,32,48,64,96,128,256
python derive_cfg.py sw_fp16.csv
./par_bench.sh ev_wd   "0 1 2 3 4 5 6 7" $(seq -s, 0 39) wmma,dequant --dtypes fp16,bf16,fp32 \
    --ms 1,2,3,4,6,8,12,16,24,32,48,64,96,128,192,256,384
./par_bench.sh ev_simt "0 1 2 3 4 5 6 7" $(seq -s, 0 39) simt --dtypes fp16,bf16,fp32 \
    --ms 1,2,3,4,6,8,12,16,24,32,48,64
PLOT_FILES=ev_wd.csv,ev_simt.csv PLOT_OUT=plots/<arch> python plots.py
```

## Data (`data/gfx1100`, W7900, ROCm 10.2.0a20260928 nightly, torch 2.15.0a0, default clocks)

| file | contents |
|---|---|
| `eval_wmma_dequant.csv` | WMMA and dequant+BLAS for the shipped kernel and tile rules: 40 shapes x fp16/bf16/fp32 x M = 1-384. |
| `eval_simt.csv` | SIMT for the same shapes, M = 1-64. |
| `sweep_fp16.csv` | Tile sweep for the shipped kernel (fp16, which is also used for bf16). The fp16/bf16 rules were derived from this file. |
| `sweep_fp32.csv` | Tile sweep for fp32 (split bf16). The fp32 rules were derived from this file. |
| `experiments/sweep_fp16_ring.csv` | The same sweep with the rejected 3-slot B-prefetch ring kernel (`experiments/gemm_4bit_wmma_ring.cu`). |

The charts and tables in `plots/gfx1100` were generated from the two `eval_*` files.

The headline result compares the new default dispatch with the previous one (SIMT for M <= 8, dequant+BLAS above). Values are the median over the 40 shapes, with the worst shape in brackets:

| M | 1 | 2 | 8 | 16 | 32 | 48 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|---|
| fp16 | 1.05x (0.89) | 1.16x (1.00) | 2.33x (1.28) | 3.24x (1.69) | 2.64x (1.49) | 2.56x (1.53) | 2.20x (1.37) | 1.64x (1.16) | 1.11x (0.83) |
| bf16 | 1.03x (0.88) | 1.15x (0.98) | 2.36x (1.24) | 4.52x (1.76) | 3.53x (1.57) | 2.93x (1.63) | 2.59x (1.44) | 1.74x (1.19) | 1.17x (0.99) |
| fp32 | 1.00x | 1.00x | 1.51x (0.93) | 4.22x (2.59) | 2.72x (1.97) | 2.15x (1.56) | 1.50x (1.11) | 1.13x (0.90) | 1.00x |

## Data (`data/gfx1201`, Radeon AI Pro R9700S, ROCm 10.2.0a20260928 nightly, torch 2.14.0, default clocks)

gfx1201 (Navi48, RDNA4): 64 CUs, 64 KB LDS per workgroup, 64 MB Infinity Cache, 32 GB GDDR6 (~640 GB/s).
The 384 MB weight rotation is 6x the Infinity Cache. Every timing file below was taken with no other
active GPU job on the machine (checked with `runs_guard.sh`). The `eval_*` files were run on 4 of the 8 GPUs
next to a parked, idle vLLM server; a SIMT re-run in that setup matched the earlier all-idle data within
±3% (p5-p95).

dequant+BLAS in the `eval_*` files uses hipBLASLt built from the nightly's exact revision plus the ROCM-31245
fix (`hipblaslt_fix_overlay.sh`); with the stock nightly every small fp32 GEMM takes ~20 ms (see Known issues).
The fix leaves fp16/bf16 dequant+BLAS unchanged (identical total time at every M) and makes fp32 6-17x faster.

| file | contents |
|---|---|
| `baseline_simt.csv`, `baseline_dequant.csv` | The previous dispatch's kernels before the port, stock hipBLASLt: SIMT (M = 1-128) and dequant+BLAS (M = 1-384), 40 shapes x fp16/bf16/fp32. fp32 dequant+BLAS here shows the ROCM-31245 slowdown. |
| `eval_wmma_dequant.csv` | WMMA with the shipped gfx1201 rules and dequant+BLAS with fixed hipBLASLt: 40 shapes x fp16/bf16/fp32 x M = 1-512. |
| `eval_simt.csv` | SIMT for the same shapes, M = 1-128, same run. |
| `eval_fp32_dispatch.csv` | fp32 default dispatch (`auto`), SIMT and WMMA at M = 1-32, including M = 5, 7, 10, 14, 20, to check the N-dependent SIMT/WMMA choice between the M values it was fitted on. |
| `sweep_fp16.csv`, `sweep_fp32.csv` | Tile sweeps the gfx1201 rules were derived from. |
| `sweep_bf16.csv` | bf16 tile sweep: the fp16 rules are within 1.03x (total time) of the per-shape best bf16 tile, and bf16's own derivation picks the same tiles. |
| `experiments/sweep_fp16_blds.csv` | fp16 sweep with the rejected B-in-LDS variant (see rejected experiments). |

The charts and tables in `plots/gfx1201` were generated from the two `eval_*` files with
`PLOT_ARCH=gfx1201 PLOT_NEW="fp16:1:256,bf16:1:256,fp32:4:512"`, plus `eval_fp32_dispatch.csv`, whose measured `auto`
dispatch replaces the M rule for fp32 at M <= 32.

New default dispatch vs the previous one (SIMT for M <= 8, dequant+BLAS above), median over the 40
shapes with the worst shape in brackets, and the total time over the 40 shapes:

| M | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | 1.00x | 1.07x (0.97) | 1.40x (1.08) | 2.25x (1.31) | 3.45x (1.92) | 2.70x (1.73) | 2.55x (1.38) | 2.11x (1.23) | 1.28x (0.95) | 1.00x |
| fp16 total | 1.00x | 1.10x | 1.59x | 2.54x | 6.74x | 5.74x | 3.68x | 2.19x | 1.29x | 1.00x |
| bf16 | 1.00x | 1.06x (0.97) | 1.39x (1.08) | 2.22x (1.30) | 3.88x (1.92) | 3.11x (1.79) | 2.68x (1.50) | 2.07x (1.33) | 1.26x (1.05) | 1.00x |
| bf16 total | 1.00x | 1.07x | 1.56x | 2.50x | 6.77x | 5.76x | 3.72x | 2.20x | 1.29x | 1.00x |
| fp32 | 1.00x | 1.01x | 1.00x | 1.29x (1.00) | 6.88x (3.35) | 4.37x (2.87) | 2.62x (1.92) | 1.84x (1.34) | 1.44x (1.16) | 1.28x (1.01) |
| fp32 total | 1.00x | 1.00x | 1.00x | 1.53x | 7.42x | 4.57x | 2.68x | 1.88x | 1.51x | 1.36x |

How the thresholds were set (total time over the 40 shapes, `eval_*` files):

- **SIMT vs WMMA, fp16/bf16.** WMMA is 2-3% slower at M = 1, where 15 of the 16 WMMA rows are padding, and
  7-10% faster at M = 2 (35/40 shapes; the other 5 lose by at most 3.4%, 0.1-0.8 µs, within run-to-run
  noise). From M = 3 it wins on all 40 shapes.
- **SIMT vs WMMA, fp32.** SIMT wins through M = 4 (WMMA 0.84x in total). Above that the faster kernel depends
  on N: with few output columns the 16-row WMMA tiles leave too few blocks to stream the weights. WMMA for every
  shape from M = 6 lost by up to 1.64x (M = 6) and still 1.13x at M = 24 on the narrowest weights. The dispatch
  picks SIMT below an N cutoff that falls with M (8192 at M = 5 down to 1024 at M = 16-31, WMMA always from
  M = 32), fitted on two runs and checked at M = 5, 7, 10, 14, 20: the default dispatch is within 2.2% of the
  per-shape faster kernel in total time at M = 5 and within 0.3% from M = 6, worst shape 1.10x at M = 5 and
  at most 1.08x from M = 6 (was 1.64x).
- **Fused vs dequant+`F.linear`, fp16/bf16.** WMMA wins on 37-40/40 shapes at M = 192 and 256 (total 1.29-1.45x,
  worst 0.93x). At M = 384 it still wins in total (1.18x) but loses on 10-13 shapes by up to 19%, and at M = 512
  it ties in total, so the cap stays at 256.
- **Fused vs dequant+`F.linear`, fp32.** The fused path wins on every shape through M = 512 (total 1.36x at
  M = 512, worst 1.01x), so gfx120x fp32 keeps the fused path to M = 512; the global ROCm cap was raised to 512
  for this, and every other arch/dtype still stops at 256 or below.

The M <= 16 bf16/fp16 rule splits N >= 3840 into two leaves (`1,1,1,2,2,2` below N = 6144, `1,1,1,2,4,2` above):
against a single tile for N >= 3840 it is 1.5% faster in total at every M <= 16 and reduces the M = 2 shapes that
lose to SIMT from 9-10 to 5 (worst 0.93x to 0.97x). On the sweeps it is within 1.014x of the per-shape best tile
(was 1.033x).

Known losses: fp16 at M = 192/256 on 2-3 shapes vs dequant+BLAS (up to 7%); fp16/bf16 at M = 2 on 5 shapes
(up to 3.4%); fp32 at M = 5-7 up to 8-10% on a few shapes near the N cutoffs.

Where the time goes: through M = 16 the kernel streams weights at 500-590 GB/s on the large layers,
85-92% of the DRAM bandwidth. From M = 64 it plateaus at 65-80 TFLOPS, about 40% of the dense fp16 WMMA
peak. The latency rises in steps that follow the tile height (fp16 median 36 µs at M = 8-16, 46 µs at
M = 24-32, 64 µs at M = 48-64): rows inside a tile are almost free, the next tile height costs a step. The
best swept tile per shape shows the same steps, 16-row tiles at M = 24-48 are 1.23-1.37x slower (each row
tile streams and decodes the weights again), and skipping fully padded sub-tiles did not help (see rejected
experiments). The derived rules are within 1.03x (fp16/bf16) and 1.004x (fp32) of the per-shape best swept tile in
total time.

## Kernel notes

### Half-wave WMMA on gfx11

In wave32, `v_wmma_f32_16x16x16_{f16,bf16}` executes as two independent 8x16x16 products, one per half-wave:

- half-wave h computes rows 2j+h;
- it uses its own copy of B;
- it takes A from the lanes with c % 2 == h.

The ISA manual says to replicate operands across the halves. The kernel instead gives each half a different 32-element K slice and adds the two halves with `permlanex16` at the end. This doubles the useful K per WMMA. The behaviour is undocumented but was verified with a probe kernel.

Each accumulator element j in lane L holds row 2j + L/16 and column L%16.

### WMMA on gfx12

`v_wmma_f32_16x16x16_{f16,bf16}` on gfx12 (the `_w32_gfx12` builtins, 8 16-bit operand elements per lane) is
one real 16x16x16 product with no half-wave duplication. The probe in
`experiments/wmma_gfx12_layout_probe.hip` gives:

- A: lane L holds row L%16, K = 8*(L/16) + e for element e = 0-7.
- B: lane L holds column L%16, K = 8*(L/16) + e.
- C/D: element j of lane L holds row j + 8*(L/16), column L%16.

Only the pairing of A and B K indices matters, so each half-wave still takes its own 32 contiguous K of a
column (one 16-byte B load per 64-K unit), now summed inside the WMMA as 4 WMMAs of 8 K per lane. No
`permlanex16` is needed, and all 32 lanes store results.

Because the two halves feed the same accumulator, their K must share an absmax. For blocksize >= 64, half
h takes K [64u + 32h, +32) of unit u. For blocksize 32, it takes 16 K from each 32-K block of the unit (two
8-byte loads), and the absmax groups cover 2 WMMAs. The first port kept the gfx11 K split and was off by
~16% whenever blocksize < 256; `check.py` caught it.

Each WMMA covers 16 rows, so the kernel's M sub-tile is 16 rows (8 on gfx11) and the A stages in LDS are twice
as large. The kernel is templated on the WMMA generation so the host can size tiles, and a kernel body is
only emitted in the device pass of its own generation. The gfx11 code generation is unchanged apart from
register allocation: over the 49 shipped gfx1100 kernels the instruction count moved by -0.09%, scratch
use is identical, and every VGPR change stays inside the same 16-register allocation granule.

Other gfx1201 WMMA variants were checked and rejected on accuracy: `iu4`/`iu8` need integer activations
and an integer codebook, `fp8`/`bf8` would round activations to 3 mantissa bits, and the f16-accumulate
variant loses precision. `16x16x32_f16` needs gfx1250.

### Structure

- **Weight decode.** Weights are decoded through a 256-entry LDS table that maps a byte to a packed pair of centroids. The absmax scale is applied in fp32 after the WMMAs; blocks with the same absmax group share it.
- **Split-K.** Warps split K and reduce the partial results deterministically in LDS.
- **Operand staging.** A is staged in LDS (double-buffered, shared by the warps). B is streamed per lane from global memory with a one-deep prefetch.
- **K-chunk rotation.** Each block starts at `blockIdx.x % num_chunks`. This removes DRAM channel camping on power-of-two row strides, which was worth up to 3-4x on those shapes.
- **fp32.** A and the centroids are split into bf16 hi/mid/lo parts. Six products (lo*hi, hi*lo, mid*mid, mid*hi, hi*mid, hi*hi) give about 3-5e-6 worst-case relative error. `tests/test_functional.py::test_matmul_4bit_fp32_precision` guards against reduced-precision paths.

### Compiler bug workaround

AMD clang 24.0.0git (ROCm 10.2 nightly) miscompiles `__builtin_bit_cast(float, v[e])` on an `ext_vector_type` element: it always reads element 0. The workaround is to copy the element to a scalar first.

- The repro is in `experiments/clang_bitcast_vector_element_repro.hip`.
- Also avoid the HIP `uint4` class type in per-lane register arrays: copies become memcpy through scratch. Use `ext_vector_type` instead.

## Performance limits and rejected experiments

**Where the time goes:**

- At M=1 the kernel reaches about 80-100% of the streaming-read ceiling on large weights.
- WMMA wins at M=1 overall (1.08x total time vs SIMT). It still loses about 10% on three tall-K down projections with N of about 5120, and about 5% on tiny weights that sit at the launch floor.
- The steps from M of about 24 onward come from roughly 50% WMMA efficiency plus costs that are paid once per row tile. Smoothing them out likely needs a redesign that decodes B into LDS once and reuses it across larger M tiles.

**Rejected experiments** (all at default clocks, real NF4 weights):

| experiment | result |
|---|---|
| 3-slot B register ring, which removes the `vmcnt(0)` forced by the prefetch copy | Neutral: -2 to -4% at M <= 16, 0 to +5% above. Kernel kept in `experiments/`. |
| Per-lane `[nibble][lane]` conflict-free lookup table | No gain at small M, up to 1.5x slower at M >= 32. |
| 64- and 128-row block tiles | Slower than two passes of 32-row tiles. |
| Nontemporal B loads | Mixed. |
| Two-deep prefetch | Slower. |
| gfx1201: skip the WMMAs of 16-row sub-tiles that lie entirely past M (warp-uniform runtime check), to flatten the latency steps | Rejected. With the shipped tiles 2-13% slower at M = 16/32/64/128/256 (nothing to skip; the checks in the unrolled loop cost more than they save) and neutral at M = 40-48; best tile vs best tile only 2-6% faster at M = 40-48/80. The steps follow the tile height; padded WMMAs are not where the time goes. |
| gfx1201: decode B once per block into LDS (fragment-ordered, double-buffered, 64 KB budget) so warps along M share it (`experiments/gemm_4bit_wmma_blds.patch`, `data/gfx1201/experiments/sweep_fp16_blds.csv`) | 15-31% slower in total at every M = 16-256 (best tile vs best tile), no shape wins. The per-warp register decode is already amortised over WM row tiles, and the extra LDS stage costs more than it saves. |

## Enabling another architecture

1. Enable the target in `gemm_4bit_wmma.cu`:
   - its device pass must emit the kernel body (`BNB_WMMA_DEVICE_ARCH`, from `IS_RDNA3` / `IS_RDNA4` in `common.cuh`);
   - add the chip to `wmma_arch_of()`, which maps the device name to a WMMA generation. It must only list chips whose
     body is emitted, or the launch would run an empty kernel;
   - a new generation also needs its own `BNB_WMMA_EMIT_*` guard, so each device pass only instantiates its own kernels.
   - gfx115x (RDNA3.5) uses the gfx11 WMMA encoding but was not validated.
   - gfx1200 shares gfx1201's WMMA and its kernels are built, but it is not in `wmma_arch_of()` or the Python check: it
     has half the CUs, so gfx1201's tile rules and N cutoffs would not transfer. It keeps SIMT plus dequant+`F.linear`
     until it is calibrated.
   - CDNA uses MFMA and wave64; gfx1250 has different (wider) WMMAs.
2. Probe the operand and accumulator layouts before trusting the fragment mapping.
3. Run `check.py`, `check_f32.py` and the relevant pytest cases:
   ```bash
   pytest tests/test_functional.py -k "gemv_4bit or matmul_4bit"
   ```
4. Sweep, derive rules, add an arch-specific rule set in `launch_gemm_4bit_wmma`, and set the Python cutover (`_gemm_4bit_use_custom_rocm` in `bitsandbytes/backends/cuda/ops.py`) from `eval_*` data taken at default clocks.
5. Commit the data under `data/<arch>` and the charts under `plots/<arch>`.

## Known issues

ROCM-31245: on ROCm 10.2 nightlies from 10.2.0a20260913 (including the 0928 used here), every small fp32
GEMM on gfx12 takes ~5-20 ms. hipBLASLt was split into `libhipblaslt.so.1` and `libtensilelite-host.so.1`,
which duplicated the TensileLite `Debug` singleton, so solution lookup runs on every call. The rocBLAS
fallback goes through the same lookup. The fix is ROCm/rocm-libraries#12465. `hipblaslt_fix_overlay.sh`
builds the nightly's exact hipBLASLt revision plus that fix (host code only, the wheel's device objects are
reused) and loads it in place of the wheel's copy; with it a 16x2048x2048 fp32 GEMM takes 0.08 ms instead of
19 ms and stays IEEE-accurate (~1e-6 relative to fp64). The `eval_*` data uses it; `baseline_dequant.csv` does not.

With this toolchain, 23 fp32 `test_gemv_4bit` cases fail their tight tolerance on `main` as well. The failures predate the WMMA work and are not caused by it: torch's fp32 reference is the less accurate side.

On gfx1201 it is 16-22 cases per run (the test is unseeded), the same with `BNB_GEMM4BIT_KERNEL=simt` and
`=wmma`: `test_gemv_4bit` calls the separate `cgemm_4bit_inference_naive_*` GEMV kernel, not this GEMM.
Against a float64 reference that kernel is ~4x more accurate than torch's fp32 `F.linear`.
