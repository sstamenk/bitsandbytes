# AMD WMMA 4-bit GEMM: benchmark and tuning harness

Harness, data and notes behind the RDNA3 WMMA kernel for the fused 4-bit GEMM
(`csrc/gemm_4bit_wmma.cu`). Use it to re-tune the tile rules and to enable more
architectures.

Dispatch on RDNA3 (gfx110x) with `K % 64 == 0` and blocksize >= 32:

| dtype | kernel choice |
|---|---|
| fp16 / bf16 | WMMA for every M the Python side routes to the custom kernel (M <= 256) |
| fp32 | SIMT for M <= 5, WMMA for M = 6-128. fp32 is emulated with 3-way split bf16 operands (6 WMMAs per tile) and is fp32-accurate. |

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

## Scripts

| script | purpose |
|---|---|
| `bench.py` | Times SIMT / WMMA / dequant+BLAS for the chosen dtypes, M values and shapes. Each fused kernel runs in its own subprocess. |
| `par_bench.sh <tag> "<gpus>" <shape idx list> <kernels> [bench args]` | Runs `bench.py` sharded across GPUs. |
| `sweep.py` | Times every candidate tile per (shape, M) in one process through `BNB_WMMA_CFG`. |
| `par_sweep.sh <tag> "<gpus>" <shape idx list> [sweep args]` | Runs `sweep.py` sharded across GPUs. |
| `derive_cfg.py sweep.csv` | For each M, finds the best single tile and the best 2-way split on a shape feature, scored by geomean slowdown vs the per-shape best. |
| `best_vs_base.py sweep.csv bench.csv` | Per-shape speedup of the best swept tile vs SIMT and dequant. |
| `summarize.py bench.csv` | Median latency per M and kernel, plus a count of wins. |
| `plots.py` | Charts and a markdown table comparing the new default dispatch to the previous one. Set `PLOT_FILES`, `PLOT_OUT` and `PLOT_CLOCKS`. |
| `check.py` | fp16/bf16 correctness sweep over odd shapes, blocksizes, nested stats and bias (`BNB_GEMM4BIT_KERNEL` picks the kernel). |
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

## Kernel notes

### Half-wave WMMA on gfx11

In wave32, `v_wmma_f32_16x16x16_{f16,bf16}` executes as two independent 8x16x16 products, one per half-wave:

- half-wave h computes rows 2j+h;
- it uses its own copy of B;
- it takes A from the lanes with c % 2 == h.

The ISA manual says to replicate operands across the halves. The kernel instead gives each half a different 32-element K slice and adds the two halves with `permlanex16` at the end. This doubles the useful K per WMMA. The behaviour is undocumented but was verified with a probe kernel.

Each accumulator element j in lane L holds row 2j + L/16 and column L%16.

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

## Enabling another architecture

1. Gate the kernel body (`BNB_WMMA_RDNA3` in `gemm_4bit_wmma.cu`) and `gemm_4bit_wmma_supported()` for the new target.
   - gfx115x (RDNA3.5) uses the same WMMA encoding but was not validated.
   - gfx12 (RDNA4) has different WMMA operand layouts and no half-wave duplication, so it needs a new fragment mapping.
   - CDNA uses MFMA and wave64.
2. Probe the operand and accumulator layouts before trusting the fragment mapping.
3. Run `check.py`, `check_f32.py` and the relevant pytest cases:
   ```bash
   pytest tests/test_functional.py -k "gemv_4bit or matmul_4bit"
   ```
4. Sweep, derive rules, add an arch-specific rule set in `launch_gemm_4bit_wmma`, and set the Python cutover (`_gemm_4bit_use_custom_rocm` in `bitsandbytes/backends/cuda/ops.py`) from `eval_*` data taken at default clocks.
5. Commit the data under `data/<arch>` and the charts under `plots/<arch>`.

## Known issues

With this toolchain, 23 fp32 `test_gemv_4bit` cases fail their tight tolerance on `main` as well. The failures predate the WMMA work and are not caused by it: torch's fp32 reference is the less accurate side.
