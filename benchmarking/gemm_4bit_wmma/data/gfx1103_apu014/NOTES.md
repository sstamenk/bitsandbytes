# gfx1103 (Radeon 760M) measurements on CS-WSPG-APU014

A second Radeon 760M machine. The earlier gfx1103 data (`../gfx1103/`, the fp32 cutoffs and the 384 limits) came from
a 760M with one DDR5-5600 SO-DIMM (64-bit, 44.8 GB/s peak); this one has 128-bit LPDDR5X-7500 (120 GB/s). The tile
preferences of the two machines differ at M <= 32 (section "The other 760M: a risk signal at M <= 32").

## Machine and versions

- OrchestrAI Playground CS-WSPG-APU014 (AMD Birman-PHX reference board, BIOS WBI3823N_Weekly_23_08_2), Ubuntu 24.04.5,
  kernel 7.0.0-43-generic, Ryzen 5 7640HS (6 cores / 12 threads).
- Radeon 760M (Phoenix1, gfx1103, PCI 1002:15bf rev 01, VBIOS 113-PHXGENERIC-001): 8 CUs (KFD `simd_count` 16), 2 MB
  L2, no L3 / Infinity Cache, 512 MB carve-out + 7.5 GB GTT (torch sees 7642 MB). It is card1.
- Memory: 16 GB LPDDR5X-7500, four Micron MT62F1G32D2DS-026 (4 GB, 32 bits wide each; dmidecode lists them as four
  "DIMMs" on channels A-D) = a 128-bit bus, 7500 MT/s x 16 B = 120 GB/s peak. Check against the measurements: with the
  best swept tile the large layers stream 81-86 GB/s at M = 1 (gemma3-27b down 86.1, gemma2-27b gate/up 83.1,
  qwen3-32b down 82.4 GB/s), more than a 64-bit bus could deliver (60 GB/s); with the gfx1100 tiles 53-65 GB/s.
- TheRock nightly `rocm[libraries,devel,device-gfx1103]==10.2.0a20260928`, `torch 2.14.0+rocm10.2.0a20260928`,
  `amd-torch-device-gfx1103` (same nightly), triton 3.8.0, Python 3.12.3, cmake 3.28.3, uv 0.13.9.
- Builds: `cmake -G "Unix Makefiles" -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH=gfx1103 [-DBNB_WMMA_DEV=ON]`
  (`scripts/build.sh`). Base = 540f377 (gfx1103 on the gfx1100 tile tables, own fp32 cutoffs, limits 384 / 384), trees
  `src` (dev) and `rel` (release); final = the gfx1103 tuning commit, trees `fin3` (dev) and `relfin3` (release).
  Two earlier candidates were also built: `fin1` / `relfin1` with `cand/S1.py` and `fin2` / `relfin2` with
  `cand/S7.py` as the bf16/fp16 table. All three have the same fp32 tables, cutoffs and limits; S1 differs from the
  final table at M = 17-32 and M = 65-96, S7 only at M = 65-96. Dev builds for timing, release builds for pytest and
  the correctness checks.
- Clocks: `power_dpm_force_performance_level=high` on card1 for every timing job, `auto` restored on exit
  (`scripts/job.sh`, trap). In every 10-s sample with the GPU busy the sclk level and `freq1_input` read 2599 MHz
  and mclk 937 MHz, except in the rest of job `altF` after its GPU reset and in the first attempt of `evalF3` (both
  below) (`logs/*.clk`, `analysis/clocks.txt`); edge temperature 56-66 C median per timing job, at most 73 C; package
  power 35-47 W median per timing job, at most 58.9 W.
  The samples are 10 s apart, so they do not show the clock in between.
  Every timing job ran under `runs_guard.sh` (every `logs/*.log` ends in CLEAN and every `.foreign` is empty = no
  GPU process of another user; the two jobs the reset watchdog stopped, `probe384` and `r2_fp32_afq_retry`, have no
  verdict line) and `logs/*.kfd` lists every KFD process every 5 s (only our python
  workers); one serial queue (`scripts/queue.sh`, `logs/queue.log`, `logs/q1.txt`), builds and pytest as queue jobs
  between the timing jobs. One exception: a 12-call `kname.py` run (a few seconds) was started by hand at 00:44 UTC
  while `alt2` was warming up; the first shape of `alt2_fp16` shows no outlier cycle.
- Harness: `BNB_BENCH_SLEEP_CYCLES=2e6` for every job (bench.py's idle sleep before a timed batch, 5e7 = 0.5 s by
  default; checked on the other 760M: median ratio 1.003, same best tile in 9/9 cases). Sweeps (`scripts/sweep1103.py`
  = `sweep.py` plus a 5-s warm-up, 3 s idle and a 2-s re-warm before each shape, and kch = 2 candidates also for
  K % 128 != 0 so that lookups are prescale-aware). In-process interleaved comparisons (`scripts/alt1103b.py`): for
  every (shape, M) the configurations run in alternating order (forward on even cycles, reverse on odd) on the same
  weight copies and activations after one untimed pass; "total" = sum over the shapes of the per-shape median over the
  cycles; "cycles a-b" = range of the same ratio computed per cycle.
- First-in-process readings: the first measurements of a worker can read 2-3x high (llama3-70b down as the first
  shape: fp32 `auto` M = 1-4 and bf16 `auto` M = 1 in `evalA`; a sweep warm-up read 5.5 ms for ~2 s before settling
  at 2.1 ms). Sweeps and interleaved runs warm up first; the final evals time one shape twice and keep the second pass
  (`scripts/q_evalF.sh`, `scripts/dedup.py`); the five base cells were re-measured (`evalA_fix`).
- GPU resets: thirteen in the session (`logs/dmesg_gpu_resets.txt`), all on one driver path: a KFD queue eviction
  after a userptr invalidation ("Failed to quiesce KFD", "Failed to evict process queues") in which
  `MES failed to respond to msg=REMOVE_QUEUE`, then a MODE2 reset ("device wedged, but recovered through reset"); five
  of them are the same timeout when a process left with a wedged context exited or was killed (17:20:22, 09:14:30,
  09:16:41, 09:19:22, 09:42:03 UTC). The kernel log since boot has no amdgpu page fault, OOM-killer, BUG or Oops line.
  The eight primary ones:
  - 17:20 UTC: the first pytest run (`test_4bit_quant_large` out of memory).
  - 02:02 UTC: queue job `altF` (build fin1), in an fp32 A / F run that had finished its 13th shape and was setting
    up the next one (680 MB of fp32 weights); python got `hipErrorLaunchFailure` at the next call. Both tables had
    completed all 5 cycles of every M on that shape.
  - 09:10, 09:18, 09:41, 09:43, 09:56 UTC: at the first dequant + `F.linear` call of an fp32 process (the first BLAS
    call of the process; llama3-70b down, 0.94 GB of fp32 weights), about 1 s after it: `r2_fp32_afq` twice
    (configurations A, F, Q; M = 256), `probe384` (A, F, Q; M = 384), `probe256` (F, Q; M = 256) and `probe256s`
    (F, Q; M = 256, queue idle before the call). The same call ran without a reset in `cap` (F, Q; M = 384; build
    fin1; 02:38 UTC), in the base eval (dequant only) and in `r2_fp32_fq` (F, Q; M = 384; 09:57 UTC).
  - 10:52 UTC: `r2_fp32_fq` had finished 39 of its 40 shapes and was setting up the last one (llama3-70b gate/up:
    `torch.randn` of 0.94 GB fp32, then `quantize_4bit`, whose code-table upload raised the error).
  What the cases with a traceback have in common is a host-to-device copy or a BLAS library load issued while a long
  kernel on 0.94 GB of fp32 data was still queued; every primary case on the second day was in an fp32 process, none
  in the bf16/fp16 runs (which use the same harness with half the bytes). This is read as a driver / MES firmware failure
  to evict a busy queue, not as a problem of the fused kernels or tiles (hypothesis; not isolated, and timing
  dependent). From 10:56 UTC the fp32 runs of `alt1103b.py` synchronize at those points (`--syncwarm`:
  one small `F.linear` at the start, `torch.cuda.synchronize()` after `torch.randn` / `quantize_4bit` of a shape and
  around the untimed calls before each timed batch; timed batches unchanged, but the extra `F.linear` shifts the
  random inputs, so such a run is not a bit-identical repeat of one without it). No reset in the 3.4 h of runs after
  that (fp32 with dequant three times, fp32 tile runs, bf16/fp16).
  Consequences for the data:
  - After a reset the level `high` no longer holds the clock until it is written again: in the rest of job `altF`
    `pp_dpm_sclk` listed 800 / 2600 / 2599 MHz, 84 of the 93 busy samples read 2600 MHz, the other nine between 800
    and 2596 MHz, and an idle sample 800 MHz. The fp32 runs of that job (`pinned/alt_s1/fp32_{hi,x,d}.csv`) are
    therefore not used as evidence; `fp32_lo` was repeated in a new job (`altF2`, clocks pinned again; its times are
    within 0.3% of the 13 shapes done before the reset), and the fp32 comparison was run again on the later builds.
  - Queue job `evalF3` first started in the second of the 09:14:30 reset: the level read back empty and sclk was not
    pinned (the sclk table read 2585-2600 MHz, `freq1_input` as low as 2505 MHz). It was stopped and run again; the
    first attempt's logs are kept as `logs/evalF3_unpinned_after_reset5.*`, its output is not used.
  - fp32 with the three configurations A, F, Q in one run (`r2_fp32_afq`, review round 1) could not be measured;
    it was replaced by `fp32_fq` (F, Q; M = 384-512), `fp32_fq256` (F, Q; M = 256) and `fp32_af` (A, F;
    M = 256-512), each a separate process; the base tiles were not timed against dequant in one process.
  - `job.sh` and `queue.sh` were changed twice because of this (the versions each job ran with are in
    `scripts/old/`). From 09:17 UTC: the level is rewritten until `pp_dpm_sclk` shows the pinned state, the reset
    count is recorded at the start and the end of a job, a job in which a reset happens is stopped and returns 99,
    and the queue runs it again, at most twice. From 11:58 UTC (review round 2): a job whose level never reads back
    pinned is not started (status 98, the queue stops); after a reset every file the job wrote under `data/` is
    renamed `*.reset_<job>_<time>`, so that a repeat cannot reuse it; a job returns its command's status and the
    `q_*.sh` scripts return 1 when a run in them failed (before, such a job was logged as `rc=0`, e.g.
    `logs/r2_fp32_fq.meta`).
  - `analysis/csv_provenance.txt` (`scripts/audit_csv.py`) lists for every CSV in `pinned/` the job that wrote it
    (by the time of its last write, `logs/csv_mtimes.txt`), the level read back at that job's start, the busy clock
    samples of the job that were not at 2599 MHz and the resets inside the job. Result: every CSV
    used as evidence was written by a job whose level read back `high` at its start, with every busy sample up to the
    file's last write at 2599 MHz and no reset in the job (the jobs before 09:17 UTC did not verify the sclk state at
    the start; their samples show it). The flagged files are all in `pinned/alt_s1/` (job `altF`, build fin1, the
    02:02 reset: `fp16`, `bf16`, `*_x`, `*_m1` were finished 9-28 min before it and the later `altF_retry` job kept
    them; they are the earlier candidate S1's runs and back no number in this file; `fp32_{hi,x,d}` and the
    13-shape part as described above) plus `r1/fp32_fq_part39_before_gpu_reset.csv`. No retry after a reset reused a
    CSV of the failed attempt: `r2_fp32_afq` and the probes wrote none, the first `evalF3` attempt's files were moved
    away and measured again, and the `fp32_fq` repeat measured all 40 shapes again.
  - `r2_fp32_fq` was ended by the 10:52 reset after 39 of its 40 shapes. Its rows (all written before the reset,
    clock pinned in every sample) are kept as `pinned/r1/fp32_fq_part39_before_gpu_reset.csv` and used only as a
    cross-check, as is `fp32_fqb.csv` (the missing shape and llama3-70b down, a separate job); the complete run was
    repeated in a new job without a reset (`fp32_fq.csv`), and that run alone is the evidence.

## Files (NF4, blocksize 64, 40 shapes unless noted)

| file | contents |
|---|---|
| `pinned/eval_base.csv` | Base build: `auto`, `dequant` at 56 M values 1-512, `simt` and fp32 `wmma` at 32 M values 1-64 (`pinned/evalA/`; bf16/fp16 `wmma` not run: WMMA at every M, the same as `auto`). The `auto` rows of the first shape at M <= 4 are from `pinned/evalA_fix/` (see first-in-process readings). |
| `pinned/eval_final.csv` | Final build (fin3): `auto` and `wmma` at the 56 M values (`pinned/evalF/`); `simt` and `dequant` are the base rows (same kernels, same BLAS). Both files are built by `scripts/final_an.py`. |
| `pinned/sweep/sweep_fp16{a,b,c}.csv` | fp16 tile sweep, every candidate tile, 23 M values 1-512 (a: 1, 4, 8, 16, 32, 64, 128, 256; b: 2, 12, 20, 24, 28, 40, 48, 96, 192, 384; c: 56, 80, 112, 160, 512). As in `sweep.py`, tiles with fewer than 32 rows are skipped at M >= 64. |
| `pinned/sweep/sweep_fp16e.csv` | fp16, M = 80, only the tiles with at most 16 rows. |
| `pinned/sweep/sweep_bf16{a,b}.csv` | bf16 tile sweep at M = 1, 4, 8, 16, 32, 64, 128, 256 (a) and 80, 96 (b). |
| `pinned/sweep/sweep_fp32{a,b,d}.csv` | fp32 tile sweep: a = M 3, 4, 8, 16, 32, 64, 128, 256 (>= 32 rows at M >= 64), b = M 6, 12, 24, 48, d = M 64, 128, 256 with every tile. `sweep_fp32c_part.csv`: the first shape of a longer run (M = 64-192) that was stopped for time. |
| `pinned/alt/alt1_{fp16,bf16}.csv` | Interleaved tile tables, base build: A = gfx1100 table, B = `scripts/rulesB.py`, C = `scripts/rulesC.py` (first fits on this machine); 25 M values 1-128, 5 cycles. |
| `pinned/alt/alt2_{fp16,bf16}.csv` | The same with A, B = `scripts/rulesB2.py` (the fit on this machine alone), S = `scripts/cand/S1.py` (the final table except at M = 17-32 and M = 65-96), T = `scripts/cand/S3.py`; 27 M values 1-128, 5 cycles. |
| `pinned/alt/alt32.csv` | fp32, every shape forced: S = SIMT, A = gfx1100 tiles, B = `scripts/rules32B.py` (the final table at M <= 48), D = gfx1150 tiles; 20 M values 1-80, 5 cycles. |
| `pinned/alt/rtf_{fp32,fp16}.csv` | Block order (`BNB_WMMA_ROW_TILES_FIRST=0/1` per call), M = 24, 48, 96, 192, 3 cycles. |
| `pinned/altF/*.csv` | Final dev build (fin3), interleaved: A = the base dispatch through overrides (gfx1100 tables; fp32 with the gfx1103 cutoffs), F = default dispatch of the build. `fp16`, `bf16`: 31 M values 1-512, 5 cycles; `*_x`: F against the final table through the same override (must match); `*_m1`: F against forced SIMT at M = 1, 2. `bs32_{fp16,bf16}`, `bs128_fp16` (blocksize 32 / 128) and `nest_{fp16,bf16,fp32}` (double-quantized absmax): 3 cycles. fp32: `fp32_lo` (23 M values 1-160, 5 cycles), `fp32_x` (F against `cand/F1.py` through the override), `fp32_d` (F against the gfx1150 fp32 table), `bs32_fp32` (blocksize 32), 3 cycles; the fp32 runs use `--syncwarm` (see GPU resets). |
| `pinned/altF_5c/*.csv` | The blocksize 32 / 128 and double-quantized runs of `pinned/altF/` again with 5 cycles (`bs32_{fp16,bf16,fp32}`, `bs128_fp16`, `nest_{fp16,bf16,fp32}`; review round 2). |
| `pinned/r1/*.csv` | Review rounds 1 and 2. `r1_{fp16,bf16}`: M = 49-96, gfx1100 table against the default dispatch of build fin2 (one tile for 49-96), 10 cycles; `abf_{fp16,bf16}`: M = 17-32, gfx1100 table, `rulesB2.py` and the default dispatch (fin2), 5 cycles; `cap_bs32_{fp16,bf16}`: blocksize 32, fused against dequant at M = 384, 416, 448, 480, 512 (fin2), 5 cycles; `fp32_fq`: fp32, fused against dequant at M = 384, 416, 448, 480, 512 (fin3), 5 cycles; `fp32_af`: fp32 at M = 256, 384, 416, 448, 480, 512, base dispatch through overrides against the default dispatch (fin3), 5 cycles (the two replace one A / F / Q run that reset the GPU, see GPU resets); `fp32_fq256`: the same as `fp32_fq` at M = 256; `fp32_fq_part39_before_gpu_reset`, `fp32_fqb`: the first `fp32_fq` run (39 shapes, ended by a GPU reset) and shapes 0 and 1 in a separate job, cross-checks only; `st_{fp16,bf16}`: gemma3-27b down and gemma3-4b gate/up at M = 28 and 56, gfx1100 table against the default dispatch (fin3), 10 cycles. |
| `pinned/alt_s7/*.csv` | The earlier candidate `cand/S7.py` through its own build (fin2), as `altF`: `fp16`, `bf16`, `*_x`, `*_m1`; `bs32_{fp16,bf16}`, `bs128_fp16` (blocksize 32 / 128) and `nest_*` (double-quantized absmax), 3 cycles; fp32: `fp32_lo` (23 M values 1-160, 5 cycles), `fp32_hi` (256, 384, 512, 3 cycles), `fp32_x`, `fp32_d` (F against the gfx1150 fp32 table), `bs32_fp32`. |
| `pinned/alt_s1/*.csv` | The earlier candidate `cand/S1.py` through its own build (fin1): `fp16`, `bf16`, `*_x`, `*_m1`, `bs32_fp16`, `bs128_fp16`; fp32: `fp32_lo` (rerun in a fresh job), `fp32_lo_part13_before_gpu_reset` (the 13 shapes done before the GPU reset), `fp32_hi`, `fp32_x`, `fp32_d` (run after the reset in the same job, not used as evidence), `bs32_fp32`. |
| `pinned/altV/*.csv` | Base build, interleaved tables: `v32_{fp16,bf16}`: M = 17-32 at blocksize 32, A, S = `cand/S1.py`, V2, V3 (= the final bucket), B = `rulesB2.py`, 5 cycles; `v64_*`: A, S, V3 at blocksize 64, 5 cycles; `b32_bf16`: bf16 at blocksize 32, other buckets; `nest_*`: double-quantized absmax, A, S, V3 (fp32: gfx1100 tiles against `cand/F1.py`), 3 cycles. |
| `pinned/cap/cap_{fp16,bf16}.csv`, `cap_fp32.csv`, `cap_bs32_{fp16,bf16}.csv` | Build fin1 (the tiles at these M are the final ones), fused (default dispatch) vs dequant + `F.linear`, interleaved: M = 384, 416, 448, 480, 512 (5 cycles); fp32 M = 384, 512 (3 cycles); blocksize 32 at M = 384, 512 (3 cycles). |
| `pinned/evalH/` | Build fin1, bench.py single pass above the limit: `auto` and `dequant` at M = 576, 640, 768, 1024 (bf16/fp16) and 1024 (fp32). |
| `rorlab35/sweep_fp16_5shapes.csv` | The other 760M (RSB-RORLAB-35, one DDR5-5600 SO-DIMM): fp16 tile sweep of the 5 largest shapes at the 23 M values (4 shapes at M = 256, 384, 512: qwen3-32b down is missing there), one pass, taken 2026-11-20 with the same `sweep1103.py`, before that machine went down (edge temperature 91-105 C at level `high`). |
| `analysis/` | Outputs of `scripts/analysis.sh` (every table below). |
| `logs/` | Per job: guard log, `.foreign`, `.meta`, `.clk` (clocks, temperature, power every 10 s), `.kfd` (GPU processes every 5 s); `queue.log`, `q1.txt` (every job line as run); build, pytest, kernel-name and `tilecheck` logs; `dmesg_gpu_resets.txt` (the kernel log lines of the GPU resets); `data/` = the stdout of the sweep / interleaved runs and of the reset probes (`data/probe/`). |
| (all) | The repository's pre-commit hooks were applied to this directory: CSV line endings are LF, trailing whitespace is removed, and the typos hook rewrote two tokens in logs: in a profiler warning line of `logs/kname1103_rel.txt` and `logs/kname3.log` the prefix of the ROCr function name `..._amd_profiling_get_dispatch_time` (the letters h, s, a) reads `has`, and `.kfd` lines that the 150-character cut ended one letter short of `/home` read `/home`. Nothing else differs from the files on the machine. |
| `scripts/` | Job and queue scripts as run on the machine (`~/bnbtune` paths), `sweep1103.py`, `alt1103b.py`, the rule-table files, and the analysis scripts (`analysis.sh` regenerates `analysis/` from this directory). `old/`: the versions of `job.sh`, `queue.sh`, `q_evalF.sh`, `q_fin3.sh`, `q_r.sh` and `alt1103b.py` that earlier jobs ran with (the file names give the time span, UTC, 24 November). |

## Findings

"Total" is the sum over the 40 shapes at one M (or over the measured M of a bucket); "oracle" the per-shape minimum
over simt, wmma, dequant and auto; "best" the per-shape best swept tile.

### Base build (gfx1100 tiles, gfx1103 fp32 cutoffs, limits 384 / 384)

`analysis/dispatch_base_final.txt` (base part), `analysis/fp32_cutoffs_base_tiles.txt`:

- The dispatch is within 0.3% of the oracle in total at every M <= 384 for bf16/fp16 and within 0.9% at M >= 3 for
  fp32; fp32 M = 2 is 4.1% off (SIMT for every shape; WMMA already wins several large layers). So what the base
  build leaves on the table is tile choice and the limit, not the SIMT/WMMA split.
- Against the best swept tile the gfx1100 tiles are far off on this machine (next section).
- At M = 416-512 the base build falls back to dequant + `F.linear`, which is 1.70-1.80x (bf16/fp16) and 2.65-2.67x
  (fp32) slower in total than the fused kernel.

### Tile rules, bf16/fp16

Sweep (blocksize 64), table tile vs best, total per bucket (`analysis/rules_vs_best_fp16.txt`,
`rules_vs_best_bf16.txt`; fp16 / bf16 where bf16 was swept):

| M | gfx1100 (base) | gfx1101 | gfx1102 | gfx1150 | gfx1151 | fit on this machine (`rulesB2.py`) | final |
|---|---|---|---|---|---|---|---|
| 1-8 | 1.263 / 1.262 | 1.229 | 1.193 | 1.308 | 1.279 | 1.012 / 1.011 | 1.019 / 1.016 |
| 9-16 | 1.219 / 1.211 | 1.144 | 1.146 | 1.178 | 1.241 | 1.024 / 1.021 | 1.048 / 1.045 |
| 17-32 | 1.231 / 1.223 | 1.158 | 1.046 | 1.091 | 1.330 | 1.011 / 1.012 | 1.112 / 1.107 |
| 33-48 | 1.087 | 1.059 | 1.092 | 1.078 | 1.277 | 1.004 | 1.004 |
| 49-64 | 1.093 / 1.091 | 1.057 | 1.023 | 1.011 | 1.099 | 1.004 / 1.008 | 1.004 / 1.008 |
| 65-96 | 1.041 / 1.027 | 1.041 | 1.041 | 1.004 | 1.164 | = gfx1100 | = gfx1100 |
| 97-512 | 1.014 / 1.009 | 1.014 | 1.014 | 1.007 | 1.171 | = gfx1100 | = gfx1100 |

No existing table is within 3% in every bucket (gfx1102's comes closest at M = 17-32 and 49-64, gfx1150's at
M = 49-96), so gfx1103 gets its own through M = 64. Above M = 96 the gfx1100 bucket stays (1.4% / 0.9% off best; a
refit would gain about 1%); M = 65-96 also keeps the gfx1100 bucket (below). At M = 49-64 one 32-row tile,
`{4, 2, 1, 4, 1, 2}`, is within 0.8% of best for bf16 and fp16.

Final table (`wmma_rules_gfx1103_f16`, `scripts/cand/S8.py`):

| M | rule |
|---|---|
| 1-8 | K < 8192: `{1, 2, 1, 1, 4, 2}`, else `{1, 2, 1, 1, 8, 2}` |
| 9-16 | `{2, 1, 1, 4, 2, 2}` |
| 17-32 | `{4, 1, 1, 4, 2, 2}` |
| 33-48 | K < 12288: `{2, 2, 1, 4, 1, 2}`, else `{4, 2, 1, 4, 1, 2}` |
| 49-64 | `{4, 2, 1, 4, 1, 2}` |
| 65+ | the gfx1100 buckets |

It is not the blocksize-64 fit on this machine alone (`rulesB2.py`). The fit is rejected at M = 17-32 on evidence
from this machine (blocksize 32), and at M <= 16 the final table takes slightly more conservative tiles because of a
risk signal from the other 760M.

#### Blocksize 32 and double-quantized absmax

Every sweep is at blocksize 64 with plain fp32 absmax. Checked afterwards, in-process interleaved against the gfx1100
table (`analysis/m17_32_blocksize_nested.txt`, base build; `analysis/earlier_candidates.txt`,
`final_blocksize_nested.txt`):

- Blocksize 32, M = 17-32 (5 cycles, all 40 shapes, fp16 / bf16, total against the gfx1100 table): the tile
  `{4, 2, 1, 4, 2, 2}` is slow there. The fit (`rulesB2.py`, that tile for K >= 10240) is 1.065 / 1.059, i.e. SLOWER
  than the gfx1100 table, with llama3-70b down 1.6x slower; the first candidate `cand/S1.py` (that tile for
  K / N >= 1/2), which had already been built and validated at blocksize 64, 1.187 / 1.184. One tile for the bucket,
  the gfx1100 tile of its wide layers `{4, 1, 1, 4, 2, 2}`, is 0.945 / 0.945 at blocksize 32 (worst shape 1.02x) and
  0.906 / 0.907 at blocksize 64 (5 cycles, cycles 0.902-0.909): that is the final bucket. `cand/V2.py`
  (`{4, 2, 1, 4, 1, 2}` instead) is faster at blocksize 32 (0.89-0.90) but 1.05-1.18x slower on small GQA k/v there;
  not taken.
- The other final buckets (final build, 5 cycles, all 40 shapes at 4-9 M values, `pinned/altF_5c/`; total F / A
  against the gfx1100 table):

  | | M = 1 | 8 | 16 | 24 | 32 | 48 | 64 | 96 | 128 |
  |---|---|---|---|---|---|---|---|---|---|
  | blocksize 32, fp16 | 0.849 | 0.858 | 0.952 | 0.948 | 0.944 | 0.925 | | 0.997 | |
  | blocksize 32, bf16 | 0.848 | 0.854 | 0.951 | 0.948 | 0.943 | 0.925 | | 1.000 | |
  | blocksize 128, fp16 | 0.805 | 0.801 | 0.860 | 0.912 | 0.903 | 0.922 | | 1.000 | |
  | double-quantized absmax, fp16 | 0.794 | 0.791 | 0.851 | 0.905 | 0.900 | 0.927 | 0.921 | 1.000 | 1.000 |
  | double-quantized absmax, bf16 | 0.793 | 0.792 | 0.851 | 0.905 | 0.899 | 0.928 | 0.924 | 1.000 | |

  At blocksize 32 the gains at M <= 32 are smaller than at blocksize 64 (15% instead of 19% at M <= 8, 5% instead
  of 14% at M = 16, 5-6% instead of 9-10% at M = 24-32), but every own bucket is still faster in total. At M = 16
  and blocksize 32 five shapes are slower than with the gfx1100 table, in both dtypes: 7-8b k/v 1.10x (fp16) /
  1.08x (bf16) and four down projections (qwen3-32b 1.07x, llama3-70b 1.06x, qwen3-14b 1.05x, gemma3-27b 1.02x), against 35
  shapes that are faster (to 1.6x); the worst cell of every other M and setting is at most 1.02x.
  fp32: double-quantized absmax 0.971, 0.944, 0.861, 0.947, 0.776, 0.927 at M = 4, 8, 16, 48, 80, 128; blocksize 32
  0.956, 0.846, 0.950, 0.925 at M = 8, 16, 48, 128 (worst cell 1.01x). An earlier 3-cycle pass of the same runs
  (`pinned/altF/bs32_*`, `bs128_fp16`, `nest_*`) agrees within about 1% per M; its single cells above 1.05x were
  the same M = 16 cells and one that the 5-cycle pass does not show (qwen3-8b gate/up, M = 32, double-quantized,
  fp16: 1.08x, now 1.00x).

#### The other 760M: a risk signal at M <= 32

The other 760M (RSB-RORLAB-35, 44.8 GB/s) left one fp16 sweep of the 5 largest shapes (`rorlab35/`), taken in one
pass at 91-105 C edge temperature shortly before that machine went down. It cannot be repeated, so it is used as a
warning, not as proof: its tile preferences for the wide layers (N >= K: gate/up, o/q) at M <= 32 differ from this
machine's. A plausible reason is DRAM (that machine reaches 41 GB/s = 91% of its peak at M = 1, this one 68-72% of
its peak), but thermal drift or the fixed candidate order of the sweep could contribute; the cause is not
established, and neither is what other single-channel or ordinary dual-channel machines prefer. Per shape, tile vs
best on APU014 / on RORLAB-35 (sum over the bucket's swept M):

| bucket, shape | best here | best there | compromise |
|---|---|---|---|
| M <= 8, llama3-70b gate/up | `{1,1,1,4,2,2}` 1.00 / 1.20 | `{1,1,1,1,8,1}` (gfx1100) 1.25 / 1.02 | `{1,2,1,1,8,2}` 1.04 / 1.04 |
| M = 9-16, qwen3-32b gate/up | `{2,2,1,4,1,2}` 1.00 / 1.36 | `{2,1,1,4,4,2}` 1.16 / 1.01 | `{2,1,1,4,2,2}` 1.05 / 1.15 |
| M = 9-16, llama3-70b o/q | `{2,1,1,4,2,2}` 1.04 / 1.23 | `{2,1,1,4,4,2}` 1.12 / 1.06 | - |
| M = 17-32, qwen3-32b gate/up | `{4,2,1,4,1,2}` 1.00 / 1.29 | `{4,1,1,4,2,2}` (gfx1100) 1.22 / 1.03 | none within 10% on both |
| M = 17-32, llama3-70b gate/up | `{4,2,1,4,1,2}` 1.00 / 1.16 | `{4,1,1,4,2,2}` (gfx1100) 1.11 / 1.08 | - |

Table vs best, total per bucket, APU014 (40 shapes, fp16) / RORLAB-35 (5 shapes, fp16; 4 at M >= 256)
(`analysis/two_machines_fp16.txt`, `rorlab35_rules_vs_best_fp16_5shapes.txt`; joint fits in `two_machines_fit_fp16.txt`):

| M | gfx1100 | fit on this machine (`rulesB2.py`) | final |
|---|---|---|---|
| 1-8 | 1.263 / 1.028 | 1.012 / 1.069 | 1.019 / 1.023 |
| 9-16 | 1.219 / 1.072 | 1.024 / 1.135 | 1.048 / 1.101 |
| 17-32 | 1.231 / 1.052 | 1.011 / 1.110 | 1.112 / 1.052 |
| 33-48 | 1.087 / 1.185 | 1.004 / 1.072 | 1.004 / 1.072 |
| 49-64 | 1.093 / 1.173 | 1.004 / 1.072 | 1.004 / 1.072 |
| 65-96 | 1.041 / 1.094 | = gfx1100 | = gfx1100 |

Against the gfx1100 table the fit costs that sweep 4.0%, 5.8% and 5.6% in total at M = 1-8, 9-16 and 17-32; the
final table -0.5%, +2.6% and 0.0%. What the final table gives up here for it:

- M <= 8: the K < 8192 leaf instead of the fit's two leaves (llama3-70b gate/up there 1.20 -> 1.04 off best). The
  fit takes 0.5% less time than the final table here (measured, fp16 and bf16).
- M = 9-16: one tile; the fit's leaf for wide N with K < 6144 (`{2, 2, 1, 4, 1, 2}`) is dropped (qwen3-32b gate/up
  1.36 off best there; three cells of that sweep remain above 1.10x against the gfx1100 table, worst 1.12x). The fit
  takes 2.6% (fp16) / 2.5% (bf16) less time than the final table here (measured). `cand/S3.py` (K / N < 1/2 on
  split-K 4) would be 3.9% better than the gfx1100 table there but takes 4.2% more time than the final table here
  (measured, `analysis/alt2.txt`); not taken.
- M = 17-32: decided by blocksize 32 (above); the final tile is also the one that leaves that sweep unchanged.
  Measured directly at blocksize 64 (`analysis/fit_vs_final_m17_32.txt`, gfx1100 table / fit / final interleaved,
  5 cycles): fit 0.825 / 0.826 and final 0.907 / 0.907 of the gfx1100 table (fp16 / bf16), i.e. the fit takes 9.0%
  less time than the final table there (the final table 9.9% more than the fit).
- M = 33-64: the fit, better on both machines.

If neither that sweep nor blocksize 32 is to be considered, `scripts/rulesB2.py` is the table to use.

#### M = 65-96 keeps the gfx1100 bucket

One tile `{4, 2, 1, 4, 1, 2}` for M = 49-96 was the first choice (it is the sweep's best at M = 80-96, 1.000 / 1.001
of best for fp16 / bf16, and it was built and validated as `cand/S7.py`, `analysis/earlier_candidates.txt`). The
M = 65-96 part does not clear the ~3% bar in bf16: in the settling run (`pinned/r1/r1_{fp16,bf16}.csv`,
`analysis/r1_m49_96.txt`: final build of that candidate against the gfx1100 table, all 40 shapes, 10 alternating
cycles) it gains 3.01% in fp16 (A-first cycles 0.9701, F-first cycles 0.9694, per-cycle totals 0.967-0.972) and 2.56%
in bf16 (0.9745 / 0.9744, 0.970-0.977); per M 2.4-2.8% at M = 65-88 in both dtypes, 4.6% (fp16, prescaled) and 2.4%
(bf16) at M = 96. The M = 49-64 part gains 8.26% / 8.19% in the same run (A-first 0.9174 / 0.9179, F-first 0.9170 /
0.9178). So the own bucket ends at M = 64 and M = 65-96 keeps the gfx1100 row. No material regression was seen with
the candidate there (36-37 of 200 cells nominally slower, worst 1.008x; 33-35 of 40 shapes faster per M, up to 1.6x);
it is left out only because of the per-bucket bar.

#### Validation (in-process interleaved, 5 cycles, bf16 and fp16 separately)

Default dispatch of the final build (F, fin3) against the gfx1100 table through the tile override (A = the base
dispatch), all 40 shapes, 31 M values, 5 cycles (`pinned/altF/{fp16,bf16}.csv`, `analysis/final_interleaved.txt`),
total F / A and its range over the cycles:

| M (measured) | fp16 | bf16 | shapes faster per M, fp16 / bf16 |
|---|---|---|---|
| 1-8 (1, 2, 3, 4, 6, 8) | 0.808 (0.805-0.809) | 0.806 (0.803-0.811) | 40 / 40 |
| 9-16 (9, 12, 16) | 0.862 (0.858-0.865) | 0.861 (0.859-0.862) | 40 / 40 |
| 17-32 (17, 20, 24, 28, 32) | 0.908 (0.904-0.908) | 0.907 (0.902-0.908) | 28-32 / 29-34 |
| 33-48 (33, 40, 48) | 0.925 (0.922-0.927) | 0.927 (0.920-0.929) | 34-37 / 34-37 |
| 49-64 (49, 56, 64) | 0.917 (0.913-0.919) | 0.918 (0.918-0.923) | 34-36 / 33-35 |
| 65-96 (65, 72, 80, 88, 96) | 0.999 (0.994-1.000) | 1.000 (0.997-1.001) | the same tiles |
| 97-512 (97, 128, 192, 256, 384, 512) | 0.999 (0.998-1.002) | 1.000 (0.997-1.003) | the same tiles |

Per M the totals are 0.795-0.820 at M <= 8, 0.857-0.866 at 9-16, 0.903-0.911 at 17-32, 0.917-0.935 at 33-48,
0.913-0.922 at 49-64 and 0.997-1.001 above, in both dtypes.

Single cells (`analysis/final_same_tile_cells.txt`, `final_regressions.txt`): in each dtype the final table picks
another tile than the gfx1100 table in 650 of the 1240 (shape, M) cells, and every one of those is faster (F / A
0.44-0.97 for fp16, 0.45-0.98 for bf16). In the other 590 cells A and F run the same kernel; they read 0.94-1.05
(fp16) and 0.96-1.07 (bf16): one cell per dtype above 1.02 (fp16 gemma3-27b down at M = 28 1.050, F slower in
four of the five cycles; bf16 gemma3-4b gate/up at M = 56 1.069), ten per dtype above 1.01. The two cells were
measured again on the same build with 10 cycles (`pinned/r1/st_{fp16,bf16}.csv`): 1.000 (fp16 gemma3-27b down,
M = 28; cycles 0.984-1.008) and 0.993 (bf16 gemma3-4b gate/up, M = 56; cycles 0.977-1.006). They also read 1.001
and 1.000 in the fin2 run, and the `*_x` control below shows single cells up to 9% apart with the same kernels, so
the two readings are taken as the run-to-run variability of a cell. No cell is more than 10% slower.

- `*_x` (3 cycles, M = 1, 8, 16, 24, 32, 48, 64, 96): the default dispatch against `cand/S8.py` through the override
  is within about 0.5% in total in every bucket (0.995-1.001; single cells 0.90-1.09 with the same kernels), so the build
  runs the table the analysis scores. `scripts/pickcheck.py` compares the C++ `wmma_pick` of the header with the
  python lookups of `cand/S8.py` and `cand/F1.py`: 96000 picks, 0 mismatches.
- `*_m1`: forced SIMT against the default dispatch at M = 1 is 1.49x (fp16) / 1.42x (bf16) slower in total (SIMT
  ahead on 2 of 40 shapes, by up to 8%), at M = 2 2.28x / 2.19x: bf16/fp16 keep WMMA at every M.

### Tile rules, fp32

Sweep, table tile vs best, total (`analysis/rules_vs_best_fp32.txt`; every shape forced to WMMA; with only the
shapes the cutoffs send to WMMA the first row reads 1.047 for gfx1100):

| M (swept) | gfx1100 (base) | gfx1101 | gfx1102 | gfx1150 | gfx1151 | final |
|---|---|---|---|---|---|---|
| 1-8 (3, 4, 6, 8) | 1.059 | 1.075 | 1.059 | 1.008 | 1.008 | 1.000 |
| 9-16 (12, 16) | 1.170 | 1.053 | 1.005 | 1.001 | 1.103 | 1.000 |
| 17-24 (24) | 1.076 | 1.076 | 1.026 | 1.001 | 1.026 | 1.000 |
| 25-48 (32, 48) | 1.052 | 1.052 | 1.052 | 1.001 | 1.109 | 1.000 |
| 64, 128, 256 (every tile) | 1.079-1.081 | 1.079-1.081 | 1.079-1.081 | 1.000-1.004 | 1.003-1.073 | 1.000 |

Final table (`wmma_rules_gfx1103_f32`): `{1, 2, 1, 2, 2, 2}` through M = 8, `{2, 2, 1, 4, 1, 2}` above. One tile per
bucket is the per-shape best almost everywhere (worst shape at most 1.18x off best: small GQA k/v at M = 12-16; at
M = 64-256 it is the best tile of every one of the 40 shapes). At M >= 49 it replaces the gfx1100 32-row tile
`{4, 1, 1, 4, 1, 1}`: with fp32's three A planes a 32-row tile only fits LDS with 64-K iterations (kch = 1), and
`sweep.py` (and so the gfx1100 fit) skips tiles under 32 rows at M >= 64; the 16-row tile with 128-K iterations is
7-8% faster at multiples of 32 and wastes fewer padded rows elsewhere (M = 72-80: 80 instead of 96 rows).
The gfx1150 fp32 table is within 0.1-0.8% of this in the sweeps (single shapes up to 1.26x off best with its
N < 3072 leaves); measured below.

Validation, in-process interleaved: default dispatch (F) against the base dispatch through overrides (A: gfx1103
cutoffs, gfx1100 tiles), total F / A (`analysis/final_interleaved.txt`), final build, 5 cycles: M <= 160
`pinned/altF/fp32_lo.csv`, M = 256-512 `pinned/r1/fp32_af.csv` (review round 1):

| M (measured) | F / A (cycles) | shapes faster | worst cell |
|---|---|---|---|
| 1-2 | 1.000 (0.997-1.004) | SIMT in both | 1.02 |
| 3-4 | 0.971 (0.969-0.974) | 22-24 of 40 (up to 1.19x; the others run SIMT in A and F) | 1.05 (small square at M = 4: SIMT in A and F) |
| 5-8 (5, 6, 8) | 0.946 (0.944-0.946) | 30-33 of 40 (up to 1.5x) | 1.01 |
| 9-16 (9, 12, 16) | 0.856 (0.855-0.856) | 36-39 (up to 1.7x) | 1.00 |
| 17-24 (17, 24) | 0.930 (0.929-0.930) | 40 | 0.97 |
| 25-48 (32, 33, 48) | 0.952 (0.951-0.952) | 29-32 | 1.01 |
| 49-160 (49, 64, 72, 80, 96, 112, 128, 160) | 0.873 (0.873) | 40 | 0.98 |
| 256, 384, 416, 448, 480, 512 | 0.926 (0.926) | 40 | 0.98 |

Per M above 48: 0.924-0.926 at M = 49 and at the multiples of 32 (64, 96, 128, 160) and 0.926 at each of M = 256,
384, 416, 448, 480, 512 (40 of 40 shapes faster at each); 0.774 at M = 72-80 and 0.812 at M = 112 (fewer padded
rows). M <= 8 as one bucket (M = 3-8 measured): 0.956. The same comparison on the two earlier builds, whose fp32
tables, cutoffs and kernels are the same (fin2: `pinned/alt_s7/fp32_*.csv`, including a 3-cycle run at M = 256, 384,
512 that reads 0.926 at each; fin1: `analysis/earlier_candidates.txt`), agrees within 0.1% in every bucket, and the
same table on the base build with every shape forced to WMMA (`analysis/alt32.txt`, B against A, 5 cycles) gives
0.945 at M = 1-8, 0.856 at 9-16, 0.930 at 17-24, 0.952 at 32-48.
`fp32_x` (3 cycles): default dispatch = the candidate file through the override within 0.2%. `fp32_d` (3 cycles): the
gfx1150 fp32 table with the gfx1103 cutoffs is 0.7-0.8% slower in total at M = 5-8 and 0.0-0.4% at M = 16-128, with
single shapes up to 1.31x slower (1024 x 2048 at M = 5; 7-8b k/v 1.26x at M = 5-8) and a few up to 1.1x faster; the
own table is kept for that and for having fewer rules.

### Block order: unchanged

`analysis/block_order.txt` (base build, gfx1100 tiles, 3 cycles, M = 24, 48, 96, 192). fp32: row-tiles-first against
the gfx110x default (N-first) 0.993, 0.998, 0.998, 1.004 with the gfx1100 tiles and 0.998, 0.995, 0.996, 1.005 with
the 16-row tile: a tie, N-first kept. fp16: N-first against the default (row-tiles-first) 1.009, 1.166, 1.040, 1.042:
row-tiles-first kept.

### fp32 SIMT/WMMA cutoffs: unchanged (`{2, INT_MAX}, {4, 3840}`)

- Base tiles, forced SIMT and WMMA runs (`analysis/fp32_cutoffs_base_tiles.txt`): the gfx1103 table is within 0.9% of
  the per-shape oracle at every M but 2 (4.1%); the gfx1100 table is 1.24-1.45x off at M = 3-4 here, the gfx1151 table
  1.24x at M = 3.
- Final tiles, in-process (`analysis/alt32.txt`, S against B, 5 cycles): dispatch / oracle 1.000 at M = 1, 1.055 at
  M = 2, 1.004 at M = 3, 1.011 at M = 4, 1.000 from M = 5. The best single N cutoff per M would be N >= 4608 at M = 2
  (1.029), N >= 2048 at M = 3 (1.001), N >= 1024 at M = 4 (1.000): gains of 2.5%, 0.3% and 1.1%, under the ~3% bar,
  so the table stays.
- Single shapes stay well off the oracle under it (not regressions: the base build ran the same kernel there). With
  the new M <= 8 tile WMMA is now faster for some narrow layers that the N >= 3840 row keeps on SIMT: at M = 3
  2048 x 512 (1.54x), 2048 x 768 (1.31x), 2816 x 704 (1.30x), 2816 x 2112 (1.20x); at M = 4 2048 x 512 (1.75x),
  2816 x 704 (1.50x), 2816 x 2112 (1.47x), 2048 x 768 (1.46x), 2560 x 10240 (1.26x); at M = 2 (SIMT for all)
  2048 x 512 (1.32x) and several large layers 1.08-1.16x. An `{4, 1024}` row would recover the M = 4 cases with no
  shape more than 2% slower; left out as a sub-3% change in total.

### Correctness

`pytest tests/test_functional.py tests/test_linear4bit.py -k "(4bit or Linear4bit or gemv) and not fsdp and not
test_4bit_quant_large"`, release builds: base 2427 passed, 0 failed (`logs/pytest_rel_base.log`); final 2427 passed,
0 failed (`logs/pytest_rel_final3.log`; the two earlier candidate builds the same, `pytest_rel_final{1,2}.log`).
`test_4bit_quant_large` (18 cases, ~10 GiB for fp32) does not fit the 7.6 GB this iGPU exposes: in the first base run
(`logs/pytest_rel_base_aborted_quantlarge.log`, stopped by hand) it ran out of memory, the driver reset the GPU twice
(MES `REMOVE_QUEUE` timeouts) and the tests after it errored; it is excluded from the runs above, so the counts are
18 below the 2445 of the other machines.
`scripts/tilecheck1103.py` (release builds, default dispatch against a float64 reference; 11 (N, K) pairs covering
every leaf of the new tables, K % 128 != 0 included; 23 M values 1-512; blocksize 32 / 64 / 128; nested and not; three
dtypes; asserts finite outputs, the errors are printed): final build max relative error 3.9e-4 (fp16), 2.9e-3 (bf16),
8.8e-7 (fp32); base build 3.9e-4, 2.9e-3, 2.1e-6 (`logs/tilecheck_rel_*.txt`).
Kernel names (profiler, `logs/kname1103_*.txt`): see the final section.

### Python M limits: 512 / 512 (were 384 / 384)

Fused (default dispatch) against dequant + `F.linear`, in-process interleaved, all 40 shapes (`analysis/cap.txt`):
total fused / dequant, its range over the cycles, and the shapes on which the fused kernel is ahead. The bf16/fp16
tiles at these M are the gfx1100 bucket in every build.

| M | fp16 | bf16 | fp32 | fp16, blocksize 32 | bf16, blocksize 32 |
|---|---|---|---|---|---|
| 384 | 0.582 (0.581-0.584), 34 | 0.568 (0.567-0.568), 35 | 0.342 (0.342-0.343), 40 | 0.634 (0.631-0.637), 32 | 0.611 (0.607-0.613), 31 |
| 416 | 0.551 (0.551-0.553), 37 | 0.553 (0.553-0.554), 37 | 0.344 (0.344-0.345), 40 | 0.599 (0.598-0.600), 34 | 0.596 (0.595-0.598), 35 |
| 448 | 0.567 (0.566-0.567), 35 | 0.569 (0.568-0.571), 37 | 0.345 (0.345-0.345), 40 | 0.615 (0.612-0.617), 32 | 0.612 (0.610-0.613), 33 |
| 480 | 0.584 (0.583-0.585), 35 | 0.593 (0.591-0.594), 37 | 0.346 (0.345-0.346), 40 | 0.630 (0.626-0.632), 32 | 0.635 (0.632-0.636), 30 |
| 512 | 0.586 (0.584-0.586), 33 | 0.568 (0.568-0.569), 35 | 0.346 (0.346-0.346), 40 | 0.636 (0.634-0.637), 29 | 0.612 (0.609-0.614), 27 |

All columns 5 cycles: blocksize 64 bf16/fp16 from `pinned/cap/` (build fin1), blocksize 32 from
`pinned/r1/cap_bs32_*.csv` (build fin2), fp32 from the final build: `pinned/r1/fp32_fq.csv`
(all 40 shapes in one job, `--syncwarm`). It is the repeat of a run that a GPU reset ended after 39 of the 40 shapes; on
those 39 shapes the totals of the two runs agree within 0.4% for the fused kernel and for dequant (single cells'
ratios within 5%), and llama3-70b down, measured a third time in `fp32_fqb.csv`, differs from the first run by up to
0.6% in fused time, 0.8% in dequant time and 1.1% in the ratio (`analysis/cap.txt`). At M = 256
(`pinned/r1/fp32_fq256.csv`, the same protocol) fp32 reads 0.338 (0.338-0.339), ahead on 40 of 40 shapes.
Earlier 3-cycle runs agree: fp32 0.342 / 0.346 at M = 384 / 512 on 40 shapes (build fin1), blocksize 32 0.636 /
0.638 (fp16) and 0.612 / 0.611 (bf16) at M = 384 / 512.
The dense single-pass runs of the base build read 0.56-0.59 at M = 416-512 for bf16/fp16 on 34-37 shapes.

- fp16: 512, by total time (1.71-1.81x faster at M = 416-512; blocksize 32 1.57-1.67x) and by the majority of the
  shapes (33-37 of 40; blocksize 32 29-34).
- bf16: 512, by total time (1.69-1.81x; blocksize 32 1.57-1.68x) and by the majority of the shapes (35-37 of 40;
  blocksize 32 27-35).
- fp32: 512, by total time (2.89-2.92x faster at M = 384-512) and on every shape.

512 is the largest M measured interleaved and the largest the ROCm dispatch allows (`_gemm_4bit_custom_max_m` in
`ops.py`), not a crossover: a single-pass look above it (`analysis/beyond_512.txt`, bench.py, `pinned/evalH`) gives
fused / dequant 0.57-0.62 (fp16) and 0.62-0.65 (bf16) at M = 576-1024 with the fused kernel ahead on 34-37 of the 40
shapes, and 0.35 for fp32 at M = 1024 on all 40. The decision is for this machine and this nightly's BLAS.

Single shapes that the higher limit makes slower, i.e. the fused kernel more than 1.2x slower than dequant at
M = 416-512 where the base build ran dequant (`analysis/cap_single_shape_losses.txt`, medians over the 5 cycles):

- blocksize 64, fp16: gemma4-26b-moe expert down (2816 x 704) 1.21 / 1.30 / 1.37 / 1.52x at M = 416 / 448 / 480 /
  512, qwen3-30b-moe expert down (2048 x 768) 1.23x and gemma4-26b dense down (2816 x 2112) 1.21x at M = 512;
  bf16: gemma4-26b-moe expert down 1.25 / 1.32 / 1.48x at M = 448 / 480 / 512.
- blocksize 32, fp16: gemma4-26b-moe expert down 1.20 / 1.29 / 1.35 / 1.52x, and at M = 512 qwen3.5-35b-moe expert
  gate_up (1024 x 2048) 1.27x, gemma4-26b dense down 1.24x, qwen3-30b-moe expert down 1.23x (the 3-cycle run also
  had gemma4-26b-moe expert gate_up 1.24x and gemma4-26b dense gate_up 1.21x above the line; 1.2x or just under in
  the 5-cycle run); bf16: gemma4-26b-moe expert down 1.25 / 1.31 / 1.48x at M = 448 / 480 / 512, qwen3-30b-moe
  expert down 1.23 / 1.31 / 1.39x, and at M = 512 qwen3.5-35b-moe expert down (2048 x 512) 1.27x and
  qwen3.5-35b-moe expert gate_up 1.25x.

These are small MoE / dense projections of three models (gate_up layers included). An ideal per-shape fallback to
dequant for every shape behind it would recover 0.02-0.17% of the fused total at M = 416-512 at blocksize 64 and
0.04-0.35% at blocksize 32 (0.303% fp16 / 0.344% bf16 at M = 512), far under the ~3% bar, so no further dispatch
rule is added and the single limit stays. The same shapes are already behind at the old limit (M = 384: up to
1.43x).

### Final dispatch against the base build (dense single-pass evals)

`analysis/dispatch_base_final.txt` from `pinned/eval_base.csv` and `pinned/eval_final.csv` (bench.py, one pass per
M; indicative, the interleaved runs above are the evidence). Default dispatch including the Python limit against the
per-shape oracle over simt, wmma, dequant and auto, total:

- bf16/fp16: base within 0.3% at M <= 384, then 1.70-1.80x off at M = 416-512 (dequant above the 384 limit); final
  within 1.6% (fp16) / 1.2% (bf16) at every M (the forced-WMMA run of the same build is in the oracle, so part of
  that is run-to-run noise). Worst single shape: up to 1.55x (fp16) / 1.46x (bf16) at M = 384-512 in the final
  build and 1.42x / 1.36x at M = 384 in the base build (the small layers where dequant is ahead; single pass).
- fp32: base within 1.1% at M >= 3 and 4.4% at M = 2, except M = 416-512 (2.65-2.67x, limit); final within 0.3% at
  M >= 5, 1.2% at M = 4, 0.4% at M = 3, 5.7% at M = 2 (cutoffs, see above; worst shape 1.75x at M = 4).

Default dispatch against upstream main (SIMT through M = 8, dequant + `F.linear` above), total main / new:

| M | 1 | 2 | 4 | 8 | 9 | 16 | 32 | 64 | 128 | 256 | 384 | 512 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16, base | 1.19 | 1.83 | 2.92 | 4.45 | 7.44 | 7.38 | 5.33 | 3.37 | 2.16 | 1.62 | 1.69 | 1.00 |
| fp16, final | 1.48 | 2.28 | 3.61 | 5.43 | 8.67 | 8.53 | 5.87 | 3.66 | 2.15 | 1.62 | 1.70 | 1.70 |
| bf16, base | 1.14 | 1.75 | 2.80 | 4.28 | 8.26 | 8.21 | 5.92 | 3.73 | 2.15 | 1.68 | 1.76 | 1.00 |
| bf16, final | 1.41 | 2.17 | 3.47 | 5.24 | 9.66 | 9.48 | 6.55 | 4.04 | 2.14 | 1.67 | 1.76 | 1.75 |
| fp32, base | 1.00 | 1.00 | 1.46 | 2.22 | 6.48 | 6.47 | 3.72 | 3.10 | 2.85 | 2.71 | 2.68 | 1.00 |
| fp32, final | 1.00 | 1.00 | 1.50 | 2.35 | 7.57 | 7.55 | 3.93 | 3.36 | 3.10 | 2.94 | 2.91 | 2.88 |

Kernel names, final release build (`logs/kname1103_relfin3.txt`, profiler; base in `kname1103_rel.txt`): bf16/fp16
`gemm_4bit_wmma<11, T, 1, 2, 1, 1, 4, 2, ...>` at M = 1-8 for K = 4096 and `<..., 1, 2, 1, 1, 8, 2, ...>` for
K = 8192, `<..., 2, 1, 1, 4, 2, 2, ...>` at M = 16, `<..., 4, 1, 1, 4, 2, 2, ...>` at M = 24,
`<..., 2, 2, 1, 4, 1, 2, ...>` at M = 40 (`<..., 4, 2, 1, 4, 1, 2, ...>` for K = 14336),
`<..., 4, 2, 1, 4, 1, 2, ...>` at M = 64, the gfx1100 tiles above (`<..., 4, 2, 1, 4, 2, 2, ...>` at M = 96,
`<..., 4, 2, 1, 4, 1, 2, ...>` at M = 128 and 512 for 4096 x 4096), prescaled for fp16 from M = 96; fp32
`gemm_4bit_simt` at M = 1-2,
`gemm_4bit_wmma<11, float, 1, 2, 1, 2, 2, 2, ...>` at M = 3-8 and `<..., 2, 2, 1, 4, 1, 2, ...>` from M = 16.

### Plots

`plots/gfx1103_apu014/kernels_median.png` and `kernels_per_shape_{fp16,bf16,fp32}.png`: `plots_kernels.py` on
`pinned/eval_final.csv` with `PLOT_PEAK_GBS=120` (manifest next to them).

## Open items

- The other 760M's evidence is 5 shapes, fp16, one pass; what the final table does for it at M <= 32 cannot be
  validated any more, and there is no fp32 tile data from it at all.
- Tiles were swept at blocksize 64 with plain absmax only; blocksize 32 / 128 and double-quantized absmax were
  checked for the final table and a few candidates (interleaved), not swept. `{4, 2, 1, 4, 2, 2}` at blocksize 32 is
  the one tile found to behave differently (cause not looked into).
- fp32 cutoffs: the `{4, 1024}` row (five narrow layers 1.26-1.75x faster at M = 4) is left out as a sub-3% change.
- The GPU resets are unexplained beyond the driver messages. The repeatable case (the first dequant + `F.linear`
  call of an fp32 process: five resets in the six such starts before `--syncwarm` was extended, none in the three
  after) was not isolated: whether it needs
  the bitsandbytes dequantize kernel, a particular BLAS kernel or only a busy queue during a userptr invalidation is
  not known. It did not occur in a bf16/fp16 process.
- At blocksize 32 the M = 9-16 tile gains 5% in total (14% at blocksize 64) and five shapes are 2-10% slower there;
  a blocksize-dependent rule was not tried.
- fp32 at M = 256-512 was measured in separate runs (fused against dequant, base against final tiles) instead of
  the one three-way run the first review asked for.
- The blocksize 32 / 128 and double-quantized checks of the final build cover 4-9 M values (5 cycles), not the 31 of
  the main validation, and blocksize 128 was checked for fp16 only.
- `plots/speedup_vs_main.*` still shows the first machine (`data/gfx1103`, gfx1100 tiles); only the limits mirror in
  `plots_vs_main.py` was updated (the table does not change: that data ends at M = 256).
