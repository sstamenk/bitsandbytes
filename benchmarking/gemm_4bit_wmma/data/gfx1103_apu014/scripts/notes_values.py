# ruff: noqa
# Values of the NOTES.md placeholders (fill_notes.py). Snapshot for review round 2: the repeat of fp32_fq and the
# fp32 runs of build fin3 at M <= 160 are still running.
ALTF_FP32 = (
    "The fp32 runs at M <= 160 are those of build fin2 in `pinned/alt_s7/` (the same fp32 tables, cutoffs and "
    "kernels); fp32 at M = 256-512 on the final build is in `pinned/r1/`."
)

EVALF_FP32 = "the fp32 rows are from build fin2, which has the same fp32 tables, cutoffs and kernels"
FP32CAP_384 = "0.342 (0.342-0.343), 40"
FP32CAP_416 = "0.344 (0.344-0.345), 40"
FP32CAP_448 = "0.345 (0.345-0.345), 40"
FP32CAP_480 = "0.346 (0.345-0.346), 40"
FP32CAP_512 = "0.346 (0.346-0.346), 40"
FP32CAP_SRC = (
    "`pinned/r1/fp32_fq.csv`\n(all 40 shapes in one job, `--syncwarm`). It is the repeat of a run that a GPU reset "
    "ended after 39 of the 40 shapes; on\nthose 39 shapes the totals of the two runs agree within 0.4% for the fused "
    "kernel and for dequant (single cells'\nratios within 5%), and llama3-70b down, measured a third time in "
    "`fp32_fqb.csv`, differs from the first run by up to\n0.6% in fused time, 0.8% in dequant time and 1.1% in the "
    "ratio (`analysis/cap.txt`). At M = 256\n(`pinned/r1/fp32_fq256.csv`, the same protocol) fp32 reads 0.338 "
    "(0.338-0.339), ahead on 40 of 40 shapes."
)
FP32CAP_TEXT = "2.89-2.92x faster at M = 384-512"
FQ_CONSEQ = (
    "`r2_fp32_fq` was ended by the 10:52 reset after 39 of its 40 shapes. Its rows (all written before the reset,\n"
    "    clock pinned in every sample) are kept as `pinned/r1/fp32_fq_part39_before_gpu_reset.csv` and used only as "
    "a\n    cross-check, as is `fp32_fqb.csv` (the missing shape and llama3-70b down, a separate job); the complete "
    "run was\n    repeated in a new job without a reset (`fp32_fq.csv`), and that run alone is the evidence."
)
RESET_COUNT = "thirteen"
RESET_LATER = "No reset in the 3.4 h of runs after\n  that (fp32 with dequant three times, fp32 tile runs, bf16/fp16)."
CLOCKS_TEXT = (
    "In every 10-s sample with the GPU busy the sclk level and `freq1_input` read 2599 MHz\n  and mclk 937 MHz, "
    "except in the rest of job `altF` after its GPU reset and in the first attempt of `evalF3` (both\n  below) "
    "(`logs/*.clk`, `analysis/clocks.txt`); edge temperature 56-66 C median per timing job, at most 73 C; package\n"
    "  power 35-47 W median per timing job, at most 58.9 W."
)
FP32LO_SRC = (
    "build fin2 (`pinned/alt_s7/`;\nthe fp32 tables, cutoffs and kernels are those of the final build, only the "
    "bf16/fp16 M = 65-96 row differs)"
)
HOOKS_TEXT = (
    "The repository's pre-commit hooks were applied to this directory: CSV line endings are LF, trailing "
    "whitespace is removed, and the typos hook rewrote two tokens in logs: in a profiler warning line of "
    "`logs/kname1103_rel.txt` and `logs/kname3.log` the prefix of the ROCr function name "
    "`..._amd_profiling_get_dispatch_time` (the letters h, s, a) reads `has`, and `.kfd` lines that the "
    "150-character cut ended one letter short of `/home` read `/home`. Nothing else differs from the files on the "
    "machine."
)
AUDIT_TEXT = (
    "Result: every CSV\n    used as evidence was written by a job whose level read back `high` at its start, with "
    "every busy sample up to the\n    file's last write at 2599 MHz and no reset in the job (the jobs before 09:17 "
    "UTC did not verify the sclk state at\n    the start; their samples show it). The flagged files are all in "
    "`pinned/alt_s1/` (job `altF`, build fin1, the\n    02:02 reset: `fp16`, `bf16`, `*_x`, `*_m1` were finished "
    "9-28 min before it and the later `altF_retry` job kept\n    them; they are the earlier candidate S1's runs and "
    "back no number in this file; `fp32_{hi,x,d}` and the\n    13-shape part as described above) plus "
    "`r1/fp32_fq_part39_before_gpu_reset.csv`. No retry after a reset reused a\n    CSV of the failed attempt: "
    "`r2_fp32_afq` and the probes wrote none, the first `evalF3` attempt's files were moved\n    away and measured "
    "again, and the `fp32_fq` repeat measured all 40 shapes again."
)
