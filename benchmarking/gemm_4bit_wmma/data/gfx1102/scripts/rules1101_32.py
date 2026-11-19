# ruff: noqa: F821  (T and INF are provided by rules11.load, which execs this file)
# gfx1101 fp32 table (tune/gfx1101 commit 16b858b, csrc/gemm_4bit_wmma_rules.cuh:216), for the reuse check.
RULES = [
    (8, [("N", 4608, 0, T(1, 1, 1, 2, 2, 1)), ("N", 8192, 0, T(1, 2, 1, 2, 2, 2))], T(1, 2, 1, 2, 2, 1)),
    (16, [("N", 2816, 0, T(1, 2, 1, 2, 2, 2)), ("N", 8192, 0, T(2, 2, 1, 2, 2, 1))], T(2, 2, 1, 4, 1, 2)),
    (24, [("N", 3072, 0, T(1, 1, 1, 2, 4, 1))], T(1, 2, 1, 2, 2, 2)),
    (48, [("N", 2560, 0, T(1, 2, 1, 2, 2, 2)), ("N", 5376, 0, T(2, 2, 1, 2, 2, 1))], T(2, 2, 1, 4, 1, 2)),
    (INF, [], T(4, 1, 1, 4, 1, 1)),
]
