# ruff: noqa: F821  (T and INF are provided by rules11.load, which execs this file)
# Candidate B32 (gfx1102 fp32): gfx11 fp32 table with the M = 9-16 and 17-24 buckets refitted from the gfx1102 sweep.
RULES = [
    (8, [("N", 3840, 0, T(1, 1, 1, 1, 4, 1)), ("N", 5120, 0, T(1, 1, 1, 2, 2, 2))], T(1, 2, 1, 2, 2, 2)),
    (16, [("N", 3840, 0, T(2, 2, 1, 2, 2, 1))], T(2, 2, 1, 4, 1, 2)),
    (24, [("N", 3072, 0, T(1, 1, 1, 2, 2, 1)), ("N", 8192, 0, T(1, 2, 1, 2, 2, 2))], T(2, 2, 1, 4, 1, 2)),
    (48, [("N", 2560, 0, T(1, 2, 1, 2, 2, 2)), ("N", 5376, 0, T(2, 2, 1, 2, 2, 1))], T(2, 2, 1, 4, 1, 2)),
    (INF, [], T(4, 1, 1, 4, 1, 1)),
]
