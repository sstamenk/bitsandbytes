# ruff: noqa: F821  (T and INF come from the namespace the loader exec()s this file in)
# Candidate C32 (fp32): gfx1101-fitted tiles at M <= 16 (WMMA-dispatched shapes of the fp32 sweep), gfx11 tiles above.
RULES = [
    (8, [("N", 4608, 0, T(1, 1, 1, 2, 2, 1)), ("N", 8192, 0, T(1, 2, 1, 2, 2, 2))], T(1, 2, 1, 2, 2, 1)),
    (16, [("N", 2816, 0, T(1, 2, 1, 2, 2, 2)), ("N", 8192, 0, T(2, 2, 1, 2, 2, 1))], T(2, 2, 1, 4, 1, 2)),
    (24, [("N", 3072, 0, T(1, 1, 1, 2, 4, 1))], T(1, 2, 1, 2, 2, 2)),
    (48, [("N", 2560, 0, T(1, 2, 1, 2, 2, 2)), ("N", 5376, 0, T(2, 2, 1, 2, 2, 1))], T(2, 2, 1, 4, 1, 2)),
    (INF, [], T(4, 1, 1, 4, 1, 1)),
]
