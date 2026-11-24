# ruff: noqa: F821  (T and INF are provided by rules1103.load, which execs this file)
# gfx1103 fp32 candidate B (fit on apu014 sweep_fp32a): one tile per bucket.
RULES = [
    (8, [], T(1, 2, 1, 2, 2, 2)),
    (48, [], T(2, 2, 1, 4, 1, 2)),
    (INF, [], T(4, 1, 1, 4, 1, 1)),
]
