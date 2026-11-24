# ruff: noqa: F821  (T and INF are provided by rules1103.load, which execs this file)
# gfx1103 fp32 candidate F1: one tile through M = 8 and one above.
RULES = [
    (8, [], T(1, 2, 1, 2, 2, 2)),
    (INF, [], T(2, 2, 1, 4, 1, 2)),
]
