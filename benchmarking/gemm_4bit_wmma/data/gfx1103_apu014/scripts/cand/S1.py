# ruff: noqa: F821  (T and INF are provided by rules1103.load, which execs this file)
# gfx1103 fp16/bf16 candidate S1: tiles that also hold on the single-channel 760M's 5-shape sweep.
RULES = [
    (8, [("K", 8192, 0, T(1, 2, 1, 1, 4, 2))], T(1, 2, 1, 1, 8, 2)),
    (16, [], T(2, 1, 1, 4, 2, 2)),
    (32, [("KN", 1, 2, T(4, 1, 1, 4, 2, 2))], T(4, 2, 1, 4, 2, 2)),
    (48, [("K", 12288, 0, T(2, 2, 1, 4, 1, 2))], T(4, 2, 1, 4, 1, 2)),
    (96, [], T(4, 2, 1, 4, 1, 2)),
    (INF, [("N", 2816, 0, T(4, 1, 1, 4, 2, 2)), ("N", 3072, 0, T(4, 2, 1, 4, 2, 1))], T(4, 2, 1, 4, 1, 2)),
]
