# ruff: noqa: F821  (T and INF are provided by rules1103.load, which execs this file)
# fp32 probe: the 16-row tile {2, 2, 1, 4, 1, 2} at every M.
RULES = [
    (INF, [], T(2, 2, 1, 4, 1, 2)),
]
