"""small_rows_fp16.py sweep_fp16c.csv sweep_fp16e.csv: fp16, M = 80 (5 row tiles of 16 or 3 of 32): the best tile with
at least 32 rows (fp16c) against the best with at most 16 rows (fp16e), per shape and in total."""

from collections import defaultdict
import csv
import sys

a, b = defaultdict(dict), defaultdict(dict)
for r in csv.DictReader(open(sys.argv[1])):
    if r["M"] == "80":
        a[r["shape"]][r["cfg"]] = float(r["us"])
for r in csv.DictReader(open(sys.argv[2])):
    b[r["shape"]][r["cfg"]] = float(r["us"])
big = sum(min(v.values()) for v in a.values())
small = sum(min(b[s].values()) for s in a)
print(
    f"per-shape best: tiles >= 32 rows {big / 1e3:.1f} ms, tiles <= 16 rows {small / 1e3:.1f} ms, ratio {small / big:.3f}"
)
c = min((sum(b[s][c] for s in b) / big, c) for c in set.intersection(*(set(v) for v in b.values())))
print(
    f"one tile for every shape: 4,2,1,4,1,2 {sum(a[s]['4,2,1,4,1,2'] for s in a) / big:.3f}, best tile <= 16 rows {c[1]} {c[0]:.3f} (of the >= 32-row best)"
)
for x, s in sorted((min(b[s].values()) / min(a[s].values()), s) for s in a):
    print(f"  {x:.3f} {s}")
