# ruff: noqa
"""normjoint.py out.csv joint.csv: divide each (device tag, M)'s us by that device's per-shape-best total at M,
so --score total on the output weighs each device's relative loss equally."""

from collections import defaultdict
import csv
import sys

rows = list(csv.DictReader(open(sys.argv[2])))
best = defaultdict(lambda: float("inf"))
for r in rows:
    k = (r["shape"], r["M"])
    best[k] = min(best[k], float(r["us"]))
tot = defaultdict(float)
for (sh, M), b in best.items():
    tot[(sh.split(":")[0], M)] += b
with open(sys.argv[1], "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["shape", "N", "K", "M", "cfg", "us"])
    w.writeheader()
    for r in rows:
        r = dict(r)
        r["us"] = float(r["us"]) / tot[(r["shape"].split(":")[0], r["M"])] * 1e6
        w.writerow(r)
