"""bestin.py sweep.csv Ms Nlo Nhi [cfg ...]: tiles ranked by total time over shapes with Nlo <= N < Nhi at the given Ms; with cfgs, per-shape times of those."""

from collections import defaultdict
import csv
import sys

ms = set(map(int, sys.argv[2].split(",")))
lo, hi = int(sys.argv[3]), int(sys.argv[4])
t = defaultdict(dict)
for r in csv.DictReader(open(sys.argv[1])):
    if int(r["M"]) in ms and lo <= int(r["N"]) < hi:
        t[(int(r["M"]), r["shape"], int(r["N"]), int(r["K"]))][r["cfg"]] = float(r["us"])


def kfix(c, K):
    p = c.split(",")
    if K % 128:
        p[5] = "1"
    return ",".join(p)


best = sum(min(v.values()) for v in t.values())
cfgs = set.intersection(*[{c for c in v} | {c[:-1] + "2" for c in v if c.endswith("1")} for v in t.values()])
tot = sorted((sum(v.get(kfix(c, k[3]), 1e9) for k, v in t.items()), c) for c in cfgs)
print(f"{len(t)} points, best-per-shape total {best:.0f}")
for s, c in tot[:8]:
    print(f"  {c:14} {s / best:.3f}")
for c in sys.argv[5:]:
    print(c, " ".join(f"{k[1][:14]}@{k[0]}:{v[kfix(c, k[3])] / min(v.values()):.2f}" for k, v in sorted(t.items())))
