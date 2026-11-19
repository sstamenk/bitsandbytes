"""aba_an.py dir dtype: per M totals over the 40 shapes: B/A (mean of B1,B2 over mean of A1,A2), A2/A1, B2/B1, worst
shape B/A (B = mean of B runs per shape), shapes slower > 10%."""

from collections import defaultdict
import csv
import sys

d, dt = sys.argv[1], sys.argv[2]
EXCL = set(sys.argv[3:])  # shapes left out at M <= 2 only (first-in-process artifact: first shape of each worker)
t = defaultdict(dict)
for run in ("a1", "a2", "b1", "b2"):
    for r in csv.DictReader(open(f"{d}/{dt}_{run}.csv")):
        if not (r["shape"] in EXCL and int(r["M"]) <= 2):
            t[(int(r["M"]), r["shape"])][run] = float(r["us"])
print(f"== {dt}: M  B/A  (A2/A1  B2/B1)  worst-shape B/A  best-shape")
for M in sorted({m for m, _ in t}):
    ks = [k for k in t if k[0] == M]
    S = {r: sum(t[k][r] for k in ks) for r in ("a1", "a2", "b1", "b2")}
    b_over_a = (S["b1"] + S["b2"]) / (S["a1"] + S["a2"])
    per = sorted(((t[k]["b1"] + t[k]["b2"]) / (t[k]["a1"] + t[k]["a2"]), k[1]) for k in ks)
    print(
        f"{M:4} {b_over_a:6.3f} ({S['a2'] / S['a1']:.3f} {S['b2'] / S['b1']:.3f})  {per[-1][0]:.2f} {per[-1][1][:24]:24} {per[0][0]:.2f} {per[0][1][:24]}"
    )
