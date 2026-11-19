"""paired_summary.py paired.csv [...]: per dtype and M, sum over shapes of per-shape medians (over cycles) for the old
and new variant, old/new, geomean, the worst and best shape, shapes >5% / >10% slower, and the spread of the per-cycle
old/new total ratio (min-max over cycles, both orders)."""

from collections import defaultdict
import csv
import math
import statistics
import sys

t = defaultdict(lambda: defaultdict(list))  # (dtype, M, shape) -> variant -> [us]
cyc = defaultdict(lambda: defaultdict(float))  # (dtype, M, cycle) -> variant -> sum
for f in sys.argv[1:]:
    for r in csv.DictReader(open(f)):
        k = (r["dtype"], int(r["M"]), r["shape"])
        t[k][r["variant"]].append(float(r["us"]))
        cyc[(r["dtype"], int(r["M"]), int(r["cycle"]))][r["variant"]] += float(r["us"])
print("dtype    M  old/new  geomean  per-cycle old/new  >5%  >10%  worst new/old          best new/old")
for dt, M in sorted({k[:2] for k in t}):
    ks = [k for k in t if k[:2] == (dt, M)]
    med = {k: {v: statistics.median(t[k][v]) for v in ("old", "new")} for k in ks}
    old = sum(m["old"] for m in med.values())
    new = sum(m["new"] for m in med.values())
    r = {k: med[k]["new"] / med[k]["old"] for k in ks}
    g = math.exp(-sum(math.log(x) for x in r.values()) / len(r))
    cr = [c["old"] / c["new"] for (d, m, _), c in cyc.items() if (d, m) == (dt, M)]
    w = max(ks, key=lambda k: r[k])
    b = min(ks, key=lambda k: r[k])
    print(
        f"{dt} {M:4d}  {old / new:.3f}   {g:.3f}    {min(cr):.3f}-{max(cr):.3f}     {sum(x > 1.05 for x in r.values()):3d} "
        f"{sum(x > 1.10 for x in r.values()):4d}  {r[w]:.2f} {w[2][:16]:16s}  {r[b]:.2f} {b[2][:16]}"
    )
