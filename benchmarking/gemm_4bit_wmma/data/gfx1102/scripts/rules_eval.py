"""rules_eval.py sweep.csv table [table...]: rule tile vs per-shape best swept tile, per M (total ratio, geomean, worst, missing)."""

from collections import defaultdict
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rules11 import load, pick

t = defaultdict(dict)
for r in csv.DictReader(open(sys.argv[1])):
    t[(int(r["M"]), r["shape"], int(r["N"]), int(r["K"]))][r["cfg"]] = float(r["us"])
tabs = [(n, load(n)) for n in sys.argv[2:]]
print("   M " + " ".join(f"| {n:>12} tot  geo worst miss" for n, _ in tabs))
for M in sorted({k[0] for k in t}):
    line = f"{M:4} "
    for n, R in tabs:
        tot = best = lg = 0
        worst = 1
        c_n = miss = 0
        for k, v in t.items():
            if k[0] != M:
                continue
            c = pick(R, M, k[2], k[3])
            if c not in v:
                miss += 1
                continue
            b = min(v.values())
            tot += v[c]
            best += b
            lg += math.log(v[c] / b)
            worst = max(worst, v[c] / b)
            c_n += 1
        if not c_n:
            line += "| n/a "
            continue
        line += f"| {tot / best:16.3f} {math.exp(lg / max(c_n, 1)):5.3f} {worst:5.2f} {miss:3} "
    print(line)
