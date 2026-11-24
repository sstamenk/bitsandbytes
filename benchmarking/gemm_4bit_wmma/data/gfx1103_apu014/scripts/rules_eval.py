"""rules_eval.py sweep.csv table [table...]: rule tile vs per-shape best swept tile per M (total ratio, geomean,
worst, missing). Tables: names from rules1103.parse() (g1100_f16, ...) or .py files. Lookup with kch = KCH_MAX
(prescale-aware: sweep1103.py has kch = 2 candidates for K % 128 != 0). --shapes-filter fp32wmma restricts fp32 to
the shapes the gfx1103 cutoffs send to WMMA. --buckets e.g. 8,16,32 prints totals per M bucket (sum over swept M)."""

import argparse
from collections import defaultdict
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rules1103 import load, pick

p = argparse.ArgumentParser()
p.add_argument("sweep")
p.add_argument("tables", nargs="+")
p.add_argument("--fp32wmma", action="store_true")
p.add_argument("--buckets", default="")
a = p.parse_args()


def wmma1103(M, N):
    return M > 4 or (M > 2 and N >= 3840)


t = defaultdict(dict)
for r in csv.DictReader(open(a.sweep)):
    M, N = int(r["M"]), int(r["N"])
    if a.fp32wmma and not wmma1103(M, N):
        continue
    t[(M, r["shape"], N, int(r["K"]))][r["cfg"]] = float(r["us"])
tabs = [(os.path.basename(n).replace(".py", ""), load(n)) for n in a.tables]
print("   M " + " ".join(f"| {n[:14]:>14}  tot   geo worst miss" for n, _ in tabs))
bk = [int(x) for x in a.buckets.split(",")] if a.buckets else []
acc = defaultdict(lambda: [0.0, 0.0])
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
        line += f"| {tot / best:15.3f} {math.exp(lg / c_n):5.3f} {worst:5.2f} {miss:3} "
        if bk:
            b_ = next((x for x in bk if M <= x), 1 << 30)
            acc[(n, b_)][0] += tot
            acc[(n, b_)][1] += best
    print(line)
if bk:
    print("bucket totals (sum over swept M in bucket) / best:")
    for b_ in sorted({k[1] for k in acc}):
        print(
            f"  M<={b_ if b_ < 1 << 30 else 'inf':>4} "
            + " ".join(
                f"{n[:14]:>14} " + (f"{acc[(n, b_)][0] / acc[(n, b_)][1]:.3f}" if acc[(n, b_)][1] else "  n/a")
                for n, _ in tabs
            )
        )
