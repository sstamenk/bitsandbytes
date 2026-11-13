"""Compare A/B/A bench runs: total time over shapes per (dtype, M), B vs mean(A1, A2), with A1/A2 noise.

usage: python ab_compare.py a1.csv b.csv a2.csv [kernel]
"""

from collections import defaultdict
import csv
import sys

kern = sys.argv[4] if len(sys.argv) > 4 else None


def load(p):
    d = defaultdict(dict)
    for r in csv.DictReader(open(p)):
        if kern and r["kernel"] != kern:
            continue
        d[(r["dtype"], int(r["M"]))][r["shape"]] = float(r["us"])
    return d


a1, b, a2 = (load(p) for p in sys.argv[1:4])
print(f"{'dtype':5} {'M':>4} {'A us':>9} {'B us':>9} {'A/B':>6} {'a1/a2':>6} {'B wins':>6} {'worst B/A':>9} worst shape")
for k in sorted(b, key=lambda k: (k[0], k[1])):
    sh = [s for s in b[k] if s in a1.get(k, {}) and s in a2.get(k, {})]
    A = {s: (a1[k][s] + a2[k][s]) / 2 for s in sh}
    ta, tb = sum(A.values()), sum(b[k][s] for s in sh)
    noise = sum(a1[k][s] for s in sh) / sum(a2[k][s] for s in sh)
    wins = sum(b[k][s] < A[s] for s in sh)
    ws = max(sh, key=lambda s: b[k][s] / A[s])
    print(f"{k[0]:5} {k[1]:4} {ta:9.0f} {tb:9.0f} {ta / tb:6.3f} {noise:6.3f} {wins:6} {b[k][ws] / A[ws]:9.3f} {ws[:34]}")
