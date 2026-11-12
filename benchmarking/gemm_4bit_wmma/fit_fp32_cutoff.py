"""Fit the fp32 SIMT/WMMA choice by N from a bench CSV with both kernels.

For each M, choose min_n (WMMA iff N >= min_n; inf = always SIMT, 0 = always WMMA) minimising
total time over the shapes. Prints per M: total time of SIMT only, WMMA only, the fitted rule and
the per-shape oracle, plus the worst shape slowdown vs the faster kernel.

usage: python fit_fp32_cutoff.py bench.csv [dtype]
"""

from collections import defaultdict
import csv
import sys

dtype = sys.argv[2] if len(sys.argv) > 2 else "fp32"
t = defaultdict(dict)  # M -> shape -> {kernel: us, "N": N}
for r in csv.DictReader(open(sys.argv[1])):
    if r["dtype"] != dtype or r["kernel"] not in ("simt", "wmma"):
        continue
    d = t[int(r["M"])].setdefault(r["shape"], {"N": int(r["N"])})
    d[r["kernel"]] = float(r["us"])

print(f"{'M':>4} {'simt':>9} {'wmma':>9} {'rule':>9} {'oracle':>9}  min_n   rule/oracle  worst")
for M in sorted(t):
    sh = [v for v in t[M].values() if "simt" in v and "wmma" in v]
    if not sh:
        continue
    simt = sum(v["simt"] for v in sh)
    wmma = sum(v["wmma"] for v in sh)
    oracle = sum(min(v["simt"], v["wmma"]) for v in sh)
    cands = sorted({v["N"] for v in sh}) + [1 << 30]
    best = None
    for n in [0] + cands:
        tot = sum(v["wmma"] if v["N"] >= n else v["simt"] for v in sh)
        if best is None or tot < best[0]:
            best = (tot, n)
    tot, n = best
    worst = max((v["wmma"] if v["N"] >= n else v["simt"]) / min(v["simt"], v["wmma"]) for v in sh)
    ns = "inf" if n >= 1 << 30 else str(n)
    print(f"{M:4} {simt:9.0f} {wmma:9.0f} {tot:9.0f} {oracle:9.0f}  {ns:>6}  {tot / oracle:10.3f}  {worst:5.2f}")
