# ruff: noqa
"""cmp_units.py ref.csv new.csv [...]: per dtype/kernel/M, new/ref total time ratio and median/min/max per-shape ratio."""

import csv
import statistics as st
import sys


def load(fs):
    t = {}
    for f in fs:
        for r in csv.DictReader(open(f)):
            t[(r["dtype"], r["kernel"], int(r["M"]), r["shape"])] = float(r["us"])
    return t


ref = load([sys.argv[1]])
new = load(sys.argv[2:])
keys = sorted({k[:3] for k in new if k in ref}, key=lambda k: (["fp16", "bf16", "fp32"].index(k[0]), k[1], k[2]))
cur = None
for dt, kern, M in keys:
    if (dt, kern) != cur:
        cur = (dt, kern)
        print(f"== {dt} {kern}: M  total_ratio  median  min  max  (new/ref)")
    rs = [new[k] / ref[k] for k in new if k[:3] == (dt, kern, M) and k in ref]
    tn = sum(new[k] for k in new if k[:3] == (dt, kern, M) and k in ref)
    tr = sum(ref[k] for k in new if k[:3] == (dt, kern, M) and k in ref)
    print(f"{M:4} {tn / tr:6.3f} {st.median(rs):6.3f} {min(rs):6.3f} {max(rs):6.3f}  n={len(rs)}")
