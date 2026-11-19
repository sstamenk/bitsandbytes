# ruff: noqa
"""sweep_cmp.py ref_sweep.csv new_sweep.csv: same (shape,M,cfg) in both. Per M: median new/ref of identical tiles,
spread; how far ref's best tile is from new's best on new; rank correlation."""

from collections import defaultdict
import csv
import statistics as st
import sys


def L(f):
    t = defaultdict(dict)
    for r in csv.DictReader(open(f)):
        t[(int(r["M"]), r["shape"])][r["cfg"]] = float(r["us"])
    return t


a, b = L(sys.argv[1]), L(sys.argv[2])
print("   M  n  med(new/ref) p10 p90 | refbest-on-new / newbest (total, worst) | newbest-on-ref / refbest (total)")
for M in sorted({k[0] for k in b}):
    ks = [k for k in b if k[0] == M and k in a]
    if not ks:
        continue
    rat = [b[k][c] / a[k][c] for k in ks for c in b[k] if c in a[k]]
    rat.sort()
    tn = tb = tr = trb = 0
    worst = 1
    for k in ks:
        com = [c for c in b[k] if c in a[k]]
        ca = min(com, key=lambda c: a[k][c])
        cb = min(com, key=lambda c: b[k][c])
        tn += b[k][ca]
        tb += b[k][cb]
        tr += a[k][cb]
        trb += a[k][ca]
        worst = max(worst, b[k][ca] / b[k][cb])
    print(
        f"{M:4} {len(ks):2}  {st.median(rat):6.3f} {rat[len(rat) // 10]:5.2f} {rat[9 * len(rat) // 10]:5.2f} | {tn / tb:6.3f} {worst:5.2f} | {tr / trb:6.3f}"
    )
