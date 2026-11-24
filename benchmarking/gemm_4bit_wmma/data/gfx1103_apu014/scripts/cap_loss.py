"""cap_loss.py cap.csv...: fused (F) vs dequant + F.linear (Q) per file and M: every shape where the fused kernel is
more than 1.2x slower than dequant (median over the cycles), and what a per-shape fallback to dequant for every shape
behind it would recover (in % of the fused total)."""

from collections import defaultdict
import csv
import os
import statistics as st
import sys

for f in sys.argv[1:]:
    t = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(f)):
        t[(r["dtype"], int(r["blocksize"]), int(r["M"]), r["shape"])][r["cfg"]].append(float(r["us"]))
    for dt, bs, M in sorted({k[:3] for k in t}):
        ks = [k for k in t if k[:3] == (dt, bs, M)]
        F = {k: st.median(t[k]["F"]) for k in ks}
        Q = {k: st.median(t[k]["Q"]) for k in ks}
        slow = sorted(((F[k] / Q[k], k[3]) for k in ks if F[k] > 1.2 * Q[k]), reverse=True)
        rec = sum(F[k] - Q[k] for k in ks if F[k] > Q[k]) / sum(F.values())
        print(
            f"{os.path.basename(f)} {dt} bs{bs} M={M}: {sum(F[k] > Q[k] for k in ks)} shapes behind dequant, per-shape fallback would recover {100 * rec:.3f}% of the fused total; > 1.2x: "
            + (", ".join(f"{s} {x:.2f}" for x, s in slow) or "none")
        )
