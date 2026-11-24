"""cap_an.py cap.csv...: fused (F) vs dequant + F.linear (Q), in-process interleaved: per dtype, blocksize and M the total
over the 40 shapes of the per-shape median over cycles, F/Q, its range over the cycles, the shapes where F is faster,
and the shapes where F is slower than Q by more than 20% (worst listed)."""

from collections import defaultdict
import csv
import statistics as st
import sys

for f in sys.argv[1:]:
    t = defaultdict(lambda: defaultdict(dict))
    for r in csv.DictReader(open(f)):
        t[(r["dtype"], int(r["blocksize"]), int(r["M"]), r["shape"])][r["cfg"]][int(r["cycle"])] = float(r["us"])
    med = lambda k, c: st.median(t[k][c].values())
    for dt, bs, M in sorted({k[:3] for k in t}):
        ks = [k for k in t if k[:3] == (dt, bs, M)]
        F, Q = sum(med(k, "F") for k in ks), sum(med(k, "Q") for k in ks)
        cyc = sorted(set.intersection(*(set(t[k]["F"]) & set(t[k]["Q"]) for k in ks)))
        rr = [sum(t[k]["F"][y] for k in ks) / sum(t[k]["Q"][y] for k in ks) for y in cyc]
        per = sorted(((med(k, "F") / med(k, "Q"), k[3][:28]) for k in ks), reverse=True)
        slow = [p for p in per if p[0] > 1.2]
        print(
            f"{dt} bs{bs} M={M:4} n={len(ks)} fused {F / 1e3:8.1f} ms dequant {Q / 1e3:8.1f} ms F/Q {F / Q:.3f} (cycles {min(rr):.3f}-{max(rr):.3f}, {len(cyc)}) "
            f"fused ahead on {sum(p[0] < 1 for p in per)}/{len(ks)}; > 1.2x slower: {len(slow)} "
            + ", ".join(f"{p[1]} {p[0]:.2f}" for p in slow[:6])
        )
