"""cap_an.py cap.csv [fused=A]: fused vs dequant+F.linear per M from an interleaved alt.py --dequant run: total ratio of
per-shape medians, per-cycle ratio range, shapes where fused is faster, worst fused/dequant shape."""

from collections import defaultdict
import csv
import statistics as st
import sys

f = sys.argv[2] if len(sys.argv) > 2 else "A"
t = defaultdict(lambda: defaultdict(list))
for r in csv.DictReader(open(sys.argv[1])):
    t[(r["dtype"], int(r["M"]), r["shape"])][r["cfg"]].append((int(r["cycle"]), float(r["us"])))
for dt in sorted({k[0] for k in t}):
    print(f"== {dt}: fused/dequant total (per-cycle range)  fused-faster shapes  worst")
    for M in sorted({k[1] for k in t if k[0] == dt}):
        ks = [k for k in t if k[0] == dt and k[1] == M]
        md = lambda k, c: st.median(u for _, u in t[k][c])
        a = sum(md(k, f) for k in ks)
        d = sum(md(k, "dequant") for k in ks)
        cyc = sorted({c for k in ks for c, _ in t[k][f]})
        rr = [
            sum(u for k in ks for c, u in t[k][f] if c == cy)
            / sum(u for k in ks for c, u in t[k]["dequant"] if c == cy)
            for cy in cyc
        ]
        wins = sum(md(k, f) < md(k, "dequant") for k in ks)
        w = max((md(k, f) / md(k, "dequant"), k[2][:26]) for k in ks)
        print(
            f"{M:4} {a / d:6.3f} ({min(rr):.3f}-{max(rr):.3f})  {wins:2}/{len(ks)}  worst {w[0]:.2f} {w[1]}  fused {a / 1e3:.1f} ms dequant {d / 1e3:.1f} ms"
        )
