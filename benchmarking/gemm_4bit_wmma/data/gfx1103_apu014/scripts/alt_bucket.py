"""alt_bucket.py alt.csv ref cfg[,cfg...] buckets [--simt-below=N@M...]: per M bucket (comma list of upper M bounds, the
last open): sum over the bucket's measured M and the shapes of the per-shape median over cycles, cfg/ref; the range of
the same ratio per cycle; shapes where cfg is slower than ref by > 5% / > 20% (worst listed)."""

from collections import defaultdict
import csv
import statistics as st
import sys

f, ref, cfgs, bk = sys.argv[1], sys.argv[2], sys.argv[3].split(","), [int(x) for x in sys.argv[4].split(",")]
t = defaultdict(lambda: defaultdict(dict))
for r in csv.DictReader(open(f)):
    t[(int(r["M"]), r["shape"])][r["cfg"]][int(r["cycle"])] = float(r["us"])
med = lambda k, c: st.median(t[k][c].values())


def bucket(M):
    return next((b for b in bk if M <= b), 1 << 30)


for b in [*bk, 1 << 30]:
    ks = [k for k in t if bucket(k[0]) == b]
    if not ks:
        continue
    ms = sorted({k[0] for k in ks})
    R = sum(med(k, ref) for k in ks)
    line = f"M<={b if b < 1 << 30 else 'inf':>4} (M {','.join(map(str, ms))}; {len(ks)} cells) {ref}={R / 1e3:9.1f} ms"
    for c in cfgs:
        C = sum(med(k, c) for k in ks)
        cyc = sorted(set.intersection(*(set(t[k][c]) & set(t[k][ref]) for k in ks)))
        rr = [sum(t[k][c][y] for k in ks) / sum(t[k][ref][y] for k in ks) for y in cyc]
        per = sorted(((med(k, c) / med(k, ref), k[1][:26], k[0]) for k in ks), reverse=True)
        n5 = sum(p[0] > 1.05 for p in per)
        n20 = sum(p[0] > 1.2 for p in per)
        line += f" | {c}/{ref} {C / R:.4f} (cycles {min(rr):.4f}-{max(rr):.4f}) slower>5%: {n5}, >20%: {n20}, worst {per[0][0]:.2f} {per[0][1]} M={per[0][2]}, best {per[-1][0]:.2f}"
    print(line)
