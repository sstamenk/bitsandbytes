"""alt2_an.py file.csv ref [cfg...]: per M: total over shapes of per-shape median over cycles, cfg/ref, per-cycle ratio
range, shapes where cfg beats ref, worst/best shape ratio; with --dispatch-simt-below N: shapes with N < value use
the measured `simt` config for every table (fp32 dispatch)."""

from collections import defaultdict
import csv
import statistics as st
import sys

args = [x for x in sys.argv[1:] if not x.startswith("--")]
simt_below = next((int(x.split("=")[1]) for x in sys.argv[1:] if x.startswith("--dispatch-simt-below=")), 0)
f, ref = args[0], args[1]
t = defaultdict(lambda: defaultdict(list))
for r in csv.DictReader(open(f)):
    t[(int(r["M"]), r["shape"], int(r["N"]))][r["cfg"]].append((int(r["cycle"]), float(r["us"])))
cfgs = args[2:] or sorted({c for v in t.values() for c in v if c != ref})


def val(k, c):
    if simt_below and k[2] < simt_below and "simt" in t[k] and c != "simt":
        c = "simt"
    return t[k][c]


for M in sorted({k[0] for k in t}):
    ks = [k for k in t if k[0] == M]
    med = lambda k, c: st.median(u for _, u in val(k, c))
    R = sum(med(k, ref) for k in ks)
    line = f"M={M:4} n={len(ks)} {ref}={R:9.0f}us"
    for c in cfgs:
        if c == ref:
            continue
        C = sum(med(k, c) for k in ks)
        cyc = sorted({cy for k in ks for cy, _ in val(k, c)})
        rr = [
            sum(u for k in ks for cy, u in val(k, c) if cy == y)
            / sum(u for k in ks for cy, u in val(k, ref) if cy == y)
            for y in cyc
        ]
        per = sorted((med(k, c) / med(k, ref), k[1][:22]) for k in ks)
        wins = sum(med(k, c) < med(k, ref) for k in ks)
        line += f" | {c}/{ref} {C / R:.4f} ({min(rr):.3f}-{max(rr):.3f}) {c}<{ref} on {wins} worst {per[-1][0]:.2f} {per[-1][1]} best {per[0][0]:.2f}"
    print(line)
