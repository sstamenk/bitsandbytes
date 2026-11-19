"""alt_an.py alt.csv [ref=A]: per (dtype, M): total over shapes of the per-shape median over cycles for each cfg,
ratio cfg/ref, per-cycle total-ratio range (cycles common to all shapes only), worst shape ratio, and the watched
shapes 22/24/36."""

from collections import defaultdict
import csv
import statistics as st
import sys

ref = sys.argv[2] if len(sys.argv) > 2 else "A"
t = defaultdict(lambda: defaultdict(list))
for r in csv.DictReader(open(sys.argv[1])):
    t[(r["dtype"], int(r["M"]), r["shape"])][r["cfg"]].append((int(r["cycle"]), float(r["us"])))
cfgs = sorted(
    {c for v in t.values() for c in v},
    key=lambda c: ["A", "B", "B2", "B3", "C32", "dequant"].index(c) if c in ["A", "B", "J2", "J", "dequant"] else 9,
)
WATCH = ["small GQA k/v", "7-8b k/v", "llama3-8b/mistral-7b gate/up", "gemma3-12b gate/up"]
for dt in sorted({k[0] for k in t}):
    print(
        f"== {dt}: cfg/{ref} total (cycle-total range) worst-shape | watched shapes cfg/{ref}: small GQA k/v, 7-8b k/v, llama3-8b gate/up, gemma3-12b gate/up"
    )
    for M in sorted({k[1] for k in t if k[0] == dt}):
        ks = [k for k in t if k[0] == dt and k[1] == M]
        med = lambda k, c: st.median(u for _, u in t[k][c])
        tot = {c: sum(med(k, c) for k in ks) for c in cfgs}
        line = f"{M:4} n={len(ks):2} {ref}={tot[ref]:9.0f}us"
        for c in cfgs:
            if c == ref:
                continue
            # only cycles every shape has (full-suite cycles), so each ratio sums the same workload
            cyc = sorted(set.intersection(*({cy for cy, _ in t[k][c]} & {cy for cy, _ in t[k][ref]} for k in ks)))
            rr = []
            for cy in cyc:
                a = sum(u for k in ks for cc, u in t[k][ref] if cc == cy)
                b = sum(u for k in ks for cc, u in t[k][c] if cc == cy)
                rr.append(b / a)
            w = max((med(k, c) / med(k, ref), k[2][:18]) for k in ks)
            watch = " ".join(f"{med(k, c) / med(k, ref):4.2f}" for nm in WATCH for k in ks if k[2] == nm)
            wins = sum(med(k, c) < med(k, ref) for k in ks)
            line += f" | {c} {tot[c] / tot[ref]:6.3f} ({min(rr):.3f}-{max(rr):.3f}) w{w[0]:4.2f} {w[1]:18} [{watch}] win{wins}"
        print(line)
