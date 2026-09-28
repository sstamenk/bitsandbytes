from collections import defaultdict
import csv
import statistics as st
import sys

sweep, base = sys.argv[1], sys.argv[2]
best = {}
for r in csv.DictReader(open(sweep)):
    k = (r["shape"], int(r["M"]))
    us = float(r["us"])
    if k not in best or us < best[k][0]:
        best[k] = (us, r["cfg"])
b = defaultdict(dict)
for r in csv.DictReader(open(base)):
    b[(r["shape"], int(r["M"]))][r["kernel"]] = float(r["us"])
Ms = sorted({k[1] for k in best})
shapes = list(dict.fromkeys(k[0] for k in best))
print(f"{'shape':32}" + "".join(f"{'M=' + str(m):>17}" for m in Ms))
agg = defaultdict(lambda: defaultdict(list))
for s in shapes:
    line = f"{s[:32]:32}"
    for m in Ms:
        if (s, m) not in best:
            line += " " * 17
            continue
        w = best[(s, m)][0]
        bb = b.get((s, m), {})
        vs = bb.get("simt", float("nan")) / w
        vd = bb.get("dequant", float("nan")) / w
        agg[m]["simt"].append(vs)
        agg[m]["deq"].append(vd)
        line += f"{w:7.0f} {vs:4.2f} {vd:4.2f}"
    print(line)
print(
    f"{'geomean (wmma speedup vs simt, deq)':32}"
    + "".join(f"{'':7} {st.geometric_mean(agg[m]['simt']):4.2f} {st.geometric_mean(agg[m]['deq']):4.2f}" for m in Ms)
)
