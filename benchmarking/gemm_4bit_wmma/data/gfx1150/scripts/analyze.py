"""analyze.py [--limit fp16=512,bf16=512,fp32=512] [--auto-from file.csv] f1.csv f2.csv ...
Per dtype and M: auto vs per-shape oracle (simt/wmma/dequant), fused(auto) vs dequant, and new dispatch vs main
(main: SIMT for M<=8, dequant above; new: auto for M<=limit, dequant above)."""

import argparse
from collections import defaultdict
import csv
import statistics

p = argparse.ArgumentParser()
p.add_argument("files", nargs="+")
p.add_argument("--limit", default="fp16=512,bf16=512,fp32=512")
p.add_argument("--dtypes", default="fp16,bf16,fp32")
p.add_argument("--quiet-shapes", action="store_true")
a = p.parse_args()
lim = {k: int(v) for k, v in (x.split("=") for x in a.limit.split(","))}
t = defaultdict(dict)  # (dtype,M,shape) -> kernel -> us
for f in a.files:
    for r in csv.DictReader(open(f)):
        t[(r["dtype"], int(r["M"]), r["shape"])][r["kernel"]] = float(r["us"])
for dt in a.dtypes.split(","):
    Ms = sorted({k[1] for k in t if k[0] == dt})
    if not Ms:
        continue
    print(
        f"== {dt} (limit {lim[dt]}): M | auto/oracle total, worst(shape) | best-kernel-per-M/oracle | auto vs dequant total, #fused wins/n, worst | new vs main: total, median, worst"
    )
    for M in Ms:
        ks = [k for k in t if k[0] == dt and k[1] == M]
        rows = [t[k] for k in ks]
        ok = [k for k in ks if "auto" in t[k] and "dequant" in t[k]]
        if not ok:
            continue

        def orc(v):
            return min(v[x] for x in ("simt", "wmma", "dequant", "auto") if x in v)

        auto = sum(t[k]["auto"] for k in ok)
        oracle = sum(orc(t[k]) for k in ok)
        w = max(ok, key=lambda k: t[k]["auto"] / orc(t[k]))
        deq = sum(t[k]["dequant"] for k in ok)
        wins = sum(t[k]["auto"] < t[k]["dequant"] for k in ok)
        wd = max(ok, key=lambda k: t[k]["auto"] / t[k]["dequant"])
        singles = {
            x: sum(t[k][x] for k in ok) for x in ("simt", "wmma", "dequant", "auto") if all(x in t[k] for k in ok)
        }
        bk = min(singles, key=singles.get)
        s = f"{M:5d} | {auto / oracle:.3f} {t[w]['auto'] / orc(t[w]):.2f}({w[2][:22]}) | {bk:7s} {singles[bk] / oracle:.3f} | {deq / auto:.3f} {wins:2d}/{len(ok)} {t[wd]['auto'] / t[wd]['dequant']:.2f}"
        # vs main
        mains = [k for k in ok if (M > 8 or "simt" in t[k])]
        if len(mains) == len(ok):
            mn = lambda v: v["simt"] if M <= 8 else v["dequant"]
            nw = lambda v: v["auto"] if M <= lim[dt] else v["dequant"]
            sp = [mn(t[k]) / nw(t[k]) for k in ok]
            s += f" | {sum(mn(t[k]) for k in ok) / sum(nw(t[k]) for k in ok):.2f}x {statistics.median(sp):.2f}x {min(sp):.2f}x"
        print(s)
