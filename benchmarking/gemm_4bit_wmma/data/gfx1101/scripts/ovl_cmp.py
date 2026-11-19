"""ovl_cmp.py dtype: fused (eval auto) vs dequant+F.linear with stock hipBLASLt (eval) and with the #12465 overlay (data/ovl)."""

from collections import defaultdict
import csv
import sys

dt = sys.argv[1]
s = defaultdict(dict)
for f, tag in [
    (f"data/eval/{dt}_dequant.csv", "stock"),
    (f"data/ovl/{dt}_dequant.csv", "ovl"),
    (f"data/eval/{dt}_auto.csv", "auto"),
]:
    for r in csv.DictReader(open(f)):
        s[(int(r["M"]), r["shape"])][tag] = float(r["us"])
print(f"{dt}: totals over 40 shapes (ms)")
for M in sorted({k[0] for k in s}):
    rows = [v for k, v in s.items() if k[0] == M and "ovl" in v]
    if not rows:
        continue
    st = sum(v["stock"] for v in rows)
    o = sum(v["ovl"] for v in rows)
    a = sum(v["auto"] for v in rows)
    print(
        f"M={M:4} stock {st / 1e3:7.1f} ovl {o / 1e3:7.1f} ovl/stock {o / st:5.3f} fused {a / 1e3:6.1f} fused/ovl {a / o:.3f} wins {sum(v['auto'] < v['ovl'] for v in rows)}/40"
    )
