"""hi_an.py dir: single-pass bench.py runs above the M limit (auto and dequant per dtype in <dir>/<dtype>_<kernel>.csv):
per dtype and M the total over the 40 shapes, fused / dequant, the shapes where the fused kernel is ahead and the
worst shape."""

from collections import defaultdict
import csv
import glob
import sys

t = defaultdict(dict)
for f in sorted(glob.glob(sys.argv[1] + "/*.csv")):
    if f.endswith(".raw.csv"):
        continue
    for r in csv.DictReader(open(f)):
        t[(r["dtype"], int(r["M"]), r["shape"])][r["kernel"]] = float(r["us"])
for dt, M in sorted({k[:2] for k in t}, key=lambda x: (["fp16", "bf16", "fp32"].index(x[0]), x[1])):
    vs = [(k[2], v) for k, v in t.items() if k[:2] == (dt, M) and "auto" in v and "dequant" in v]
    if not vs:
        continue
    a = sum(v["auto"] for _, v in vs)
    d = sum(v["dequant"] for _, v in vs)
    w = max((v["auto"] / v["dequant"], s) for s, v in vs)
    print(
        f"{dt} M={M:5} n={len(vs)} fused {a / 1e3:8.1f} ms dequant {d / 1e3:8.1f} ms fused/dequant {a / d:.3f} ahead on {sum(v['auto'] < v['dequant'] for _, v in vs)}/{len(vs)} worst {w[0]:.2f} {w[1]}"
    )
