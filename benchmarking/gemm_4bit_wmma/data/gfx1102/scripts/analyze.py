"""analyze.py cap16 cap32 file.csv [...]: per dtype/M totals over shapes for auto, oracle(simt,wmma,dequant), main, new."""

from collections import defaultdict
import csv
import sys

cap = {"fp16": int(sys.argv[1]), "bf16": int(sys.argv[1]), "fp32": int(sys.argv[2])}
t = defaultdict(dict)
for f in sys.argv[3:]:
    for r in csv.DictReader(open(f)):
        t[(r["dtype"], int(r["M"]), r["shape"])][r["kernel"]] = float(r["us"])
dts = sorted({k[0] for k in t}, key=["fp16", "bf16", "fp32"].index)
for dt in dts:
    Ms = sorted({k[1] for k in t if k[0] == dt})
    print(f"== {dt} (cap {cap[dt]})")
    print("   M |  auto/orc worst(shape) | wmma/orc simt/orc | auto/deq wins(auto<deq) | new/orc  main/new | n")
    for M in Ms:
        rows = [v for k, v in t.items() if k[0] == dt and k[1] == M]
        rows = [v for v in rows if "auto" in v and "dequant" in v]
        if not rows:
            continue
        orc = lambda v: min(v[k] for k in ("simt", "wmma", "dequant", "auto") if k in v)
        so = sum(orc(v) for v in rows)
        sa = sum(v["auto"] for v in rows)
        sd = sum(v["dequant"] for v in rows)
        worst = max(v["auto"] / orc(v) for v in rows)
        sw = sum(v.get("wmma", float("nan")) for v in rows)
        ss = sum(v.get("simt", float("nan")) for v in rows)
        new = sum((v["auto"] if M <= cap[dt] else v["dequant"]) for v in rows)
        main = sum((v.get("simt", v["auto"]) if M <= 8 else v["dequant"]) for v in rows)
        wins = sum(v["auto"] < v["dequant"] for v in rows)
        print(
            f"{M:4} | {sa / so:6.3f} {worst:6.2f} | {sw / so:6.3f} {ss / so:6.3f} | {sa / sd:6.3f} {wins:3}/{len(rows)} | {new / so:6.3f} {main / new:6.2f} | {len(rows)}"
        )
