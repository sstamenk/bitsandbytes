"""final_an.py <pinned dir> <out dir>: assemble pinned/eval_base.csv and pinned/eval_final.csv and print the dispatch
summary of both.
eval_base.csv  = evalA (base build 540f377): simt, dequant, auto, fp32 wmma; the auto rows of llama3-70b down at
                 M <= 4 (the first shape of each bench worker, first-in-process readings up to 3x high) are replaced by
                 evalA_fix (base build, same shape after a warm-up shape).
eval_final.csv = evalF (final build): auto and wmma; simt and dequant from evalA (kernels and BLAS path unchanged).
Summary per dtype and M: total over the 40 shapes of auto / per-shape oracle (min of simt, wmma, dequant, auto), worst
shape, fused vs dequant (total, shapes ahead), default dispatch incl. the Python M limit vs oracle, and upstream main
(SIMT through M = 8, dequant + F.linear above) / default dispatch."""

from collections import defaultdict
import csv
import glob
import os
import sys

L, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)


def rd(pat):
    rows = []
    for f in sorted(glob.glob(pat)):
        if not f.endswith(".raw.csv"):
            rows += list(csv.DictReader(open(f)))  # .raw.csv: with the warm-up rows
    return rows


base = [r for r in rd(f"{L}/evalA/*.csv")]
fix = {(r["shape"], r["M"], r["dtype"]): r for r in rd(f"{L}/evalA_fix/*_auto.csv")}
nfix = 0
for r in base:
    k = (r["shape"], r["M"], r["dtype"])
    if r["kernel"] == "auto" and r["shape"] == "llama3-70b down" and int(r["M"]) <= 4 and k in fix:
        r["us"] = fix[k]["us"]
        nfix += 1
fin = rd(f"{L}/evalF/*.csv")
fin = [r for r in fin if r["kernel"] in ("auto", "wmma")] + [r for r in base if r["kernel"] in ("simt", "dequant")]
F = ["shape", "N", "K", "M", "dtype", "kernel", "us"]
for name, rows in (("eval_base.csv", base), ("eval_final.csv", fin)):
    with open(f"{OUT}/{name}", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=F, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
print(f"eval_base.csv: {len(base)} rows ({nfix} auto rows from evalA_fix); eval_final.csv: {len(fin)} rows")


def summary(rows, cap, title):
    t = defaultdict(dict)
    for r in rows:
        t[(r["dtype"], int(r["M"]), r["shape"])][r["kernel"]] = float(r["us"])
    for dt in ("fp16", "bf16", "fp32"):
        c = cap[dt == "fp32"]
        print(f"== {title} {dt} (M limit {c})")
        print("   M | auto/orc worst | fused/deq ahead | dispatch/orc worst | main/dispatch")
        for M in sorted({k[1] for k in t if k[0] == dt}):
            vs = [v for k, v in t.items() if k[0] == dt and k[1] == M and "auto" in v and "dequant" in v]
            if len(vs) < 40:
                continue
            orc = [min(v.values()) for v in vs]
            a = [v["auto"] for v in vs]
            d = [v["dequant"] for v in vs]
            disp = a if M <= c else d
            main = [v.get("simt", v["auto"]) for v in vs] if M <= 8 else d
            print(
                f"{M:4} | {sum(a) / sum(orc):6.3f} {max(x / o for x, o in zip(a, orc)):5.2f} | {sum(a) / sum(d):6.3f} {sum(x < y for x, y in zip(a, d)):2}/40 | "
                f"{sum(disp) / sum(orc):6.3f} {max(x / o for x, o in zip(disp, orc)):5.2f} | {sum(main) / sum(disp):5.2f}"
            )


summary(base, (384, 384), "base (gfx1100 tiles, limits 384/384)")
summary(fin, (512, 512), "final")
