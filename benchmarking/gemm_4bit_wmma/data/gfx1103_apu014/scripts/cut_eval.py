"""cut_eval.py simt_wmma.csv: fp32 SIMT/WMMA cutoff tables vs per-shape oracle (total), per M."""

from collections import defaultdict
import csv
import sys

INF = 1 << 30
TABS = {
    "gfx11": [(4, INF), (5, 3072), (7, 2816), (8, 2048), (12, 2816), (64, 1024)],
    "gfx1103": [(2, INF), (4, 3840)],
    "gfx115": [(3, INF), (5, 2560), (7, 1024)],
    "gfx12": [(4, INF), (5, 8192), (6, 4096), (9, 2048), (15, 1408), (31, 1024)],
}
for a in sys.argv[2:]:
    n, s = a.split("=")
    TABS[n] = [tuple(map(int, x.split(":"))) for x in s.split(",")]


def wm(tab, M, N):
    for mx, mn in tab:
        if M <= mx:
            return N >= mn
    return True


t = defaultdict(dict)
for r in csv.DictReader(open(sys.argv[1])):
    if r["dtype"] == "fp32" and r["kernel"] in ("simt", "wmma"):
        t[(int(r["M"]), r["shape"])][r["kernel"]] = float(r["us"])
        t[(int(r["M"]), r["shape"])]["N"] = int(r["N"])
print("   M " + " ".join(f"{n:>14}" for n in TABS) + "   (total/oracle, worst shape)")
agg = defaultdict(float)
aggo = 0
for M in sorted({m for m, _ in t}):
    rows = [v for (m, s), v in t.items() if m == M and "simt" in v and "wmma" in v]
    if not rows:
        continue
    o = sum(min(v["simt"], v["wmma"]) for v in rows)
    aggo += o
    line = f"{M:4} "
    for n, tab in TABS.items():
        x = [v["wmma"] if wm(tab, M, v["N"]) else v["simt"] for v in rows]
        tot = sum(x)
        agg[n] += tot
        worst = max(xi / min(v["simt"], v["wmma"]) for xi, v in zip(x, rows))
        line += f" {tot / o:6.3f} {worst:5.2f}  "
    print(line)
print("all  " + " ".join(f" {agg[n] / aggo:6.3f}        " for n in TABS))
