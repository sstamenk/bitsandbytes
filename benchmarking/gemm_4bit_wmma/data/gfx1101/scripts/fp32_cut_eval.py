"""fp32_cut_eval.py simt_wmma.csv: total time of each target's fp32 SIMT/WMMA cutoff table vs per-shape oracle, per M."""

from collections import defaultdict
import csv
import sys

INF = 1 << 30
TABLES = {
    "gfx11": [(4, INF), (5, 3072), (7, 2816), (8, 2048), (12, 2816), (64, 1024)],
    "gfx1103": [(2, INF), (4, 3840)],
    "gfx115": [(3, INF), (5, 2560), (7, 1024)],
    "gfx12": [(4, INF), (5, 8192), (6, 4096), (9, 2048), (15, 1408), (31, 1024)],
}
for a in sys.argv[2:]:
    n, s = a.split("=")
    TABLES[n] = eval(s)


def pick(tab, M, N):
    for mx, mn in tab:
        if M <= mx:
            return N >= mn
    return True


t = defaultdict(dict)
for r in csv.DictReader(open(sys.argv[1])):
    t[int(r["M"])].setdefault(r["shape"], {"N": int(r["N"])})[r["kernel"]] = float(r["us"])
print("   M " + " ".join(f"{k:>16}" for k in TABLES))
agg = defaultdict(float)
aggo = 0
for M in sorted(t):
    sh = [v for v in t[M].values() if "simt" in v and "wmma" in v]
    if not sh:
        continue
    o = sum(min(v["simt"], v["wmma"]) for v in sh)
    aggo += o
    row = []
    for k, tab in TABLES.items():
        tot = sum(v["wmma"] if pick(tab, M, v["N"]) else v["simt"] for v in sh)
        agg[k] += tot
        worst = max((v["wmma"] if pick(tab, M, v["N"]) else v["simt"]) / min(v["simt"], v["wmma"]) for v in sh)
        row.append(f"{tot / o:7.3f} ({worst:4.2f})")
    print(f"{M:4} " + " ".join(f"{x:>16}" for x in row))
print("all  " + " ".join(f"{agg[k] / aggo:16.4f}" for k in TABLES))
