"""cutoff_tables.py bench.csv... : total fp32 time over shapes per M for candidate SIMT/WMMA cutoff tables (forced simt/wmma)."""

from collections import defaultdict
import csv
import sys

INF = 1 << 30
TABLES = {
    "gfx115 (inherited)": [(3, INF), (5, 2560), (7, 1024)],
    "gfx1103": [(2, INF), (4, 3840)],
    "gfx11": [(4, INF), (5, 3072), (7, 2816), (8, 2048), (12, 2816), (64, 1024)],
    "refit on inherited tiles": [(3, INF), (4, 2560), (5, 1024)],
    "gfx1150 final": [(3, INF), (4, 1024)],
}


def pick(tab, M, N):
    for mx, mn in tab:
        if M <= mx:
            return N >= mn
    return True


t = defaultdict(dict)
for f in sys.argv[1:]:
    for r in csv.DictReader(open(f)):
        if r["dtype"] == "fp32" and r["kernel"] in ("simt", "wmma"):
            t[(int(r["M"]), r["shape"], int(r["N"]))][r["kernel"]] = float(r["us"])
Ms = sorted({k[0] for k in t if len(t[k]) == 2 and k[0] <= 16})
print("M    " + "  ".join(f"{n[:18]:>18}" for n in TABLES) + "   oracle  (ratio to oracle, worst shape ratio)")
for M in Ms:
    ks = [k for k in t if k[0] == M and len(t[k]) == 2]
    orc = sum(min(t[k].values()) for k in ks)
    s = f"{M:3d} "
    for name, tab in TABLES.items():
        tot = sum(t[k]["wmma" if pick(tab, M, k[2]) else "simt"] for k in ks)
        worst = max(t[k]["wmma" if pick(tab, M, k[2]) else "simt"] / min(t[k].values()) for k in ks)
        s += f"  {tot / orc:7.3f} ({worst:4.2f})    "
    print(s + f" {orc:9.0f}")
