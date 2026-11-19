# ruff: noqa
"""fp32_cut_eval.py bench.csv...: current gfx115 fp32 cutoff table vs fitted per-M best cutoff vs oracle, from forced simt/wmma."""

from collections import defaultdict
import csv
import sys

CUR = [(3, 1 << 30), (5, 2560), (7, 1024)]


def cur(M):
    for mx, n in CUR:
        if M <= mx:
            return n
    return 0


t = defaultdict(dict)
for f in sys.argv[1:]:
    for r in csv.DictReader(open(f)):
        if r["dtype"] == "fp32" and r["kernel"] in ("simt", "wmma"):
            t[int(r["M"])].setdefault(r["shape"], {"N": int(r["N"])})[r["kernel"]] = float(r["us"])
print("   M  current(min_n) cur/orc  best(min_n) best/orc  cur/best  worst_cur")
for M in sorted(t):
    sh = [v for v in t[M].values() if "simt" in v and "wmma" in v]
    if len(sh) < 40:
        continue
    tot = lambda mn: sum(v["wmma"] if v["N"] >= mn else v["simt"] for v in sh)
    orc = sum(min(v["simt"], v["wmma"]) for v in sh)
    cands = sorted({v["N"] for v in sh}) + [1 << 30]
    bn = min(cands, key=tot)
    c = cur(M)
    worst = max((v["wmma"] if v["N"] >= c else v["simt"]) / min(v["simt"], v["wmma"]) for v in sh)
    print(
        f"{M:4}  {c if c < 1 << 30 else 'inf':>8}  {tot(c) / orc:7.3f}  {bn if bn < 1 << 30 else 'inf':>8}  {tot(bn) / orc:7.3f}  {tot(c) / tot(bn):7.3f}  {worst:5.2f}"
    )
