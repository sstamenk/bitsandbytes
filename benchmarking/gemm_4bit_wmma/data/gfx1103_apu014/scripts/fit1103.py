"""fit1103.py sweep.csv[,sweep2.csv...] --ms 1,2,4,8 [--leaves 3] [--min-gain 0.01] [--feats N,K] [--ref g1100_f16]
Joint fit of ONE M bucket in the rule format (up to 3 first-match leaves "N < t" / "K < t", each on its own feature, plus
otherwise), minimizing the total time (sum over shapes and the bucket's swept M) / per-shape best. Exhaustive over
leaf conditions; the tile per region is the region's best common tile. A leaf is only added if it gains >= --min-gain
in total. Prints the rule, per-M totals, worst shape vs best, and shapes slower than --ref by > 10%."""

import argparse
from collections import defaultdict
import csv
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rules1103 import load, pick

p = argparse.ArgumentParser()
p.add_argument("sweep")
p.add_argument("--ms", required=True)
p.add_argument("--leaves", type=int, default=3)
p.add_argument("--min-gain", type=float, default=0.01)
p.add_argument("--feats", default="N,K")
p.add_argument("--ref", default="g1100_f16")
p.add_argument("--show", type=int, default=5)
p.add_argument(
    "--maxreg",
    type=float,
    default=0,
    help="only tiles no worse than --ref by this factor on any shape/M of their region",
)
p.add_argument("--fixed", default="", help="evaluate a given rule instead: 'N<1024:tile;K<4096:tile;tile'")
a = p.parse_args()
ms = [int(x) for x in a.ms.split(",")]
t = defaultdict(dict)
for f in a.sweep.split(","):
    for r in csv.DictReader(open(f)):
        if int(r["M"]) in ms:
            t[(r["shape"], int(r["N"]), int(r["K"]))].setdefault(int(r["M"]), {})[r["cfg"]] = float(r["us"])
shapes = sorted(t, key=lambda k: (k[1], k[2]))
shapes = [s for s in shapes if all(m in t[s] for m in ms)]
tiles = sorted(set.intersection(*(set(t[s][m]) for s in shapes for m in ms)))
X = np.array([[sum(t[s][m][c] for m in ms) for c in tiles] for s in shapes])  # shape x tile, summed over M
best = np.array([sum(min(t[s][m].values()) for m in ms) for s in shapes])
BT = best.sum()
feat = {"N": np.array([s[1] for s in shapes]), "K": np.array([s[2] for s in shapes])}
R0 = load(a.ref)
REG = np.array(
    [[max(t[s][m][c] / t[s][m].get(pick(R0, m, s[1], s[2]), float("inf")) for m in ms) for c in tiles] for s in shapes]
)
conds = [(f, v) for f in a.feats.split(",") for v in sorted(set(feat[f].tolist()))[1:]]


def pickj(m):
    x = X[m].sum(0)
    if a.maxreg:
        x = np.where(REG[m].max(0) <= a.maxreg, x, np.inf)
    j = x.argmin()
    return None if not np.isfinite(x[j]) else j


def solve(cs):
    rem = np.ones(len(shapes), bool)
    tot = 0
    out = []
    for f, v in cs:
        m = rem & (feat[f] < v)
        if not m.any():
            return None
        j = pickj(m)
        if j is None:
            return None
        tot += X[m, j].sum()
        out.append((f, v, tiles[j]))
        rem &= ~m
    if not rem.any():
        return None
    j = pickj(rem)
    if j is None:
        return None
    tot += X[rem, j].sum()
    out.append(("", 0, tiles[j]))
    return tot, out


if a.fixed:
    parts = a.fixed.split(";")
    rule = []
    for q in parts[:-1]:
        c, tl = q.split(":")
        f, v = c.split("<")
        rule.append((f, int(v), tl))
    rule.append(("", 0, parts[-1]))
    res = (None, rule)
else:
    res = solve([]) or (float("inf"), None)
    for n in range(1, a.leaves + 1):
        bestn = None
        for cs in itertools.product(conds, repeat=n):
            r = solve(cs)
            if r and (bestn is None or r[0] < bestn[0]):
                bestn = r
        if bestn and bestn[0] < res[0] * (1 - a.min_gain):
            res = bestn
rule = res[1]


def tile_of(s):
    for f, v, tl in rule:
        if f == "" or (f == "N" and s[1] < v) or (f == "K" and s[2] < v):
            return tl


R = load(a.ref)
print("rule: " + "; ".join((f"{f}<{v}: " if f else "else: ") + tl for f, v, tl in rule))
for m in ms:
    tot = sum(t[s][m][tile_of(s)] for s in shapes)
    bt = sum(min(t[s][m].values()) for s in shapes)
    rt = sum(t[s][m].get(pick(R, m, s[1], s[2]), float("nan")) for s in shapes)
    print(f"  M={m:<4} total/best {tot / bt:.3f}  {a.ref}/best {rt / bt:.3f}  new/{a.ref} {tot / rt:.3f}")
tot = sum(t[s][m][tile_of(s)] for s in shapes for m in ms)
print(
    f"  bucket total/best {tot / BT:.3f}; ref/best {sum(t[s][m].get(pick(R, m, s[1], s[2]), float('nan')) for s in shapes for m in ms) / BT:.3f}"
)
rows = []
for s in shapes:
    for m in ms:
        v = t[s][m]
        c = tile_of(s)
        r = v.get(pick(R, m, s[1], s[2]))
        rows.append((v[c] / min(v.values()), v[c] / r if r else float("nan"), s[0], m))
print(
    "  worst vs best: " + ", ".join(f"{x[2][:22]} M={x[3]} {x[0]:.2f}" for x in sorted(rows, reverse=True)[: a.show])
)
bad = sorted([x for x in rows if x[1] > 1.10], key=lambda x: -x[1])
print(f"  slower than {a.ref} by >10%: " + (", ".join(f"{x[2][:22]} M={x[3]} {x[1]:.2f}" for x in bad[:12]) or "none"))
