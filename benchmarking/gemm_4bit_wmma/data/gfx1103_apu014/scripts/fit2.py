"""fit2.py apu_sweep.csv rorlab_sweep.csv --ms 1,2,4,8 [--leaves 2] [--w 1,0.75,0.5,0.25] [--ref g1100_f16]
Joint fit of one M bucket on two machines: rule (first-match leaves "N < t" / "K < t" + otherwise) minimizing
w * (APU014 total / APU014 best) + (1 - w) * (RORLAB total / RORLAB best); RORLAB has 5 shapes only. Prints, per w
and leaf count, the rule and both ratios, with the --ref table's ratios, and with --fixed a given rule's."""

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
p.add_argument("apu")
p.add_argument("ror")
p.add_argument("--ms", required=True)
p.add_argument("--leaves", type=int, default=2)
p.add_argument("--w", default="1,0.75,0.5,0.25")
p.add_argument("--ref", default="g1100_f16")
p.add_argument("--fixed", action="append", default=[])
p.add_argument("--feats", default="N,K")
p.add_argument(
    "--maxreg", type=float, default=0, help="no tile slower than --ref by this factor on any APU shape/M of its region"
)
p.add_argument("--rormax", type=float, default=0, help="only print rules whose ROR ratio is at most this")
a = p.parse_args()
ms = [int(x) for x in a.ms.split(",")]


def rd(files):
    t = defaultdict(dict)
    for f in files.split(","):
        for r in csv.DictReader(open(f)):
            if int(r["M"]) in ms:
                t[(r["shape"], int(r["N"]), int(r["K"]))].setdefault(int(r["M"]), {})[r["cfg"]] = float(r["us"])
    return {s: v for s, v in t.items() if all(m in v for m in ms)}


ta, tr = rd(a.apu), rd(a.ror)
shapes = sorted(ta, key=lambda k: (k[1], k[2]))
tiles = sorted(set.intersection(*(set(ta[s][m]) for s in shapes for m in ms)))
XA = np.array([[sum(ta[s][m][c] for m in ms) for c in tiles] for s in shapes])
XR = np.array([[sum(tr[s][m].get(c, np.inf) for m in ms) if s in tr else 0.0 for c in tiles] for s in shapes])
BA = sum(sum(min(ta[s][m].values()) for m in ms) for s in shapes)
BR = sum(sum(min(tr[s][m].values()) for m in ms) for s in tr)
feat = {"N": np.array([s[1] for s in shapes]), "K": np.array([s[2] for s in shapes])}
R0 = load(a.ref)
REG = np.array(
    [[max(ta[s][m][c] / ta[s][m].get(pick(R0, m, s[1], s[2]), np.inf) for m in ms) for c in tiles] for s in shapes]
)
conds = [(f, v) for f in a.feats.split(",") for v in sorted(set(feat[f].tolist()))[1:]]


def solve(cs, X):
    rem = np.ones(len(shapes), bool)
    out = []
    idx = np.zeros(len(shapes), int)
    for f, v in [*cs, ("", 0)]:
        m = rem & (feat[f] < v) if f else rem
        if not m.any():
            return None
        x = X[m].sum(0)
        if a.maxreg:
            x = np.where(REG[m].max(0) <= a.maxreg, x, np.inf)
        j = x.argmin()
        if not np.isfinite(x[j]):
            return None
        out.append((f, v, tiles[j]))
        idx[m] = j
        rem &= ~m
    return X[np.arange(len(shapes)), idx].sum(), out, idx


def ratios(idx):
    r = np.arange(len(shapes))
    return XA[r, idx].sum() / BA, XR[r, idx].sum() / BR


def show(tag, rule, idx):
    ra, rr = ratios(idx)
    if a.rormax and rr > a.rormax and tag.startswith("w="):
        return
    reg = max(REG[i, idx[i]] for i in range(len(shapes)))
    tag = f"{tag} maxreg {reg:.2f}"
    wr = max((XR[i, idx[i]] / sum(min(tr[s][m].values()) for m in ms), s[0]) for i, s in enumerate(shapes) if s in tr)
    print(
        f"{tag:26} APU {ra:.3f} ROR {rr:.3f} (worst ROR shape {wr[0]:.2f} {wr[1][:22]}) | "
        + "; ".join((f"{f}<{v}: " if f else "else: ") + tl for f, v, tl in rule)
    )


def idx_of(fn):
    return np.array([tiles.index(fn(s)) for s in shapes])


try:
    show(a.ref, [("", 0, "table")], idx_of(lambda s: pick(R0, ms[0], s[1], s[2])))
except ValueError as e:
    print("ref n/a", e)
for fx in a.fixed:
    parts = fx.split(";")
    rule = []
    for q in parts[:-1]:
        c, tl = q.split(":")
        f, v = c.split("<")
        rule.append((f, int(v), tl))
    rule.append(("", 0, parts[-1]))

    def tile_of(s):
        for f, v, tl in rule:
            if f == "" or (f == "N" and s[1] < v) or (f == "K" and s[2] < v):
                return tl

    show("fixed", rule, idx_of(tile_of))
for w in map(float, a.w.split(",")):
    X = w * XA / BA + (1 - w) * XR / BR
    for n in range(0, a.leaves + 1):
        best = None
        for cs in itertools.product(conds, repeat=n):
            if len({c[0] + str(c[1]) for c in cs}) < n:
                continue
            r = solve(list(cs), X)
            if r and (best is None or r[0] < best[0] - 1e-12):
                best = r
        show(f"w={w} L={n}", best[1], best[2])
