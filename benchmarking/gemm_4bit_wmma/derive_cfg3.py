"""Fit WMMA tile rules with up to three N leaves per M bucket from a sweep CSV.

derive_cfg.py [sweep.csv] finds one tile or a two-way split per swept M. This fits the shape the
launcher rules take: per M bucket (the swept M values, merged when they share a rule), up to two
N thresholds and one tile per leaf. The score is the geomean over shapes of time / per-shape best
swept tile; the total-time ratio over the 40 shapes is printed as well. A split is only kept if it
gains at least --min-gain (default 1%) in geomean over the simpler rule.

usage: python derive_cfg3.py sweep.csv [--min-gain 0.01] [--merge 0.01]
"""

import argparse
from collections import defaultdict
import csv
import math

p = argparse.ArgumentParser()
p.add_argument("sweep")
p.add_argument("--min-gain", type=float, default=0.01)
p.add_argument("--merge", type=float, default=0.01, help="merge adjacent M buckets if the joint rule loses <= this")
p.add_argument("--score", choices=["geomean", "total"], default="geomean", help="objective: geomean of time/best or total time")
args = p.parse_args()

t = defaultdict(dict)  # (shape, N, K, M) -> cfg -> us
for r in csv.DictReader(open(args.sweep)):
    t[(r["shape"], int(r["N"]), int(r["K"]), int(r["M"]))][r["cfg"]] = float(r["us"])
Ms = sorted({k[3] for k in t})


def score(keys, cfg):
    lg, tot, best_tot, worst = 0.0, 0.0, 0.0, 1.0
    for k in keys:
        v = t[k]
        if cfg not in v:
            return None
        b = min(v.values())
        lg += math.log(v[cfg] / b)
        tot += v[cfg]
        best_tot += b
        worst = max(worst, v[cfg] / b)
    return lg, tot, best_tot, worst


def obj(s):
    return s[1] if args.score == "total" else s[0]


def best_leaf(keys):
    common = set.intersection(*(set(t[k]) for k in keys))
    best = None
    for c in common:
        s = score(keys, c)
        if s and (best is None or obj(s) < obj(best[0])):
            best = (s, c)
    return best


def fit(keys):
    """Best rule with 1, 2 or 3 N leaves: (geomean, total ratio, worst, [(n_lo, cfg), ...])."""
    n = len(keys)
    Ns = sorted({k[1] for k in keys})

    def summarize(leaves):
        lg = sum(s[0] for s, _, _ in leaves)
        tot = sum(s[1] for s, _, _ in leaves)
        bt = sum(s[2] for s, _, _ in leaves)
        return math.exp(lg / n), tot / bt, max(s[3] for s, _, _ in leaves), [(lo, c) for _, c, lo in leaves]

    def leaf(lo, hi):
        ks = [k for k in keys if lo <= k[1] < hi]
        if len(ks) < 3:
            return None
        b = best_leaf(ks)
        return (b[0], b[1], lo) if b else None

    key = (lambda r: r[1]) if args.score == "total" else (lambda r: r[0])
    one = summarize([leaf(0, 1 << 30)])
    best2 = one
    for a in Ns[1:]:
        l1, l2 = leaf(0, a), leaf(a, 1 << 30)
        if l1 and l2:
            s = summarize([l1, l2])
            if key(s) < key(best2):
                best2 = s
    best3 = best2
    for i, a in enumerate(Ns[1:], 1):
        l1 = leaf(0, a)
        if not l1:
            continue
        for b in Ns[i + 1 :]:
            l2, l3 = leaf(a, b), leaf(b, 1 << 30)
            if l2 and l3:
                s = summarize([l1, l2, l3])
                if key(s) < key(best3):
                    best3 = s
    rule = one
    if key(best2) < key(rule) * (1 - args.min_gain):
        rule = best2
    if key(best3) < key(rule) * (1 - args.min_gain):
        rule = best3
    return rule


# Fit per swept M, then greedily merge adjacent buckets whose joint rule costs <= --merge.
buckets = [[m] for m in Ms]
fits = {}


def keys_of(ms):
    return [k for k in t if k[3] in ms]


def get(ms):
    key = tuple(ms)
    if key not in fits:
        fits[key] = fit(keys_of(ms))
    return fits[key]


merged = True
while merged:
    merged = False
    for i in range(len(buckets) - 1):
        a, b = buckets[i], buckets[i + 1]
        ix = 1 if args.score == "total" else 0
        ga, gb, gab = get(a)[ix], get(b)[ix], get(a + b)[ix]
        na, nb = len(keys_of(a)), len(keys_of(b))
        sep = math.exp((na * math.log(ga) + nb * math.log(gb)) / (na + nb))
        if gab <= sep * (1 + args.merge):
            buckets[i : i + 2] = [a + b]
            merged = True
            break

for ms in buckets:
    g, tr, w, leaves = get(ms)
    desc = "; ".join(f"N>={lo}: {c}" if lo else c for lo, c in leaves)
    print(f"M {ms[0]:>3}-{ms[-1]:<3}  geomean {g:.3f}  total {tr:.3f}  worst {w:.2f}  {desc}")
    if len(ms) > 1:
        for m in ms:
            ks = keys_of([m])
            lg = tot = bt = 0.0
            worst = 1.0
            for k in ks:
                c = next(c for lo, c in reversed(leaves) if k[1] >= lo)
                v = t[k]
                lg += math.log(v[c] / min(v.values()))
                tot += v[c]
                bt += min(v.values())
                worst = max(worst, v[c] / min(v.values()))
            print(f"    M={m:<4} geomean {math.exp(lg / len(ks)):.3f}  total {tot / bt:.3f}  worst {worst:.2f}")
