"""leaf_search.py sweep.csv rules.cuh table M1,M2,.. : try prepending one n_lt(X, tile) leaf to the bucket covering the
given swept M values. Reports the best candidates by geomean (over shapes and the given M) without losing total time."""

from collections import defaultdict
import csv
import math
import sys

sys.path.insert(0, sys.path[0])
import rules_eval_lib as lib

t = defaultdict(dict)
for r in csv.DictReader(open(sys.argv[1])):
    t[(r["shape"], int(r["N"]), int(r["K"]), int(r["M"]))][r["cfg"]] = float(r["us"])
rules = lib.parse(open(sys.argv[2]).read(), sys.argv[3])
Ms = [int(x) for x in sys.argv[4].split(",")]
keys = [k for k in t if k[3] in Ms]


def cfg(tile, K):
    tile = list(tile)
    if K % 128:
        tile[5] = 1
    return ",".join(map(str, tile))


def score(pickf):
    lg = tot = 0
    worst = (0, None)
    for k in keys:
        c = pickf(k)
        if c not in t[k]:
            return None
        v = t[k][c]
        b = min(t[k].values())
        lg += math.log(v / b)
        tot += v
        if v / b > worst[0]:
            worst = (v / b, k)
    return math.exp(lg / len(keys)), tot, worst


base = score(lambda k: cfg(lib.pick(rules, k[3], k[1], k[2]), k[2]))
best_tot = sum(min(t[k].values()) for k in keys)
print(
    f"base: geomean {base[0]:.4f} total/best {base[1] / best_tot:.4f} worst {base[2][0]:.2f} {base[2][1][0]} M={base[2][1][3]}"
)
Ns = sorted({k[1] for k in keys})
Ks = sorted({k[2] for k in keys})
tiles = {c for k in keys for c in t[k] if c.endswith(",2")} | {c for k in keys for c in t[k]}
res = []
for feat, vals, idx in (("n_lt", Ns, 1), ("k_lt", Ks, 2)):
    for X in vals[1:]:
        for c in tiles:
            tile = tuple(int(x) for x in c.split(","))
            s = score(lambda k: cfg(tile, k[2]) if k[idx] < X else cfg(lib.pick(rules, k[3], k[1], k[2]), k[2]))
            if s and s[1] <= base[1] * 1.001:
                res.append((s[0], f"{feat}({X}", c, s))
res.sort()
for g, X, c, s in res[:60]:
    print(
        f"{X}, {{{c}}}): geomean {g:.4f} total/best {s[1] / best_tot:.4f} worst {s[2][0]:.2f} {s[2][1][0]} M={s[2][1][3]}"
    )
