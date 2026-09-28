"""Derive a compact WMMA tile-selection rule from a sweep CSV.

For every M, find the config minimizing geomean(time/best) over shapes, then try
splitting shapes by a feature threshold and choosing one config per side."""

from collections import defaultdict
import csv
import math
import sys

rows = list(csv.DictReader(open(sys.argv[1])))
NUM_CUS = 96
t = defaultdict(dict)  # (shape, N, K, M) -> cfg -> us
for r in rows:
    t[(r["shape"], int(r["N"]), int(r["K"]), int(r["M"]))][r["cfg"]] = float(r["us"])
Ms = sorted({k[3] for k in t})


def loss(keys, cfg):
    tot, worst, n = 0.0, 1.0, 0
    for k in keys:
        v = t[k]
        if cfg not in v:
            return float("inf"), float("inf")
        r = v[cfg] / min(v.values())
        tot += math.log(r)
        worst = max(worst, r)
        n += 1
    return math.exp(tot / max(n, 1)), worst


def best_cfg(keys):
    common = set.intersection(*(set(t[k]) for k in keys)) if keys else set()
    return min(((loss(keys, c), c) for c in common), default=((float("inf"), float("inf")), None))


def cfg_tiles(cfg, N, M):
    wm, wn, wmp, wnp, ks, _kch = map(int, cfg.split(","))
    return math.ceil(N / (16 * wn * wnp)) * math.ceil(M / (8 * wm * wmp)) * wmp * wnp * ks


feats = {
    "n_tiles16": lambda N, K, M: N / 16,
    "weight_mb": lambda N, K, M: N * K / 2 / 1e6,
    "k_over_n": lambda N, K, M: K / N,
}
for M in Ms:
    keys = [k for k in t if k[3] == M]
    (g, w), c = best_cfg(keys)
    print(f"M={M:4}  single cfg {c:14}  geomean {g:.3f}  worst {w:.2f}")
    best_split = None
    for fname, f in feats.items():
        vals = sorted({f(k[1], k[2], M) for k in keys})
        for the in vals[1:]:
            lo = [k for k in keys if f(k[1], k[2], M) < the]
            hi = [k for k in keys if f(k[1], k[2], M) >= the]
            if len(lo) < 4 or len(hi) < 4:
                continue
            (gl, wl), cl = best_cfg(lo)
            (gh, wh), ch = best_cfg(hi)
            gg = math.exp((len(lo) * math.log(gl) + len(hi) * math.log(gh)) / len(keys))
            if best_split is None or gg < best_split[0]:
                best_split = (gg, max(wl, wh), fname, the, cl, ch, len(lo), len(hi))
    if best_split and best_split[0] < g - 0.01:
        gg, ww, fname, the, cl, ch, nl, nh = best_split
        print(
            f"          split {fname} < {the:.3g}: {cl} ({nl} shapes) else {ch} ({nh})  geomean {gg:.3f} worst {ww:.2f}"
        )
