# ruff: noqa
"""rules_eval.py sweep.csv f16|f32 [Ms]: current gfx115x rule tile vs per-shape best swept tile, per M (total, geomean, worst)."""

from collections import defaultdict
import csv
import math
import sys

INF = 1 << 30


def T(*a):
    return ",".join(map(str, a))


# (max_m, [(feature, lt, den, tile)], otherwise); feature N: N<lt, K: K<lt, KN: K*den < lt*N
F16 = [
    (8, [("N", 3072, 0, T(1, 1, 1, 1, 4, 1)), ("N", 28672, 0, T(1, 1, 1, 1, 8, 2))], T(1, 1, 1, 1, 2, 2)),
    (16, [("N", 3072, 0, T(2, 1, 1, 2, 4, 1)), ("N", 25600, 0, T(2, 2, 1, 1, 4, 2))], T(2, 1, 1, 2, 2, 2)),
    (24, [("K", 8192, 0, T(4, 1, 1, 4, 2, 2)), ("K", 10240, 0, T(4, 1, 1, 2, 1, 2))], T(1, 2, 1, 1, 8, 2)),
    (32, [("N", 3072, 0, T(2, 1, 1, 2, 8, 1)), ("N", 5120, 0, T(2, 2, 1, 2, 4, 2))], T(4, 1, 1, 4, 2, 2)),
    (48, [("N", 2816, 0, T(2, 1, 1, 4, 2, 2)), ("N", 3840, 0, T(2, 1, 1, 4, 1, 2))], T(2, 1, 1, 4, 4, 2)),
    (64, [("N", 3072, 0, T(4, 1, 1, 2, 4, 1)), ("N", 5120, 0, T(4, 1, 1, 4, 2, 2))], T(4, 2, 1, 4, 2, 2)),
    (128, [("K", 8192, 0, T(4, 2, 1, 4, 2, 2))], T(4, 1, 1, 4, 2, 2)),
    (256, [("KN", 6, 5, T(4, 2, 1, 4, 2, 2))], T(4, 1, 1, 4, 2, 2)),
    (INF, [("K", 21504, 0, T(4, 2, 1, 4, 1, 2))], T(4, 1, 1, 4, 2, 2)),
]
F32 = [
    (8, [("N", 2048, 0, T(1, 1, 1, 2, 4, 1)), ("N", 3072, 0, T(1, 1, 1, 4, 4, 1))], T(1, 2, 1, 2, 2, 2)),
    (16, [("K", 4608, 0, T(2, 2, 1, 2, 2, 1)), ("K", 8192, 0, T(2, 2, 1, 4, 1, 2))], T(2, 1, 1, 4, 1, 2)),
    (32, [("K", 3840, 0, T(1, 1, 1, 4, 2, 1)), ("K", 17408, 0, T(2, 2, 1, 4, 1, 2))], T(1, 2, 1, 2, 2, 2)),
    (64, [("N", 3840, 0, T(2, 2, 1, 4, 1, 2)), ("N", 10240, 0, T(2, 2, 1, 2, 2, 1))], T(2, 2, 1, 4, 1, 2)),
    (INF, [("N", 2048, 0, T(2, 2, 1, 4, 1, 2)), ("N", 3072, 0, T(2, 2, 1, 2, 1, 1))], T(2, 2, 1, 4, 1, 2)),
]
if sys.argv[2].endswith(".py"):
    ns = {"T": T, "INF": INF}
    exec(open(sys.argv[2]).read(), ns)
    RULES = ns["RULES"]
else:
    RULES = {"f16": F16, "f32": F32}[sys.argv[2]]


def pick(M, N, K):
    for mx, leaves, oth in RULES:
        if M <= mx:
            for f, lt, den, c in leaves:
                if (f == "N" and N < lt) or (f == "K" and K < lt) or (f == "KN" and K * den < lt * N):
                    return c
            return oth
    return RULES[-1][2]


def kch_fix(c, K):
    p = c.split(",")
    if K % 128:
        p[5] = "1"
    return ",".join(p)


t = defaultdict(dict)
for r in csv.DictReader(open(sys.argv[1])):
    t[(int(r["M"]), r["shape"], int(r["N"]), int(r["K"]))][r["cfg"]] = float(r["us"])
Ms = sorted({k[0] for k in t}) if len(sys.argv) < 4 else list(map(int, sys.argv[3].split(",")))
print("   M  rule/best(total) geomean worst(shape)  missing  n")
allr = allb = 0
for M in Ms:
    tot = best = lg = 0
    worst = (1, "")
    n = miss = 0
    for k, v in t.items():
        if k[0] != M:
            continue
        c = kch_fix(pick(M, k[2], k[3]), k[3])
        if c not in v:
            miss += 1
            continue
        b = min(v.values())
        tot += v[c]
        best += b
        lg += math.log(v[c] / b)
        n += 1
        worst = max(worst, (v[c] / b, k[1]))
    if not n:
        continue
    allr += tot
    allb += best
    print(f"{M:4}  {tot / best:7.3f}  {math.exp(lg / n):7.3f}  {worst[0]:5.2f} {worst[1][:28]:28} {miss} {n}")
print(f"all  {allr / allb:7.3f}")
