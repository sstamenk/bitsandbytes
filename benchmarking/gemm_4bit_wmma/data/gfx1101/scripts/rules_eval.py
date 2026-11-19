"""rules_eval.py sweep.csv rules.py [rules2.py ...]: rule tile vs per-shape best swept tile per M (total, geomean, worst, worst shape)."""

from collections import defaultdict
import csv
import math
import sys

INF = 1 << 30


def T(*a):
    return ",".join(map(str, a))


def load(x):
    ns = {"T": T, "INF": INF}
    exec(open(x).read(), ns)
    return ns["RULES"]


def pick(R, M, N, K):
    for mx, leaves, oth in R:
        if M <= mx:
            for f, lt, den, c in leaves:
                if (f == "N" and N < lt) or (f == "K" and K < lt) or (f == "KN" and K * den < lt * N):
                    c0 = c
                    break
            else:
                c0 = oth
            break
    p = c0.split(",")
    if K % 128:
        p[5] = "1"
    return ",".join(p)


t = defaultdict(dict)
for r in csv.DictReader(open(sys.argv[1])):
    t[(int(r["M"]), r["shape"], int(r["N"]), int(r["K"]))][r["cfg"]] = float(r["us"])
for rf in sys.argv[2:]:
    R = load(rf)
    print(f"== {rf}\n   M  rule/best(total) geomean worst  miss  worst-shape")
    allr = allb = 0
    for M in sorted({k[0] for k in t}):
        tot = best = lg = 0
        worst = (1, "")
        n = miss = 0
        for k, v in t.items():
            if k[0] != M:
                continue
            c = pick(R, M, k[2], k[3])
            if c not in v:
                miss += 1
                continue
            b = min(v.values())
            tot += v[c]
            best += b
            lg += math.log(v[c] / b)
            worst = max(worst, (v[c] / b, k[1]))
            n += 1
        allr += tot
        allb += best
        print(f"{M:4}  {tot / best:7.3f}  {math.exp(lg / max(n, 1)):7.3f}  {worst[0]:5.2f}  {miss:3}  {worst[1]}")
    print(f"all  {allr / allb:7.3f}")
