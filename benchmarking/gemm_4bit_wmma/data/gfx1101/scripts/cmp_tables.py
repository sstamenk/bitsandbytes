"""cmp_tables.py sweep.csv A.py B.py [the]: per M total B/A on the sweep, and shapes where B/A > the (default 1.10) or < 1/thr."""

from collections import defaultdict
import csv
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


the = float(sys.argv[4]) if len(sys.argv) > 4 else 1.10
t = defaultdict(dict)
for r in csv.DictReader(open(sys.argv[1])):
    t[(int(r["M"]), r["shape"], int(r["N"]), int(r["K"]))][r["cfg"]] = float(r["us"])
A, B = load(sys.argv[2]), load(sys.argv[3])
for M in sorted({k[0] for k in t}):
    ta = tb = 0
    bad = []
    for k, v in t.items():
        if k[0] != M:
            continue
        a, b = v[pick(A, M, k[2], k[3])], v[pick(B, M, k[2], k[3])]
        ta += a
        tb += b
        if b / a > the or b / a < 1 / the:
            bad.append(f"{k[1][:24]}({k[2]}x{k[3]}) {b / a:.2f} [{a:.0f}->{b:.0f}us]")
    print(f"M={M:4} B/A {tb / ta:.3f}  " + "; ".join(bad))
