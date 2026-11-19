# ruff: noqa
"""cmp_tables.py sweep.csv ref.py|f16 cand.py...: per M total of each table / ref table and / oracle, common points."""

from collections import defaultdict
import csv
import sys

INF = 1 << 30


def T(*a):
    return ",".join(map(str, a))


def load_rules(x):
    if x == "f16":
        ns = {}
        exec(
            open(__file__.replace("cmp_tables.py", "rules_eval.py"))
            .read()
            .split("RULES = ")[0]
            .split("if sys.argv")[0],
            ns,
        )
        return ns["F16"]
    ns = {"T": T, "INF": INF}
    exec(open(x).read(), ns)
    return ns["RULES"]


def pick(R, M, N, K):
    for mx, leaves, oth in R:
        if M <= mx:
            for f, lt, den, c in leaves:
                if (f == "N" and N < lt) or (f == "K" and K < lt) or (f == "KN" and K * den < lt * N):
                    return c
            return oth
    return R[-1][2]


def fix(c, K):
    p = c.split(",")
    if K % 128:
        p[5] = "1"
    return ",".join(p)


t = defaultdict(dict)
for r in csv.DictReader(open(sys.argv[1])):
    t[(int(r["M"]), r["shape"], int(r["N"]), int(r["K"]))][r["cfg"]] = float(r["us"])
tabs = [(x, load_rules(x)) for x in sys.argv[2:]]
print(
    "M    "
    + "  ".join(f"{x[:10]:>10}/ref worst" for x, _ in tabs[1:])
    + "   ref/orc "
    + " ".join(f"{x[:8]}/orc" for x, _ in tabs[1:])
)
for M in sorted({k[0] for k in t}):
    ks = [k for k in t if k[0] == M and all(fix(pick(R, M, k[2], k[3]), k[3]) in t[k] for _, R in tabs)]
    if not ks:
        continue
    tot = [sum(t[k][fix(pick(R, M, k[2], k[3]), k[3])] for k in ks) for _, R in tabs]
    orc = sum(min(t[k].values()) for k in ks)
    worst = [
        max(
            (t[k][fix(pick(R, M, k[2], k[3]), k[3])] / t[k][fix(pick(tabs[0][1], M, k[2], k[3]), k[3])], k[1][:20])
            for k in ks
        )
        for _, R in tabs[1:]
    ]
    print(
        f"{M:4} "
        + "  ".join(f"{tt / tot[0]:14.3f} {w[0]:4.2f}" for tt, w in zip(tot[1:], worst))
        + f"   {tot[0] / orc:6.3f} "
        + " ".join(f"{tt / orc:10.3f}" for tt in tot[1:])
        + f"  n={len(ks)}"
    )
