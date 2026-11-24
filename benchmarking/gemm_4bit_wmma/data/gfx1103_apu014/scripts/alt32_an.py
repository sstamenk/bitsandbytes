"""alt32_an.py alt32.csv: fp32 dispatch from in-process interleaved runs (cfgs S=simt, A/B/D = forced WMMA tile tables).
Per M: total over shapes (per-shape median over cycles) of dispatch "cutoff table + tile cfg" vs the per-shape oracle
over S and that cfg; the best single N cutoff per M for B; per-cycle ratio range for new/cur."""

from collections import defaultdict
import csv
import statistics as st
import sys

INF = 1 << 30
CUT = {
    "g1103": [(2, INF), (4, 3840)],
    "g11": [(4, INF), (5, 3072), (7, 2816), (8, 2048), (12, 2816), (64, 1024)],
    "g115": [(3, INF), (5, 2560), (7, 1024)],
}
for a in sys.argv[2:]:
    n, s = a.split("=")
    CUT[n] = [tuple(int(x) if x != "INF" else INF for x in r.split(":")) for r in s.split(",")]


def wm(tab, M, N):
    for mx, mn in tab:
        if M <= mx:
            return N >= mn
    return True


t = defaultdict(lambda: defaultdict(list))
for r in csv.DictReader(open(sys.argv[1])):
    t[(int(r["M"]), r["shape"], int(r["N"]))][r["cfg"]].append((int(r["cycle"]), float(r["us"])))
med = lambda k, c: st.median(u for _, u in t[k][c])
DISP = [("cur", "g1103", "A")] + [
    (f"{c}+{x}", c, x) for c in CUT for x in ("B", "D") if c in ("g1103",) or c not in ("g11", "g115")
]
for M in sorted({k[0] for k in t}):
    ks = [k for k in t if k[0] == M]
    orc = sum(min(med(k, "S"), med(k, "B")) for k in ks)

    def tot(cut, x, cyc=None):
        f = (lambda k, c: med(k, c)) if cyc is None else (lambda k, c: next(u for y, u in t[k][c] if y == cyc))
        return sum(f(k, x) if wm(CUT[cut], M, k[2]) else f(k, "S") for k in ks)

    cur = tot("g1103", "A")
    line = f"M={M:3} oracle(S,B)={orc:9.0f} cur/orc {cur / orc:.3f}"
    for n, c, x in DISP[1:]:
        line += f" | {n} {tot(c, x) / orc:.3f}"
    # best single threshold for B at this M
    Ns = [*sorted({k[2] for k in ks}), INF]
    best = min((sum(med(k, "B") if k[2] >= th else med(k, "S") for k in ks), th) for th in Ns)
    line += f" | bestcut N>={best[1] if best[1] < INF else 'inf'} {best[0] / orc:.3f}"
    worst = max(
        (
            min(med(k, "S"), med(k, "B"))
            and (med(k, "B") if k[2] >= best[1] else med(k, "S")) / min(med(k, "S"), med(k, "B")),
            k[1][:20],
        )
        for k in ks
    )
    line += f" worst {worst[0]:.2f} {worst[1]}"
    print(line)
