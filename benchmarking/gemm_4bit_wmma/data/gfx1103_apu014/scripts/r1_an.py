"""r1_an.py r1.csv...: A = gfx1100 table, F = final table, 10 alternating cycles (A first on even cycles, F first on
odd). Per file, for M = 49-64 and M = 65-96 separately: F / A from the per-cell medians over all cycles, over the
even and over the odd cycles (order-conditioned), the range of the per-cycle totals, per M, and the cells where F is
slower than A (count, worst)."""

from collections import defaultdict
import csv
import os
import statistics as st
import sys

for f in sys.argv[1:]:
    t = defaultdict(lambda: defaultdict(dict))
    for r in csv.DictReader(open(f)):
        t[(int(r["M"]), r["shape"])][r["cfg"]][int(r["cycle"])] = float(r["us"])
    print(os.path.basename(f), "cycles", sorted({c for v in t.values() for c in v["A"]}))

    def ratio(ks, cyc=None):
        m = lambda k, c: st.median(u for y, u in t[k][c].items() if cyc is None or y in cyc)
        return sum(m(k, "F") for k in ks) / sum(m(k, "A") for k in ks)

    for name, lo, hi in (("M = 49-64", 49, 64), ("M = 65-96", 65, 96), ("M = 49-96", 49, 96)):
        ks = [k for k in t if lo <= k[0] <= hi]
        cyc = sorted(set.intersection(*(set(t[k]["A"]) & set(t[k]["F"]) for k in ks)))
        per = [sum(t[k]["F"][y] for k in ks) / sum(t[k]["A"][y] for k in ks) for y in cyc]
        cell = sorted(
            ((st.median(t[k]["F"].values()) / st.median(t[k]["A"].values()), k[1][:26], k[0]) for k in ks),
            reverse=True,
        )
        print(
            f"  {name}: F/A {ratio(ks):.4f} (gain {100 * (1 - ratio(ks)):.2f}%); A-first cycles {ratio(ks, set(cyc[0::2])):.4f}, F-first cycles {ratio(ks, set(cyc[1::2])):.4f}; "
            f"per-cycle totals {min(per):.4f}-{max(per):.4f}; cells with F slower: {sum(c[0] > 1 for c in cell)}/{len(cell)}, worst {cell[0][0]:.4f} {cell[0][1]} M={cell[0][2]}"
        )
    print("  per M: " + " ".join(f"{M}:{ratio([k for k in t if k[0] == M]):.4f}" for M in sorted({k[0] for k in t})))
