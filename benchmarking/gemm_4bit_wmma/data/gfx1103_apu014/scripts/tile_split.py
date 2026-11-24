"""tile_split.py <table> <csv>...: split the (shape, M) cells of an A-vs-F interleaved run (alt1103b.py) into those where
the table (rules1103 name or .py file, the default dispatch of F) picks the tile A ran and those where it picks another,
and print F / A (medians over the cycles) for both groups."""

from collections import defaultdict
import csv
import statistics as st
import sys

import rules1103 as R

rules = R.load(sys.argv[1])
for path in sys.argv[2:]:
    t = defaultdict(lambda: defaultdict(list))
    info = {}
    for r in csv.DictReader(open(path)):
        k = (r["shape"], int(r["M"]))
        t[k][r["cfg"]].append(float(r["us"]))
        if r["cfg"] == "A":
            info[k] = (int(r["N"]), int(r["K"]), r["tile"])
    groups = {"same tile": [], "other tile": []}
    for k, v in t.items():
        N, K, at = info[k]
        ft = R.pick(rules, k[1], N, K)
        groups["same tile" if ft == at else "other tile"].append((st.median(v["F"]) / st.median(v["A"]), k, at, ft))
    print(f"== {path.split('/')[-1]}")
    for g, cells in groups.items():
        cells.sort(reverse=True)
        print(
            f"{g}: {len(cells)} cells, F / A {cells[-1][0]:.3f}-{cells[0][0]:.3f}, "
            f"F slower: {sum(c[0] > 1 for c in cells)}, by > 1%: {sum(c[0] > 1.01 for c in cells)}, "
            f"by > 2%: {sum(c[0] > 1.02 for c in cells)}"
        )
        for ratio, (shape, M), at, ft in cells[:4]:
            print(f"   {ratio:.3f} {shape} M={M} A {at} F {ft}")
