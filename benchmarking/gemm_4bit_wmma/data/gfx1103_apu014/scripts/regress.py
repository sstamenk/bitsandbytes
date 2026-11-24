"""regress.py altF.csv...: cells (shape, M) where the final default dispatch (F) is slower than the base dispatch (A)
by more than 10% (median over the cycles), per file, worst first; and the count above 20%."""

from collections import defaultdict
import csv
import os
import statistics as st
import sys

for f in sys.argv[1:]:
    t = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(f)):
        t[(r["shape"], int(r["M"]))][r["cfg"]].append(float(r["us"]))
    per = sorted(((st.median(v["F"]) / st.median(v["A"]), k) for k, v in t.items()), reverse=True)
    bad = [p for p in per if p[0] > 1.10]
    print(
        f"{os.path.basename(f)}: {len(per)} cells, F slower than A by > 10%: {len(bad)}, by > 20%: {sum(p[0] > 1.2 for p in per)}; "
        f"F faster by > 10%: {sum(p[0] < 0.9 for p in per)}, by > 20%: {sum(p[0] < 0.8 for p in per)}"
    )
    for x, (s, M) in bad[:20]:
        print(f"  {x:.2f} {s} M={M}")
