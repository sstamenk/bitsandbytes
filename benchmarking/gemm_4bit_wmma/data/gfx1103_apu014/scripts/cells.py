"""cells.py <csv> <ref> <cfg>: per (shape, M) cell of an interleaved run: median time of both configurations over the
cycles, cfg / ref, and the same ratio per cycle."""

from collections import defaultdict
import csv
import statistics as st
import sys

t = defaultdict(lambda: defaultdict(dict))
for r in csv.DictReader(open(sys.argv[1])):
    t[(r["shape"], int(r["M"]))][r["cfg"]][int(r["cycle"])] = float(r["us"])
a, b = sys.argv[2], sys.argv[3]
for (shape, M), v in sorted(t.items()):
    cyc = sorted(set(v[a]) & set(v[b]))
    ma, mb = st.median(v[a].values()), st.median(v[b].values())
    per = " ".join(f"{v[b][c] / v[a][c]:.3f}" for c in cyc)
    print(f"{shape:28} M={M:4} {a} {ma:8.1f} us {b} {mb:8.1f} us {b}/{a} {mb / ma:.4f} | per cycle {per}")
