"""bandwidth_m1.py sweep.csv...: achieved bandwidth at M = 1 with the best swept tile: (4-bit weights + fp32 absmax
per 64 + A + C bytes) / time, the 12 fastest-streaming shapes per file."""

from collections import defaultdict
import csv
import os
import sys

for f in sys.argv[1:]:
    t = defaultdict(dict)
    for r in csv.DictReader(open(f)):
        if r["M"] == "1":
            t[(r["shape"], int(r["N"]), int(r["K"]))][r["cfg"]] = float(r["us"])
    rows = []
    for (s, N, K), v in t.items():
        b = N * K / 2 + N * K / 64 * 4 + K * 2 + N * 2
        c = min(v, key=v.get)
        rows.append((b / v[c] / 1e3, s, N, K, c, v[c], b / 1e6))
    print(os.path.basename(f))
    for r in sorted(rows, reverse=True)[:12]:
        print(f"  {r[0]:6.1f} GB/s {r[1][:28]:28} {r[2]:6} x {r[3]:6} tile {r[4]:12} {r[5]:8.1f} us {r[6]:6.1f} MB")
