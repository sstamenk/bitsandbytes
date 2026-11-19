# ruff: noqa
"""joint.py out.csv a.csv:tagA b.csv:tagB ...: concatenate sweeps, shape names prefixed by tag, only (shape,M) present in all."""

import csv
import sys

srcs = [a.rsplit(":", 1) for a in sys.argv[2:]]
data = []
for f, tag in srcs:
    data.append((tag, list(csv.DictReader(open(f)))))
keysets = [{(r["shape"], r["M"]) for r in rows} for _, rows in data]
common = set.intersection(*keysets)
with open(sys.argv[1], "w", newline="") as fout:
    w = csv.DictWriter(fout, fieldnames=["shape", "N", "K", "M", "cfg", "us"])
    w.writeheader()
    for tag, rows in data:
        for r in rows:
            if (r["shape"], r["M"]) in common:
                r = dict(r)
                r["shape"] = tag + ":" + r["shape"]
                w.writerow(r)
