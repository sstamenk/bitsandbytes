# ruff: noqa
"""merge_sweep.py out.csv a.csv b.csv: rows of b replace any (shape, M) also in a."""

import csv
import sys

rows_b = list(csv.DictReader(open(sys.argv[3])))
kb = {(r["shape"], r["M"]) for r in rows_b}
rows = [r for r in csv.DictReader(open(sys.argv[2])) if (r["shape"], r["M"]) not in kb] + rows_b
with open(sys.argv[1], "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["shape", "N", "K", "M", "cfg", "us"])
    w.writeheader()
    w.writerows(rows)
