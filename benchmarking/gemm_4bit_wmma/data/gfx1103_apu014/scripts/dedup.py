"""dedup.py in.csv out.csv: keep the last row of every (shape, M, dtype, kernel); q_evalF.sh times one shape
first as a warm-up (first-in-process readings run high) and again in its normal place."""

import csv
import sys

rows = {}
r = csv.DictReader(open(sys.argv[1]))
for x in r:
    k = (x["shape"], x["M"], x["dtype"], x["kernel"])
    rows.pop(k, None)
    rows[k] = x
w = csv.DictWriter(open(sys.argv[2], "w", newline=""), fieldnames=r.fieldnames)
w.writeheader()
w.writerows(rows.values())
