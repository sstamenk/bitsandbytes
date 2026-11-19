"""regress.py sweep.csv A B [threshold]: per (M, shape) B/A tile-time ratio from the sweep, listing > threshold and < 1/threshold."""

from collections import defaultdict
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rules11 import load, pick

t = defaultdict(dict)
for r in csv.DictReader(open(sys.argv[1])):
    t[(int(r["M"]), r["shape"], int(r["N"]), int(r["K"]))][r["cfg"]] = float(r["us"])
A, B = load(sys.argv[2]), load(sys.argv[3])
threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 1.1
for k in sorted(t):
    M, s, N, K = k
    v = t[k]
    a, b = v.get(pick(A, M, N, K)), v.get(pick(B, M, N, K))
    if a and b and b / a > threshold:
        print(f"M={M:4} {s[:32]:32} {N:6} {K:6} B/A {b / a:5.2f}  ({a:8.1f} -> {b:8.1f} us)")
