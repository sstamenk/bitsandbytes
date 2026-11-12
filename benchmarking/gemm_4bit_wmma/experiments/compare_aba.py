"""Compare matching benchmark CSVs; show baseline drift and worst regression beyond both baselines."""
import csv
import sys
from collections import defaultdict

def read(path):
    return {(r["dtype"], int(r["M"]), r["shape"]): float(r["us"]) for r in csv.DictReader(open(path))}

a1, b, a2 = map(read, sys.argv[1:4])
assert a1.keys() == b.keys() == a2.keys(), "incomplete/mismatched runs"
groups = defaultdict(list)
for key in a1:
    groups[key[:2]].append(key)
print("dtype M A1_us B_us A2_us meanA/B A2/A1 outside_2pct_baseline_envelope worst_envelope_ratio shape")
for (dtype, m), keys in sorted(groups.items()):
    sums = [sum(d[k] for k in keys) for d in (a1, b, a2)]
    worst = max(keys, key=lambda k: b[k] / max(a1[k], a2[k]))
    bad = sum(b[k] > max(a1[k], a2[k]) * 1.02 for k in keys)
    print(dtype, m, *(round(v, 2) for v in sums), round((sums[0]+sums[2])/2/sums[1], 4),
          round(sums[2]/sums[0], 4), bad, round(b[worst]/max(a1[worst], a2[worst]), 4), worst[2])
