"""new_vs_old.py old.csv new.csv [dtype]: auto of the new rules vs auto of the inherited ones per M (total, geomean,
worst shape, shapes >5% slower)."""

import csv
import math
import sys


def load(f, dt):
    d = {}
    for r in csv.DictReader(open(f)):
        if r["kernel"] == "auto" and r["dtype"] == dt:
            d[(int(r["M"]), r["shape"])] = float(r["us"])
    return d


dt = sys.argv[3] if len(sys.argv) > 3 else "fp16"
o, n = load(sys.argv[1], dt), load(sys.argv[2], dt)
print(f"== {dt}: M | old/new total | geomean old/new | worst new/old (shape) | shapes new >5% slower")
for M in sorted({k[0] for k in n}):
    ks = [k for k in n if k in o and k[0] == M]
    if not ks:
        continue
    r = {k: n[k] / o[k] for k in ks}
    w = max(ks, key=lambda k: r[k])
    print(
        f"{M:4d} {sum(o[k] for k in ks) / sum(n[k] for k in ks):.3f} {math.exp(-sum(math.log(r[k]) for k in ks) / len(ks)):.3f} "
        f"{r[w]:.2f} ({w[1][:28]}) {sum(r[k] > 1.05 for k in ks)}"
    )
