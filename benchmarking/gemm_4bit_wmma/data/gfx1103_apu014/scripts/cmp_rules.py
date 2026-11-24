"""cmp_rules.py [--ref g1100_f16] [--buckets 8,16,32,48,64,96] table...: per M bucket, each table's tile vs the
per-shape best swept tile (total over the bucket's swept M and the shapes) on the APU014 fp16 sweep (40 shapes), the
APU014 bf16 sweep (M 1,4,8,16,32,64,128,256) and the RORLAB-35 fp16 sweep (5 shapes); the worst single cell vs the
--ref table on APU014 (fp16) and on RORLAB."""

import argparse
from collections import defaultdict
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rules1103 import load, pick

p = argparse.ArgumentParser()
p.add_argument("tables", nargs="+")
p.add_argument("--ref", default="g1100_f16")
p.add_argument("--buckets", default="8,16,32,48,64,96")
p.add_argument("--apu", required=True)
p.add_argument("--bf", required=True)
p.add_argument("--ror", required=True)
a = p.parse_args()
bk = [int(x) for x in a.buckets.split(",")] + [1 << 30]


def rd(fs):
    t = defaultdict(dict)
    for f in fs.split(","):
        for r in csv.DictReader(open(f)):
            t[(int(r["M"]), r["shape"], int(r["N"]), int(r["K"]))][r["cfg"]] = float(r["us"])
    return t


D = {"apu": rd(a.apu), "bf16": rd(a.bf), "ror": rd(a.ror)}
R0 = load(a.ref)
for name in a.tables:
    R = load(name)
    print(f"== {os.path.basename(name)}  (vs best | worst cell vs {a.ref})")
    for lo, hi in zip([0, *bk[:-1]], bk):
        line = f"  M {lo + 1:>3}-{hi if hi < 1 << 30 else 'inf':<4}"
        for dn, t in D.items():
            ks = [k for k in t if lo < k[0] <= hi]
            if not ks:
                line += f" | {dn}: n/a"
                continue
            tot = sum(t[k][pick(R, k[0], k[2], k[3])] for k in ks)
            best = sum(min(t[k].values()) for k in ks)
            ref = sum(t[k][pick(R0, k[0], k[2], k[3])] for k in ks)
            w = max((t[k][pick(R, k[0], k[2], k[3])] / t[k][pick(R0, k[0], k[2], k[3])], k[1][:20], k[0]) for k in ks)
            n10 = sum(t[k][pick(R, k[0], k[2], k[3])] / t[k][pick(R0, k[0], k[2], k[3])] > 1.10 for k in ks)
            line += f" | {dn}: {tot / best:.3f} (ref {ref / best:.3f}, new/ref {tot / ref:.3f}; >10% slower: {n10}, worst {w[0]:.2f} {w[1]} M={w[2]})"
        print(line)
