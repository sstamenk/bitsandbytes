"""order_cmp.py rtf.csv nfirst.csv [rules.cuh table]: row-tiles-first vs N-first sweeps with the same tiles.
Per M: total of per-shape best (rtf / nfirst), and, with a rule table, total of the rule's tiles under each order."""

from collections import defaultdict
import csv
import sys

sys.path.insert(0, sys.path[0])


def load(f):
    t = defaultdict(dict)
    for r in csv.DictReader(open(f)):
        if int(r["K"]) % 128 and r["cfg"].endswith(",2"):
            continue
        t[(r["shape"], int(r["N"]), int(r["K"]), int(r["M"]))][r["cfg"]] = float(r["us"])
    return t


a, b = load(sys.argv[1]), load(sys.argv[2])
rules = None
if len(sys.argv) > 4:
    import importlib.util

    spec = importlib.util.spec_from_file_location("re_", sys.path[0] + "/rules_eval_lib.py")
    lib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lib)
    rules = lib.parse(open(sys.argv[3]).read(), sys.argv[4])
print("M | best-rtf/best-nfirst total, shapes nfirst better | rule rtf/nfirst total")
for M in sorted({k[3] for k in a}):
    ks = [k for k in a if k[3] == M and k in b]
    ba = sum(min(a[k].values()) for k in ks)
    bb = sum(min(b[k].values()) for k in ks)
    nb = sum(min(b[k].values()) < min(a[k].values()) for k in ks)
    s = f"{M:4d} {ba / bb:.3f} {nb}/{len(ks)}"
    if rules:
        ra = rb = 0
        for k in ks:
            tile = list(lib.pick(rules, M, k[1], k[2]))
            if k[2] % 128:
                tile[5] = 1
            c = ",".join(map(str, tile))
            if c in a[k] and c in b[k]:
                ra += a[k][c]
                rb += b[k][c]
        s += f" | {ra / rb:.3f}" if rb else " | n/a"
    print(s)
