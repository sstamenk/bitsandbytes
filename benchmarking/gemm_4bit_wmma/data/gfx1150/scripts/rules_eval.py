"""rules_eval.py rules.cuh sweep.csv table_name [table_name ...]
Total time of the tiles a rule table picks vs the per-shape best swept tile, per M (and overall).

A tile with kch = 2 on a K % 128 != 0 shape runs with kch = 1, which the sweep measured as the kch = 1 tile, except for
fp16 tables (name contains "f16") where the gfx11 prescaled kernel would run (it keys on the tile's kch = 2: M >= 96,
wm >= 2, M <= 128 or K <= N): that kernel is not in the sweep, so those (shape, M) count as missing."""

from collections import defaultdict
import csv
import re
import sys

src = open(sys.argv[1]).read()


def parse(name):
    blk = src[src.index(name + "[] = {") :]
    blk = blk[: blk.index("};")]
    rules = []
    for line in blk.splitlines():
        m = re.match(r"\s*\{(\d+|INT_MAX),\s*\{(.*)\},\s*\{([\d, ]+)\}\},", line)
        if not m:
            continue
        mx = 1 << 30 if m.group(1) == "INT_MAX" else int(m.group(1))
        leaves = []
        for f, args, tile in re.findall(r"(n_lt|k_lt|k_over_n_lt)\(([\d, ]+?),\s*\{([\d, ]+)\}\)", m.group(2)):
            nums = [int(x) for x in args.split(",") if x.strip()]
            leaves.append((f, nums, tuple(int(x) for x in tile.split(","))))
        rules.append((mx, leaves, tuple(int(x) for x in m.group(3).split(","))))
    return rules


def pick(rules, M, N, K):
    for mx, leaves, other in rules:
        if M > mx:
            continue
        for f, nums, tile in leaves:
            if f == "n_lt" and N < nums[0]:
                return tile
            if f == "k_lt" and K < nums[0]:
                return tile
            if f == "k_over_n_lt" and K * nums[1] < nums[0] * N:
                return tile
        return other
    return rules[-1][2]


t = defaultdict(dict)
for r in csv.DictReader(open(sys.argv[2])):
    t[(r["shape"], int(r["N"]), int(r["K"]), int(r["M"]))][r["cfg"]] = float(r["us"])
for name in sys.argv[3:]:
    rules = parse(name)
    print(f"== {name}: M | rule total / best-per-shape total | worst | missing")
    gtot = gbest = 0
    for M in sorted({k[3] for k in t}):
        tot = best = 0
        worst = (1, "")
        miss = 0
        for k, v in t.items():
            if k[3] != M:
                continue
            tile = list(pick(rules, M, k[1], k[2]))
            if k[2] % 128 and tile[5] == 2:
                prescaled = "f16" in name and M >= 96 and tile[0] >= 2 and (M <= 128 or k[2] <= k[1])
                if prescaled:
                    miss += 1
                    continue
                tile[5] = 1
            c = ",".join(map(str, tile))
            b = min(v.values())
            if c not in v:
                miss += 1
                continue
            tot += v[c]
            best += b
            if v[c] / b > worst[0]:
                worst = (v[c] / b, k[0])
        gtot += tot
        gbest += best
        print(f"{M:5d} {tot / best if best else float('nan'):.3f} {worst[0]:.2f}({worst[1][:24]}) {miss}")
    print(f"  all {gtot / gbest:.3f}")
