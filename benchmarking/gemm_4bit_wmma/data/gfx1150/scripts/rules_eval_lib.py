import re


def parse(src, name):
    blk = src[src.index(name + "[] = {") :]
    blk = blk[: blk.index("};")]
    rules = []
    for line in blk.splitlines():
        m = re.match(r"\s*\{(\d+|INT_MAX),\s*\{(.*)\},\s*\{([\d, ]+)\}\},", line)
        if not m:
            continue
        mx = 1 << 30 if m.group(1) == "INT_MAX" else int(m.group(1))
        leaves = [
            (f, [int(x) for x in a.split(",") if x.strip()], tuple(int(x) for x in t.split(",")))
            for f, a, t in re.findall(r"(n_lt|k_lt|k_over_n_lt)\(([\d, ]+?),\s*\{([\d, ]+)\}\)", m.group(2))
        ]
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
