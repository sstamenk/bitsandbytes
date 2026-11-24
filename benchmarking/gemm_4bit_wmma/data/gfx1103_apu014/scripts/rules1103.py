"""Rule tables in python form (alt.py / rules_eval.py format) parsed from csrc/gemm_4bit_wmma_rules.cuh, plus pick().
load(name): 'g1100_f16' etc. (any wmma_rules_gfxNNNN_{f16,f32} in the header) or a .py file defining RULES.
pick(R, M, N, K): tile string; kch is KCH_MAX (not lowered for K % 128 != 0; sweep1103.py has those candidates)."""

import os
import re

INF = 1 << 30


def T(*a):
    return ",".join(map(str, a))


HDR = os.environ.get("RULES_HDR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemm_4bit_wmma_rules.cuh"))


def parse(hdr=HDR):
    src = open(hdr).read()
    out = {}
    for m in re.finditer(r"wmma_rules_gfx(\d+)_(f16|f32)\[\] = \{(.*?)\n\};", src, re.S):
        rules = []
        for line in m.group(3).splitlines():
            line = line.split("//")[0].strip()
            if not line.startswith("{"):
                continue
            mx = line[1:].split(",")[0].strip()
            mx = INF if mx == "INT_MAX" else int(mx)
            leaves = []
            for f, a, b, t in re.findall(r"(n_lt|k_lt|k_over_n_lt)\((\d+),\s*(?:(\d+),\s*)?\{([\d,\s]+)\}\)", line):
                tile = T(*map(int, t.split(",")))
                if f == "n_lt":
                    leaves.append(("N", int(a), 0, tile))
                elif f == "k_lt":
                    leaves.append(("K", int(a), 0, tile))
                else:
                    leaves.append(("KN", int(a), int(b), tile))
            oth = re.findall(r"\{([\d,\s]+)\}\},?\s*$", line)
            rules.append((mx, leaves, T(*map(int, oth[-1].split(",")))))
        out[f"g{m.group(1)}_{m.group(2)}"] = rules
    return out


def load(x):
    if x.endswith(".py"):
        ns = {"T": T, "INF": INF}
        exec(open(x).read(), ns)
        return ns["RULES"]
    return parse()[x]


def pick(R, M, N, K):
    for mx, leaves, oth in R:
        if M <= mx:
            for f, lt, den, c in leaves:
                if (f == "N" and N < lt) or (f == "K" and K < lt) or (f == "KN" and K * den < lt * N):
                    return c
            return oth
    return R[-1][2]


def dump(R):
    s = "RULES = [\n"
    for mx, leaves, oth in R:
        lv = ", ".join(f'("{f}", {lt}, {den}, T({c}))' for f, lt, den, c in leaves)
        s += f"    ({'INF' if mx == INF else mx}, [{lv}], T({oth})),\n"
    return s + "]\n"


if __name__ == "__main__":
    for k, v in parse().items():
        print(k, len(v), v[0])
