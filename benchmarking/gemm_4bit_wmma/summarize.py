from collections import defaultdict
import csv
import statistics as st
import sys

rows = list(csv.DictReader(open(sys.argv[1])))
t = defaultdict(dict)
for r in rows:
    t[(r["dtype"], r["shape"], int(r["M"]))][r["kernel"]] = float(r["us"])
for dtype in sorted({k[0] for k in t}):
    Ms = sorted({k[2] for k in t if k[0] == dtype})
    kernels = sorted({kk for k, v in t.items() if k[0] == dtype for kk in v})
    print(f"== {dtype}: median over shapes (us), and #shapes where kernel is fastest")
    print(
        "   M "
        + "".join(f"{k:>10}" for k in kernels)
        + "  | best: "
        + " ".join(f"{k:>7}" for k in kernels)
        + " | wmma/simt  wmma/deq"
    )
    for M in Ms:
        cells = {k: [v[k] for kk, v in t.items() if kk[0] == dtype and kk[2] == M and k in v] for k in kernels}
        best = defaultdict(int)
        ratios_s, ratios_d = [], []
        for kk, v in t.items():
            if kk[0] != dtype or kk[2] != M:
                continue
            best[min(v, key=v.get)] += 1
            if "wmma" in v and "simt" in v:
                ratios_s.append(v["simt"] / v["wmma"])
            if "wmma" in v and "dequant" in v:
                ratios_d.append(v["dequant"] / v["wmma"])
        rs = f"{st.median(ratios_s):8.2f}x" if ratios_s else ""
        rd = f"{st.median(ratios_d):8.2f}x" if ratios_d else ""
        print(
            f"{M:>4} "
            + "".join(f"{st.median(cells[k]):10.1f}" for k in kernels)
            + "  | best: "
            + " ".join(f"{best[k]:>7}" for k in kernels)
            + f" | {rs} {rd}"
        )
