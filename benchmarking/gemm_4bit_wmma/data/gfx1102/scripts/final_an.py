"""final_an.py: inherited vs final default dispatch against the per-shape oracle, and speedup vs main.
Inherited: eval/ (A build): auto, wmma (fp16, fp32), simt (M <= 64), dequant. Final: auto = mean of ab b1/b2 (fp16, bf16
from aba/, fp32 from aba3/); wmma = final auto for bf16/fp16 (WMMA at every M), fp32 forced WMMA from aba3 (M <= 64) and
the eval (A) above 24 (same tiles); simt and dequant from the eval (unchanged code paths).
Writes data/eval_final.csv. EXCL: shapes/M left out (first-in-process artifact)."""

from collections import defaultdict
import csv
import os
import sys

D = os.path.dirname(os.path.abspath(__file__)) + "/data"
EXCL = {("llama3-70b gate/up", 1), ("llama3-70b gate/up", 2)} if "--excl" in sys.argv else set()


def rd(f, kern=None):
    for r in csv.DictReader(open(f)):
        if kern:
            r["kernel"] = kern
        yield r


ev = defaultdict(dict)
fi = defaultdict(dict)
meta = {}
for dt in ("fp16", "bf16", "fp32"):
    for k in ("auto", "wmma", "simt", "dequant"):
        p = f"{D}/eval/{dt}_{k}.csv"
        if os.path.exists(p):
            for r in rd(p):
                ev[(dt, int(r["M"]), r["shape"])][k] = float(r["us"])
                meta[r["shape"]] = (r["N"], r["K"])
    ab_dir = "aba3" if dt == "fp32" else "aba"
    b = defaultdict(list)
    for run in ("b1", "b2"):
        for r in rd(f"{D}/{ab_dir}/{dt}_{run}.csv"):
            b[(dt, int(r["M"]), r["shape"])].append(float(r["us"]))
    for k, v in b.items():
        fi[k]["auto"] = sum(v) / len(v)
        if dt != "fp32":
            fi[k]["wmma"] = fi[k]["auto"]
for r in rd(f"{D}/aba3/fp32_wmma_b.csv"):
    fi[("fp32", int(r["M"]), r["shape"])]["wmma"] = float(r["us"])
for k, v in ev.items():
    if k not in fi:
        continue
    for kk in ("simt", "dequant"):
        if kk in v:
            fi[k][kk] = v[kk]
    if k[0] == "fp32" and "wmma" not in fi[k] and k[1] > 24 and "wmma" in v:
        fi[k]["wmma"] = v["wmma"]
with open(f"{D}/eval_final.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["shape", "N", "K", "M", "dtype", "kernel", "us"])
    for (dt, M, s), v in sorted(fi.items()):
        for kk, us in v.items():
            w.writerow([s, *meta[s], M, dt, kk, us])
CAP_OLD = {"fp16": 384, "bf16": 384, "fp32": 128}
CAP_NEW = 512


def orc(v):
    return min(v[k] for k in ("simt", "wmma", "dequant", "auto") if k in v)


for dt in ("fp16", "bf16", "fp32"):
    print(f"== {dt}: M | inherited new/orc (worst) | final new/orc (worst) | final/inherited | main/final | n")
    for M in sorted({k[1] for k in fi if k[0] == dt}):
        ks = [k for k in fi if k[0] == dt and k[1] == M and (k[2], M) not in EXCL and k in ev and "dequant" in fi[k]]
        if not ks:
            continue
        inherited = lambda k: (ev[k]["auto"] if M <= CAP_OLD[dt] else ev[k]["dequant"])
        new = lambda k: (fi[k]["auto"] if M <= CAP_NEW else fi[k]["dequant"])
        oi = sum(orc({**ev[k], **({"wmma": ev[k]["auto"]} if dt != "fp32" else {})}) for k in ks)
        of = sum(orc(fi[k]) for k in ks)
        si = sum(inherited(k) for k in ks)
        sf = sum(new(k) for k in ks)
        wi = max(inherited(k) / orc(ev[k]) for k in ks)
        wf = max(new(k) / orc(fi[k]) for k in ks)
        main = sum((ev[k]["simt"] if (M <= 8 and "simt" in ev[k]) else ev[k]["dequant"]) for k in ks)
        print(
            f"{M:4} | {si / oi:6.3f} ({wi:4.2f}) | {sf / of:6.3f} ({wf:4.2f}) | {sf / si:6.3f} | {main / sf:5.2f} | {len(ks)}"
        )
