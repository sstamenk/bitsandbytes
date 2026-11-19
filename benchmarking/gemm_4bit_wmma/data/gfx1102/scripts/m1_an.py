"""m1_an.py dir: per dtype/M, per cycle total simt/wmma ratio, median-of-cycles totals, shapes SIMT faster."""

from collections import defaultdict
import csv
import glob
import statistics as st
import sys

t = defaultdict(lambda: defaultdict(list))  # (dt,M,shape) -> kernel -> [(cyc,us)]
for f in sorted(glob.glob(sys.argv[1] + "/*_c*.csv")):
    c = int(f.rsplit("_c", 1)[1].split(".")[0])
    for r in csv.DictReader(open(f)):
        t[(r["dtype"], int(r["M"]), r["shape"])][r["kernel"]].append((c, float(r["us"])))
for dt, M in sorted({(k[0], k[1]) for k in t}):
    ks = [k for k in t if k[0] == dt and k[1] == M]
    cyc = sorted({c for k in ks for c, _ in t[k]["simt"]})
    per = [
        sum(u for k in ks for cc, u in t[k]["simt"] if cc == c)
        / sum(u for k in ks for cc, u in t[k]["wmma"] if cc == c)
        for c in cyc
    ]
    med = lambda k, kn: st.median(u for _, u in t[k][kn])
    S = sum(med(k, "simt") for k in ks)
    Wm = sum(med(k, "wmma") for k in ks)
    o = sum(min(med(k, "simt"), med(k, "wmma")) for k in ks)
    nsimt = sum(med(k, "simt") < med(k, "wmma") for k in ks)
    print(
        f"{dt} M={M}: simt/wmma total {S / Wm:.3f} (cycles {min(per):.3f}-{max(per):.3f}, n={len(cyc)}), simt/orc {S / o:.3f} wmma/orc {Wm / o:.3f}, simt faster on {nsimt}/{len(ks)}"
    )
