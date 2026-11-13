"""Speedup of one default dispatch over another, from bench.py runs of kernel "auto".

Used to show what a change round gained (e.g. the final branch vs the first WMMA branch). The baseline can be
given twice (an A/B/A run): its per-shape time is then the mean of both runs, and the plot shows the A1/A2
spread so noise is visible next to the gain.

Environment:
  CMP_A=a1.csv[,a2.csv]   baseline run(s)      CMP_B=b.csv   new run
  CMP_A_NAME, CMP_B_NAME  legend labels        CMP_KERNEL    kernel column to compare (default auto)
  CMP_OUT=plots/<arch>    CMP_TAG           output file suffix
  CMP_ARCH, CMP_CLOCKS    title text
Writes <out>/speedup_<tag>.png and <out>/speedup_<tag>.md.
"""

from collections import defaultdict
import csv
import math
import os
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator


def write_manifest(out_dir, name, inputs, **params):
    """Record the inputs (with SHA-256) and parameters behind a plot, so it can be audited/reproduced."""
    import hashlib
    import json
    import subprocess

    def sha(p):
        return hashlib.sha256(open(p, "rb").read()).hexdigest()

    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        rev = ""
    with open(f"{out_dir}/{name}.manifest.json", "w") as f:
        json.dump({"inputs": {p: sha(p) for p in inputs}, "git": rev, **params}, f, indent=1)

# Same palette and axis style as plots.py (which runs on import, so it is not imported).
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
COL = {"new": "#2a78d6", "dequant": "#1baf7a"}
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.facecolor": SURF,
        "figure.facecolor": SURF,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": INK,
    }
)


def logx(ax):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator([1, 2, 4, 8, 16, 32, 64, 128, 256, 512]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))

KERN = os.environ.get("CMP_KERNEL", "auto")
A_FILES = os.environ["CMP_A"].split(",")
B_FILE = os.environ["CMP_B"]
A_NAME = os.environ.get("CMP_A_NAME", "baseline")
B_NAME = os.environ.get("CMP_B_NAME", "new")
OUT = os.environ.get("CMP_OUT", "plots/gfx1100")
TAG = os.environ.get("CMP_TAG", "vs_baseline")
ARCH = os.environ.get("CMP_ARCH", "gfx1100")
CLOCKS = os.environ.get("CMP_CLOCKS", "default clocks")
os.makedirs(OUT, exist_ok=True)


def load(p):
    d = {}
    for r in csv.DictReader(open(p)):
        if r["kernel"] == KERN:
            d[(r["dtype"], r["shape"], int(r["M"]))] = float(r["us"])
    return d


As = [load(p) for p in A_FILES]
B = load(B_FILE)
keys = [k for k in B if all(k in a for a in As)]
A = {k: sum(a[k] for a in As) / len(As) for k in keys}
DTYPES = [d for d in ("fp16", "bf16", "fp32") if any(k[0] == d for k in keys)]
Ms = sorted({k[2] for k in keys})

rows = []
fig, axes = plt.subplots(1, len(DTYPES), figsize=(5.5 * len(DTYPES), 4), sharey=True, squeeze=False)
for ax, dt in zip(axes[0], DTYPES):
    xs, med, lo, hi, tot, noise = [], [], [], [], [], []
    for M in Ms:
        ks = [k for k in keys if k[0] == dt and k[2] == M]
        if not ks:
            continue
        rs = [A[k] / B[k] for k in ks]
        t = sum(A[k] for k in ks) / sum(B[k] for k in ks)
        n = sum(As[0][k] for k in ks) / sum(As[1][k] for k in ks) if len(As) > 1 else float("nan")
        xs.append(M)
        med.append(st.median(rs))
        lo.append(min(rs))
        hi.append(max(rs))
        tot.append(t)
        noise.append(n)
        wins = sum(r > 1 for r in rs)
        rows.append((dt, M, t, st.median(rs), min(rs), max(rs), wins, len(rs), n))
    ax.fill_between(xs, lo, hi, color=COL["new"], alpha=0.10, linewidth=0, label="min-max across shapes")
    ax.plot(xs, med, color=COL["new"], lw=2, marker="o", ms=6, mec=SURF, mew=2, label="median shape")
    ax.plot(xs, tot, color=COL["dequant"], lw=2, marker="^", ms=6, mec=SURF, mew=2, label="total time")
    if len(As) > 1:
        ax.plot(xs, noise, color=INK2, lw=1, ls=":", label=f"{A_NAME} run-to-run")
    ax.axhline(1, color=INK2, lw=1)
    logx(ax)
    ax.set_yscale("log", base=2)
    ax.yaxis.set_major_locator(FixedLocator([0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}x"))
    ax.set_xlabel("batch M (rows)")
    ax.set_title(dt, loc="left", fontsize=11, color=INK)
    ax.grid(True, color=GRID)
axes[0][0].set_ylabel("speedup (baseline time / new time)")
axes[0][0].legend(frameon=False, loc="upper left", fontsize=8)
fig.suptitle(
    f"{ARCH} 4-bit GEMM default dispatch: {B_NAME} vs {A_NAME}\n40 LLM shapes, weights streamed from DRAM; {CLOCKS}",
    fontsize=11,
    x=0.01,
    ha="left",
    color=INK,
)
fig.tight_layout()
fig.savefig(f"{OUT}/speedup_{TAG}.png", dpi=160)

with open(f"{OUT}/speedup_{TAG}.md", "w") as f:
    f.write(f"{ARCH}: {B_NAME} vs {A_NAME} ({KERN}), 40 shapes, {CLOCKS}\n\n")
    f.write("| dtype | M | total time | median shape | min | max | shapes faster | baseline run-to-run |\n")
    f.write("|---|---|---|---|---|---|---|---|\n")
    for dt, M, t, md, mn, mx, w, n, nz in rows:
        nzs = "" if math.isnan(nz) else f"{nz:.3f}"
        f.write(f"| {dt} | {M} | {t:.3f}x | {md:.3f}x | {mn:.2f}x | {mx:.2f}x | {w}/{n} | {nzs} |\n")
write_manifest(OUT, f"speedup_{TAG}", A_FILES + [B_FILE], kernel=KERN, baseline=A_NAME, new=B_NAME, arch=ARCH, clocks=CLOCKS)
print(f"wrote {OUT}/speedup_{TAG}.png and .md ({len(rows)} rows)")
