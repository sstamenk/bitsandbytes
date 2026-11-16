"""Speedup of the final default dispatch over upstream main, on all four GPUs.

main dispatches SIMT for M <= 8 and dequant + F.linear above. The new dispatch is the "auto" kernel up to each GPU's
Python M limit (_gemm_4bit_use_custom_rocm) and dequant + F.linear above it. Inputs are the eval_final runs; the
gfx1201 fp32 baseline uses hipBLASLt with the ROCm/rocm-libraries#12465 fix (see README, Known issues).
Writes plots/speedup_vs_main.png and plots/speedup_vs_main.md.
"""

from collections import defaultdict
import csv
import os
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "plots")
RUNS = [
    ("Radeon Pro W7900", "gfx1100", f"{D}/gfx1100/pinned/eval_final.csv"),
    ("Radeon AI PRO R9700", "gfx1201", f"{D}/gfx1201/pinned/eval_final_fixedblas.csv"),
    ("Radeon 8060S", "gfx1151", f"{D}/gfx1151/pinned/dense/eval_final.csv"),
    ("Radeon 760M", "gfx1103", f"{D}/gfx1103/pinned/eval_final.csv"),
]
# Largest M on the fused path: (bf16/fp16, fp32).
CAP = {"gfx1100": (384, 128), "gfx1201": (384, 512), "gfx1151": (512, 512), "gfx1103": (384, 384)}
DTYPES = ("fp16", "bf16", "fp32")
TABLE_MS = (1, 4, 8, 16, 32, 64, 128, 256, 384)

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
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


def load(path):
    t = defaultdict(dict)  # (dtype, shape, M) -> kernel -> us
    for row in csv.DictReader(open(path)):
        t[(row["dtype"], row["shape"], int(row["M"]))][row["kernel"]] = float(row["us"])
    return t


def style_axes(ax):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator([1, 2, 4, 8, 16, 32, 64, 128, 256, 512]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.set_yscale("log", base=2)
    ax.yaxis.set_major_locator(FixedLocator([0.5, 1, 2, 4, 8]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}x"))
    ax.set_ylim(0.6, 16)
    ax.grid(True, color=GRID)
    ax.axhline(1, color=INK2, lw=1)


fig, axes = plt.subplots(len(RUNS), 3, figsize=(15, 3.1 * len(RUNS)), sharey=True)
md = [
    "| GPU | dtype | " + " | ".join(f"M={m}" for m in TABLE_MS) + " |",
    "|---|---|" + "---|" * len(TABLE_MS),
]
for r, (gpu, arch, path) in enumerate(RUNS):
    t = load(path)
    for c, dt in enumerate(DTYPES):
        cap = CAP[arch][dt == "fp32"]
        xs, med, lo, hi, tot = [], [], [], [], []
        total_by_m = {}
        for M in sorted({k[2] for k in t if k[0] == dt}):
            prev_k = "simt" if M <= 8 else "dequant"
            new_k = "auto" if M <= cap else "dequant"
            ks = [k for k in t if k[0] == dt and k[2] == M and {prev_k, new_k} <= set(t[k])]
            if len(ks) < 40:
                continue
            ratios = [t[k][prev_k] / t[k][new_k] for k in ks]
            xs.append(M)
            med.append(st.median(ratios))
            lo.append(min(ratios))
            hi.append(max(ratios))
            tot.append(sum(t[k][prev_k] for k in ks) / sum(t[k][new_k] for k in ks))
            total_by_m[M] = tot[-1]
        ax = axes[r][c]
        ax.fill_between(xs, lo, hi, color="#2a78d6", alpha=0.10, lw=0, label="min-max across shapes")
        ax.plot(xs, med, color="#2a78d6", lw=2, marker="o", ms=4, mec=SURF, mew=1, label="median shape")
        ax.plot(xs, tot, color="#1baf7a", lw=2, marker="^", ms=4, mec=SURF, mew=1, label="total time, 40 shapes")
        style_axes(ax)
        ax.set_title(f"{gpu} ({arch})  {dt}", loc="left", fontsize=10)
        if r == len(RUNS) - 1:
            ax.set_xlabel("batch M (rows)")
        if c == 0:
            ax.set_ylabel("speedup vs main")
        cells = [f"{total_by_m[m]:.2f}x" if m in total_by_m else "-" for m in TABLE_MS]
        md.append(f"| {gpu} ({arch}) | {dt} | " + " | ".join(cells) + " |")
axes[0][0].legend(frameon=False, loc="upper left", fontsize=8)
fig.suptitle(
    "4-bit GEMM (NF4, blocksize 64): this PR vs main (SIMT for M<=8, dequant + F.linear above)\n"
    "40 LLM linear-layer shapes, weights streamed from DRAM, pinned clocks. gfx1201 fp32 baseline uses a hipBLASLt "
    "with its fp32 slow path fixed;\ngfx1151 fp32: a few shapes hit a slow fp32 BLAS path (up to ~50x, clipped)",
    x=0.01,
    ha="left",
    fontsize=11,
)
fig.tight_layout()
fig.savefig(f"{OUT}/speedup_vs_main.png", dpi=130)
with open(f"{OUT}/speedup_vs_main.md", "w") as f:
    f.write("Speedup in total time over the 40 shapes, this branch vs main (see plots_vs_main.py)\n\n")
    f.write("\n".join(md) + "\n")
print("\n".join(md))
