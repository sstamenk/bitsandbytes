"""Kernel latency and achieved bandwidth vs the theoretical memory-bandwidth limit.

Plots SIMT (the GEMV-style kernel), WMMA and dequant+BLAS from bench.py CSVs, with the time it takes to move the
GEMM's bytes at the GPU's peak DRAM bandwidth: packed 4-bit weights + fp32 absmax (blocksize 64, not nested) +
activations [M, K] + output [M, N]. That line is a lower bound for a bandwidth-bound kernel; above it the
kernel is limited by something else (compute, latency, launch overhead). Rows of the same (dtype, shape, M,
kernel) in several files (e.g. A/B/A runs) are averaged.

Environment:
  PLOT_FILES=a.csv,b.csv   bench.py outputs (kernels simt / wmma / dequant; others ignored)
  PLOT_PEAK_GBS=864        theoretical peak DRAM bandwidth in GB/s
  PLOT_GPU="Radeon Pro W7900"  PLOT_ARCH=gfx1100  PLOT_CLOCKS="default clocks"
  PLOT_OUT=plots/gfx1100   PLOT_SUFFIX=""
Writes <out>/kernels_median<suffix>.png (median over the shapes: latency and achieved bandwidth, one column
per dtype) and <out>/kernels_per_shape_<dtype><suffix>.png (every shape).
"""

from collections import defaultdict
import csv
import os
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullLocator
from shapes import SHAPES

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
COL = {"simt": "#eb6834", "wmma": "#2a78d6", "dequant": "#1baf7a", "bound": "#0b0b0b"}
MARK = {"simt": "s", "wmma": "o", "dequant": "^"}
NAME = {"simt": "SIMT (GEMV-style)", "wmma": "WMMA", "dequant": "dequant + BLAS"}
KERNELS = ("simt", "wmma", "dequant")
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.facecolor": SURF,
        "figure.facecolor": SURF,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": INK,
    }
)

FILES = os.environ["PLOT_FILES"].split(",")
PEAK = float(os.environ["PLOT_PEAK_GBS"])
GPU = os.environ.get("PLOT_GPU", "")
ARCH = os.environ.get("PLOT_ARCH", "gfx1100")
CLOCKS = os.environ.get("PLOT_CLOCKS", "default clocks")
OUT = os.environ.get("PLOT_OUT", f"plots/{ARCH}")
SUFFIX = os.environ.get("PLOT_SUFFIX", "")
BLOCKSIZE = 64
ESIZE = {"fp16": 2, "bf16": 2, "fp32": 4}
os.makedirs(OUT, exist_ok=True)

acc = defaultdict(list)  # (dtype, shape, M, kernel) -> [us]
for f in FILES:
    for r in csv.DictReader(open(f)):
        if r["kernel"] in KERNELS:
            acc[(r["dtype"], r["shape"], int(r["M"]), r["kernel"])].append(float(r["us"]))
t = {k: sum(v) / len(v) for k, v in acc.items()}
NK = {name: (n, k) for name, n, k in SHAPES}
DTYPES = [d for d in ("fp16", "bf16", "fp32") if any(k[0] == d for k in t)]
Ms = sorted({k[2] for k in t})


def gemm_bytes(dt, shape, M):
    N, K = NK[shape]
    return N * K / 2 + N * K / BLOCKSIZE * 4 + M * K * ESIZE[dt] + M * N * ESIZE[dt]


def bound_us(dt, shape, M):
    return gemm_bytes(dt, shape, M) / (PEAK * 1e9) * 1e6


def logx(ax):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator([1, 2, 4, 8, 16, 32, 64, 128, 256, 512]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))


def logy(ax):
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))


shapes = [s for s, _, _ in SHAPES]
gpu_txt = f"{GPU} ({ARCH})" if GPU else ARCH

# 1) Median over the shapes: latency (top) and achieved bandwidth (bottom), one column per dtype.
fig, axes = plt.subplots(2, len(DTYPES), figsize=(5.6 * len(DTYPES), 8.2), squeeze=False)
for j, dt in enumerate(DTYPES):
    ax_t, ax_b = axes[0][j], axes[1][j]
    for k in KERNELS:
        xs = [M for M in Ms if all((dt, s, M, k) in t for s in shapes)]
        if not xs:
            continue
        lat = [st.median(t[(dt, s, M, k)] for s in shapes) for M in xs]
        bw = [st.median(gemm_bytes(dt, s, M) / (t[(dt, s, M, k)] * 1e-6) / 1e9 for s in shapes) for M in xs]
        style = dict(color=COL[k], lw=2, marker=MARK[k], ms=5, mec=SURF, mew=1.5, label=NAME[k])
        ax_t.plot(xs, lat, **style)
        ax_b.plot(xs, bw, **style)
    bound = [st.median(bound_us(dt, s, M) for s in shapes) for M in Ms]
    ax_t.plot(Ms, bound, color=COL["bound"], lw=1.5, ls="--", label=f"bandwidth limit ({PEAK:g} GB/s)")
    ax_b.axhline(PEAK, color=COL["bound"], lw=1.5, ls="--")
    ax_b.annotate(f"theoretical peak {PEAK:g} GB/s", (1, PEAK), xytext=(2, 4), textcoords="offset points",
                  fontsize=8, color=INK2)
    for ax in (ax_t, ax_b):
        logx(ax)
        ax.set_xlabel("batch M (rows)")
    logy(ax_t)
    ax_t.set_title(dt, loc="left", fontsize=11, color=INK)
    ax_b.set_ylim(0, PEAK * 1.12)
    if j == 0:
        ax_t.set_ylabel("median latency over the shapes (µs, log)")
        ax_b.set_ylabel("median achieved bandwidth (GB/s)")
        ax_t.legend(frameon=False, loc="upper left", fontsize=8)
fig.suptitle(
    f"{gpu_txt} 4-bit GEMM (NF4, blocksize 64): median over {len(shapes)} LLM shapes, weights streamed from DRAM; "
    f"{CLOCKS}\nbandwidth = (4-bit weights + absmax + A + C bytes) / time; the dashed line is that traffic at "
    f"the theoretical peak",
    fontsize=11,
    x=0.01,
    ha="left",
    color=INK,
)
fig.tight_layout()
fig.savefig(f"{OUT}/kernels_median{SUFFIX}.png", dpi=150)
plt.close(fig)

# 2) Every shape, one figure per dtype.
for dt in DTYPES:
    fig, axes = plt.subplots(5, 8, figsize=(26, 15))
    for ax, (s, N, K) in zip(axes.flat, SHAPES):
        for k in KERNELS:
            xs = [M for M in Ms if (dt, s, M, k) in t]
            if xs:
                ax.plot(xs, [t[(dt, s, M, k)] for M in xs], color=COL[k], lw=1.6, marker=MARK[k], ms=3.5,
                        mec=SURF, mew=1, label=NAME[k])
        ax.plot(Ms, [bound_us(dt, s, M) for M in Ms], color=COL["bound"], lw=1.2, ls="--",
                label=f"bandwidth limit ({PEAK:g} GB/s)")
        logx(ax)
        logy(ax)
        ax.tick_params(labelsize=7)
        ax.set_title(f"{s[:34]}\nN={N} K={K} ({N * K / 2 / 1e6:.1f} MB)", fontsize=8, loc="left", color=INK)
    axes.flat[0].legend(frameon=False, loc="upper left", fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("M", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("µs (log)", fontsize=8)
    fig.suptitle(
        f"{gpu_txt} 4-bit GEMM {dt}: latency per shape; dashed = bytes at the theoretical peak {PEAK:g} GB/s; "
        f"{CLOCKS}",
        fontsize=13,
        x=0.01,
        ha="left",
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(f"{OUT}/kernels_per_shape_{dt}{SUFFIX}.png", dpi=110)
    plt.close(fig)
print(f"wrote {OUT}/kernels_median{SUFFIX}.png and kernels_per_shape_<dtype>{SUFFIX}.png for {DTYPES}")
