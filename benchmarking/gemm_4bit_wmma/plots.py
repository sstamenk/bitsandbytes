from collections import defaultdict
import csv
import os
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from shapes import SHAPES

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
COL = {"wmma": "#2a78d6", "simt": "#eb6834", "dequant": "#1baf7a", "new": "#2a78d6"}
MARK = {"wmma": "o", "simt": "s", "dequant": "^"}
NAME = {"wmma": "WMMA (this PR)", "simt": "SIMT", "dequant": "dequant+BLAS"}
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

t = defaultdict(dict)
FILES = os.environ.get("PLOT_FILES", "data/gfx1100/eval_wmma_dequant.csv,data/gfx1100/eval_simt.csv").split(",")
SUFFIX = os.environ.get("PLOT_SUFFIX", "")
OUT = os.environ.get("PLOT_OUT", "plots/gfx1100")
os.makedirs(OUT, exist_ok=True)
CLOCKS = os.environ.get("PLOT_CLOCKS", "default clocks")
ARCH = os.environ.get("PLOT_ARCH", "gfx1100")
for f in FILES:
    for r in csv.DictReader(open(f)):
        t[(r["dtype"], r["shape"], int(r["M"]))][r["kernel"]] = float(r["us"])
Ms = sorted({k[2] for k in t})
DTYPES = [d for d in ("fp16", "bf16", "fp32") if any(k[0] == d for k in t)]


def old_default(v, M):  # previous gfx1100 and gfx1201 dispatch: SIMT for M<=8, dequant+F.linear above
    return v["simt"] if M <= 8 else v["dequant"]


# New dispatch per dtype: SIMT through simt_max_m, WMMA through fused_max_m, dequant+F.linear above.
# PLOT_NEW="fp16:1:256,bf16:1:256,fp32:5:128"; the default is the gfx1100 bf16/fp16 dispatch.
NEW_SIMT_MAX_M = int(os.environ.get("PLOT_NEW_SIMT_MAX_M", "0"))
NEW = {d: (NEW_SIMT_MAX_M, 256) for d in ("fp16", "bf16", "fp32")}
for spec in filter(None, os.environ.get("PLOT_NEW", "").split(",")):
    d, simt_max, fused_max = spec.split(":")
    NEW[d] = (int(simt_max), int(fused_max))


def new_default(v, M, dt):
    # A measured "auto" (default C dispatch, bench.py kernel auto) replaces the M rule where present.
    simt_max, fused_max = NEW[dt]
    if M > fused_max:
        return v["dequant"]
    if "auto" in v:
        return v["auto"]
    return v["simt"] if M <= simt_max else v["wmma"]


def logx(ax):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator([1, 2, 4, 8, 16, 32, 64, 128, 256, 512]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))


# 1) Headline: new vs previous default dispatch
fig, axes = plt.subplots(1, len(DTYPES), figsize=(5.5 * len(DTYPES), 4), sharey=len(DTYPES) < 3)
table = []


def speedups(dt, M):
    """Previous/new dispatch per shape, or [] when an M lacks the kernels both need (e.g. auto-only runs)."""
    try:
        return [old_default(v, M) / new_default(v, M, dt) for (d, s, m), v in t.items() if d == dt and m == M]
    except KeyError:
        return []


for ax, dt in zip(axes, DTYPES):
    xs, med, lo, hi = [], [], [], []
    for M in Ms:
        rs = speedups(dt, M)
        if not rs:
            continue
        xs.append(M)
        med.append(st.median(rs))
        lo.append(min(rs))
        hi.append(max(rs))
        table.append((dt, M, st.median(rs), min(rs), max(rs)))
    ax.fill_between(xs, lo, hi, color=COL["new"], alpha=0.10, linewidth=0)
    ax.plot(xs, med, color=COL["new"], lw=2, marker="o", ms=6, mec=SURF, mew=2, solid_capstyle="round")
    ax.axhline(1, color=INK2, lw=1)
    logx(ax)
    ax.set_yscale("log", base=2)
    ax.yaxis.set_major_locator(FixedLocator([0.5, 1, 2, 4, 8, 16, 64, 256, 1024]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}x"))
    ax.set_xlabel("batch M (rows)")
    # Optional per-dtype caveat, e.g. PLOT_NOTE_fp32="previous dequant+BLAS uses a slow fp32 GEMM".
    note = os.environ.get(f"PLOT_NOTE_{dt}", "")
    ax.set_title(dt + (f"  ({note})" if note else ""), loc="left", fontsize=11 if not note else 9, color=INK)
    i = xs.index(16)
    ax.annotate(
        f"{med[i]:.1f}x at M=16", (16, med[i]), xytext=(6, 8), textcoords="offset points", color=INK, fontsize=9
    )
    i = xs.index(128)
    ax.annotate(
        f"{med[i]:.2f}x at M=128", (128, med[i]), xytext=(8, 12), textcoords="offset points", color=INK, fontsize=9
    )
axes[0].set_ylabel("speedup vs previous dispatch")
fig.suptitle(
    f"{ARCH} 4-bit GEMM: new default dispatch vs previous (SIMT M≤8, dequant+BLAS above)\n"
    f"median over 40 LLM shapes; band = min-max across shapes; {CLOCKS}",
    fontsize=11,
    x=0.01,
    ha="left",
    color=INK,
)
fig.tight_layout()
fig.savefig(f"{OUT}/speedup_vs_previous_dispatch{SUFFIX}.png", dpi=160)

# 2) Median latency per kernel
fig, axes = plt.subplots(1, len(DTYPES), figsize=(5.5 * len(DTYPES), 4.2), sharey=True)
for ax, dt in zip(axes, DTYPES):
    for k in ("simt", "wmma", "dequant"):
        xs = [M for M in Ms if any(k in v for (d, s, m), v in t.items() if d == dt and m == M)]
        ys = [st.median([v[k] for (d, s, m), v in t.items() if d == dt and m == M and k in v]) for M in xs]
        ax.plot(xs, ys, color=COL[k], lw=2, marker=MARK[k], ms=6, mec=SURF, mew=2, label=NAME[k])
        if k == "simt":
            ax.annotate(
                NAME[k],
                (xs[-1], ys[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
                color=INK2,
            )
        else:
            i = xs.index(4)
            ax.annotate(
                NAME[k],
                (4, ys[i]),
                xytext=(0, 9 if k == "dequant" else -9),
                textcoords="offset points",
                ha="center",
                va="bottom" if k == "dequant" else "top",
                fontsize=9,
                color=INK2,
            )
    ax.axvline(NEW[dt][1], color=INK2, lw=1)
    ax.annotate(
        f"default dispatch\nuses dequant+BLAS\nabove M={NEW[dt][1]} →",
        (NEW[dt][1], 0.03),
        xycoords=("data", "axes fraction"),
        xytext=(-4, 0),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=8,
        color=INK2,
    )
    logx(ax)
    ax.set_yscale("log")
    ax.set_xlabel("batch M (rows)")
    ax.set_title(dt, loc="left", fontsize=11, color=INK)
    lo = min(min(ln.get_ydata()) for ln in ax.get_lines() if len(ln.get_ydata()) > 2)
    hi = max(max(ln.get_ydata()) for ln in ax.get_lines() if len(ln.get_ydata()) > 2)
    ax.set_xlim(0.8, 900)
    ax.set_ylim(lo * 0.72, hi * 1.4)
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
axes[0].set_ylabel("median latency (µs, log)")
axes[0].legend(frameon=False, loc="upper left")
fig.suptitle(
    f"{ARCH} 4-bit GEMM kernels, median latency over 40 LLM shapes (weights streamed from DRAM), {CLOCKS}",
    fontsize=11,
    x=0.01,
    ha="left",
    color=INK,
)
fig.tight_layout()
fig.savefig(f"{OUT}/median_latency{SUFFIX}.png", dpi=160)

# 3) Per-shape grids
for dt in DTYPES:
    fig, axes = plt.subplots(5, 8, figsize=(24, 14))
    for ax, (name, N, K) in zip(axes.flat, SHAPES):
        for k in ("simt", "wmma", "dequant"):
            xs = [M for M in Ms if k in t[(dt, name, M)]]
            ax.plot(xs, [t[(dt, name, M)][k] for M in xs], color=COL[k], lw=1.6, marker=MARK[k], ms=3.5)
        logx(ax)
        ax.set_yscale("log")
        ax.tick_params(labelsize=7)
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1, 2, 5)))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
        ax.set_title(f"{name}\nN={N} K={K}", fontsize=8, color=INK)
    handles = [
        plt.Line2D([], [], color=COL[k], marker=MARK[k], lw=2, label=NAME[k]) for k in ("simt", "wmma", "dequant")
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.supxlabel("batch M")
    fig.supylabel("latency (µs, log)", x=0.003)
    fig.suptitle(f"{ARCH} {dt}: per-shape latency, {CLOCKS}", x=0.01, ha="left", fontsize=13)
    fig.tight_layout(rect=(0.012, 0, 1, 0.975))
    fig.savefig(f"{OUT}/per_shape_{dt}{SUFFIX}.png", dpi=110)

with open(f"{OUT}/speedup_table{SUFFIX}.md", "w") as f:
    f.write("| M | " + " | ".join(f"{dt} median (min-max)" for dt in DTYPES) + " |\n")
    f.write("|---" * (len(DTYPES) + 1) + "|\n")
    rows = defaultdict(dict)
    for dt, M, m, lo, hi in table:
        rows[M][dt] = f"{m:.2f}x ({lo:.2f}-{hi:.2f})"
    for M in Ms:
        if rows[M]:
            f.write(f"| {M} | " + " | ".join(rows[M].get(dt, "-") for dt in DTYPES) + " |\n")
print(open(f"{OUT}/speedup_table{SUFFIX}.md").read())
