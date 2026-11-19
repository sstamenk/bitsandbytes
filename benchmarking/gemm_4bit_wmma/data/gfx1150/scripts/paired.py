"""paired.py: interleaved in-process comparison of two dispatch tables (or fused vs dequant), LESSONS.md protocol.

Needs a -DBNB_WMMA_DEV=ON build: each variant is selected per call through BNB_GEMM4BIT_KERNEL (simt|wmma) and
BNB_WMMA_CFG (the tile; the launcher applies the same KCH lowering and fp16 prescale decision as the default dispatch).
For every (shape, M) both variants are timed in --cycles cycles, alternating which variant runs first, with
bench.timed_rotation (weights rotated from DRAM); the result per variant is the median over the cycles. (shape, M) whose
two variants pick the same kernel and tile are timed once per cycle and count for both.

usage: python paired.py --mode tables --old old_rules.cuh --new new_rules.cuh --dtype fp16 --ms 8,16 --out x.csv
       python paired.py --mode cap --dtype bf16 --ms 384,512 --out y.csv
"""

import argparse
import csv
import math
import os
import re
import statistics
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))  # benchmarking/gemm_4bit_wmma
sys.path.insert(0, HERE)
from bench import timed_rotation  # noqa: E402
import rules_eval_lib as lib  # noqa: E402
from shapes import SHAPES  # noqa: E402
from sweep_table import ROWS, TABLE, smem_ok  # noqa: E402

from bitsandbytes.backends.cuda.ops import _dequant_linear_fallback, _gemm_4bit_kernel_impl  # noqa: E402
import bitsandbytes.functional as F  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["tables", "cap"], default="tables")
p.add_argument("--old")
p.add_argument("--new")
p.add_argument("--target-old", default="gfx115")
p.add_argument("--target-new", default="gfx1150")
p.add_argument("--dtype", default="fp16")
p.add_argument("--ms", default="8,16")
p.add_argument("--shapes", default="")
p.add_argument("--cycles", type=int, default=5)
p.add_argument("--out", required=True)
args = p.parse_args()
dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
fp32 = args.dtype == "fp32"
INF = 1 << 30


def cutoffs(src, target):
    m = re.search(rf"wmma_fp32_cutoffs_{target}\[\] = \{{(.*?)\}};", src)
    if not m:  # gfx1150 shares the gfx115x cutoff table
        m = re.search(r"wmma_fp32_cutoffs_gfx115\[\] = \{(.*?)\};", src)
    return [
        (INF if a == "INT_MAX" else int(a), INF if b == "INT_MAX" else int(b))
        for a, b in re.findall(r"\{(\w+), (\w+)\}", m.group(1))
    ]


def choice(src, target, M, N, K):
    """(kernel, cfg) the default dispatch of `target` in rules file `src` picks."""
    if fp32:
        for mx, mn in cutoffs(src, target):
            if M <= mx:
                if N < mn:
                    return ("simt", "")
                break
    tile = lib.pick(lib.parse(src, f"wmma_rules_{target}_{'f32' if fp32 else 'f16'}"), M, N, K)
    planes = 3 if fp32 else 1
    wm, wn, wmp, wnp, ks, kch = tile
    assert (wm, wn, wmp, wnp, ks) in TABLE and smem_ok(wm, wn, wmp, wnp, ks, kch, planes), f"tile {tile} not forcible"
    return ("wmma", ",".join(map(str, tile)))


def set_variant(v):
    kern, cfg = v
    if kern == "dequant":
        return
    os.environ["BNB_GEMM4BIT_KERNEL"] = kern
    if cfg:
        os.environ["BNB_WMMA_CFG"] = cfg
    else:
        os.environ.pop("BNB_WMMA_CFG", None)


assert ROWS == 8
src_old = open(args.old).read() if args.mode == "tables" else ""
src_new = open(args.new).read() if args.mode == "tables" else ""
torch.manual_seed(0)
shapes = SHAPES if not args.shapes else [SHAPES[int(i)] for i in args.shapes.split(",")]
Ms = [int(m) for m in args.ms.split(",")]
out = open(args.out, "w", newline="")
w = csv.DictWriter(
    out, fieldnames=["shape", "N", "K", "M", "dtype", "cycle", "order", "variant", "kernel", "cfg", "us"]
)
w.writeheader()
for name, N, K in shapes:
    W = torch.randn(N, K, device="cuda", dtype=dt)
    q, s = F.quantize_4bit(W, blocksize=64, quant_type="nf4", compress_statistics=False)
    del W
    wbytes = q.numel() + s.absmax.numel() * 4
    ncopies = max(2, math.ceil(384e6 / wbytes))
    copies = [(q.clone().view(-1, 1), s.absmax.clone()) for _ in range(ncopies)]
    reps = min(max(20, 2 * ncopies // 3), 120)
    for M in Ms:
        A = torch.randn(M, K, device="cuda", dtype=dt)
        if args.mode == "tables":
            va, vb = choice(src_old, args.target_old, M, N, K), choice(src_new, args.target_new, M, N, K)
        else:
            va, vb = ("auto", ""), ("dequant", "")
        fused = [(lambda b=b, a=a: _gemm_4bit_kernel_impl(A, b, (N, K), a, 64, "nf4")) for b, a in copies]
        deq = [(lambda b=b, a=a: _dequant_linear_fallback(A, b, (N, K), a, 64, "nf4")) for b, a in copies]
        res = {"old": [], "new": []}
        for c in range(args.cycles):
            order = ["old", "new"] if c % 2 == 0 else ["new", "old"]
            for which in order:
                v = va if which == "old" else vb
                if which == "new" and vb == va and args.mode == "tables":
                    res["new"].append(res["old"][-1] if order[0] == "old" else None)
                    continue
                if args.mode == "cap":
                    for k in ("BNB_GEMM4BIT_KERNEL", "BNB_WMMA_CFG"):
                        os.environ.pop(k, None)
                    fns = fused if which == "old" else deq
                else:
                    set_variant(v)
                    fns = fused
                for f in fns[:2]:
                    f()
                us = timed_rotation(fns, reps)
                if us < 4:
                    sys.exit(f"suspicious {us} us for {name} M={M} {v}")
                res[which].append(us)
                if which == "old" and vb == va and args.mode == "tables" and order[0] == "new":
                    res["new"][-1] = us
            for which in ("old", "new"):
                v = va if which == "old" else vb
                w.writerow(
                    {
                        "shape": name,
                        "N": N,
                        "K": K,
                        "M": M,
                        "dtype": args.dtype,
                        "cycle": c,
                        "order": "".join(o[0] for o in order),
                        "variant": which,
                        "kernel": v[0],
                        "cfg": v[1],
                        "us": res[which][-1],
                    }
                )
        out.flush()
        print(
            f"{name[:28]:28} M={M:4} old {statistics.median(res['old']):9.1f} new {statistics.median(res['new']):9.1f} {va} {vb}",
            flush=True,
        )
    del copies
    torch.cuda.empty_cache()
out.close()
