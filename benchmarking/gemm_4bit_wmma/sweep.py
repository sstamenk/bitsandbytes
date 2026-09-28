"""In-process WMMA config sweep: best (wm,wn,warps_n,ksplit) per (shape, M).
Requires a -DBNB_WMMA_DEV=ON build and BNB_GEMM4BIT_KERNEL=wmma; BNB_WMMA_CFG is re-read by the launcher per call."""

import argparse
import csv
import math
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from bench import timed_rotation
from shapes import SHAPES

from bitsandbytes.backends.cuda.ops import _gemm_4bit_kernel_impl
import bitsandbytes.functional as F

p = argparse.ArgumentParser()
p.add_argument("--shapes", default="0,2,8,9,10,11,15,19,23,24,28,35")
p.add_argument("--ms", default="1,8,16,32,64,128,256")
p.add_argument("--dtype", default="fp16")
p.add_argument("--cfgs", default="auto")
p.add_argument("--out", default="sweep.csv")
p.add_argument("--reps", type=int, default=20)
args = p.parse_args()
dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
PLANES = 3 if args.dtype == "fp32" else 1

# Candidate tiles = the BNB_WMMA_DEV_TRY table compiled into the dev launcher.
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../csrc/gemm_4bit_wmma.cu")).read()
TABLE = sorted(
    {tuple(map(int, t)) for t in re.findall(r"BNB_WMMA_DEV_TRY\((\d+), (\d+), (\d+), (\d+), (\d+)\)", _src)}
)


def smem_ok(wm, wn, wmp, wnp, ks, kch):
    mt, nt = wmp * 8 * wm, wnp * 16 * wn
    return max(2 * PLANES * ks * mt * (64 * kch + 8) * 2, ks * mt * nt * 4 if ks > 1 else 0) <= 48 * 1024


def cands(M, K):
    if args.cfgs != "auto":
        return [tuple(map(int, c.split(","))) for c in args.cfgs.split(";")]
    out = []
    for wm, wn, wmp, wnp, ks in TABLE:
        mt = wmp * 8 * wm
        if mt > max(8, 2 * M) or (M >= 64 and mt < 32):  # skip mostly-empty or tiny M tiles
            continue
        for kch in (1, 2):
            if K % (64 * kch) == 0 and smem_ok(wm, wn, wmp, wnp, ks, kch):
                out.append((wm, wn, wmp, wnp, ks, kch))
    return out


rows = []
for si in map(int, args.shapes.split(",")):
    name, N, K = SHAPES[si]
    W = torch.randn(N, K, device="cuda", dtype=dt)
    q, s = F.quantize_4bit(W, blocksize=64, quant_type="nf4", compress_statistics=False)
    del W
    wbytes = q.numel() + s.absmax.numel() * 4
    ncopies = max(2, math.ceil(384e6 / wbytes))
    copies = [(q.clone().view(-1, 1), s.absmax.clone()) for _ in range(ncopies)]
    for M in map(int, args.ms.split(",")):
        A = torch.randn(M, K, device="cuda", dtype=dt)
        fns = [(lambda b=b, a=a: _gemm_4bit_kernel_impl(A, b, (N, K), a, 64, "nf4")) for b, a in copies]
        res = []
        for c in cands(M, K):
            os.environ["BNB_WMMA_CFG"] = ",".join(map(str, c))
            for f in fns[:2]:
                f()
            us = timed_rotation(fns, min(max(args.reps, 2 * ncopies // 3), 120))
            res.append((us, c))
            rows.append({"shape": name, "N": N, "K": K, "M": M, "cfg": ",".join(map(str, c)), "us": us})
        res.sort()
        print(
            f"{name[:30]:30} M={M:4} best {res[0][0]:8.1f} {res[0][1]}  2nd {res[1][0]:8.1f} {res[1][1]}  worst {res[-1][0]:8.1f}",
            flush=True,
        )
    del copies
    torch.cuda.empty_cache()
with open(args.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
