"""alt.py: in-process interleaved comparison of WMMA tile tables (via BNB_WMMA_CFG, needs -DBNB_WMMA_DEV=ON) and
optionally dequant+F.linear, on the same weight copies and activations. For every (shape, M) the configurations
run in alternating order (forward on even cycles, reverse on odd), --cycles times; one CSV row per measurement.

usage: python alt.py --tables A=a.py,B=b.py,J2=j2.py [--dequant] --dtype fp16 --ms 24,96 --cycles 5 --out x.csv
Table files define RULES as in rules11.py; builtin tables g11_f16, g115_f16, g11_f32, g115_f32."""

import argparse
import csv
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import timed_rotation
from rules11 import load, pick
from shapes import SHAPES

from bitsandbytes.backends.cuda.ops import _dequant_linear_fallback, _gemm_4bit_kernel_impl
import bitsandbytes.functional as F

p = argparse.ArgumentParser()
p.add_argument("--tables", required=True)
p.add_argument("--dequant", action="store_true")
p.add_argument("--dtype", default="fp16")
p.add_argument("--ms", required=True)
p.add_argument("--shapes", default=",".join(map(str, range(40))))
p.add_argument("--cycles", type=int, default=5)
p.add_argument("--out", required=True)
a = p.parse_args()
tabs = [(n, load(f)) for n, f in (s.split("=") for s in a.tables.split(","))]
dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]
os.environ["BNB_GEMM4BIT_KERNEL"] = "wmma"
torch.manual_seed(0)
out = open(a.out, "w", newline="")
w = csv.DictWriter(out, fieldnames=["shape", "N", "K", "M", "dtype", "cfg", "tile", "cycle", "us"])
w.writeheader()
for si in map(int, a.shapes.split(",")):
    name, N, K = SHAPES[si]
    W = torch.randn(N, K, device="cuda", dtype=dt)
    q, s = F.quantize_4bit(W, blocksize=64, quant_type="nf4", compress_statistics=False)
    del W
    wbytes = q.numel() + s.absmax.numel() * 4
    nc = max(2, math.ceil(384e6 / wbytes))
    copies = [(q.clone().view(-1, 1), s.absmax.clone()) for _ in range(nc)]
    reps = min(max(20, 2 * nc // 3), 120)
    for M in map(int, a.ms.split(",")):
        A = torch.randn(M, K, device="cuda", dtype=dt)
        fk = [(lambda b=b, x=x: _gemm_4bit_kernel_impl(A, b, (N, K), x, 64, "nf4")) for b, x in copies]
        cfgs = [(n, pick(R, M, N, K, kfix=False)) for n, R in tabs] + ([("dequant", "")] if a.dequant else [])
        for cyc in range(a.cycles):
            for n, tile in cfgs if cyc % 2 == 0 else cfgs[::-1]:
                if n == "dequant":
                    fns = [(lambda b=b, x=x: _dequant_linear_fallback(A, b, (N, K), x, 64, "nf4")) for b, x in copies]
                else:
                    os.environ["BNB_WMMA_CFG"] = tile
                    fns = fk
                for f in fns[:2]:
                    f()
                us = timed_rotation(fns, reps)
                w.writerow(
                    {
                        "shape": name,
                        "N": N,
                        "K": K,
                        "M": M,
                        "dtype": a.dtype,
                        "cfg": n,
                        "tile": tile,
                        "cycle": cyc,
                        "us": us,
                    }
                )
        out.flush()
    print(f"done {name}", flush=True)
    del copies
    torch.cuda.empty_cache()
