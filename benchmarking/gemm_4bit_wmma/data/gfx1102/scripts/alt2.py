"""alt2.py: in-process interleaved comparison of dispatch configurations on the same weight copies and activations
(needs a -DBNB_WMMA_DEV=ON build: BNB_GEMM4BIT_KERNEL and BNB_WMMA_CFG are read on every call). For every (shape, M)
the configurations run in alternating order (forward on even cycles, reverse on odd), --cycles times, after one
warm-up pass over all configurations
one CSV row per measurement.

Configurations (--configs name=spec,...):
  table:<rules>              forced WMMA with the tile the rule table picks (rules11 names or a RULES file)
  simt                       forced SIMT
  cut:<cutoffs>:<rules>      SIMT or WMMA by an fp32 cutoff list "max_m/min_n+..." (WMMA iff N >= min_n of the
                             first row with M <= max_m
                             WMMA past the last row), tile from <rules>
  dequant[:<blas>]           dequant + F.linear
  optional torch.backends.cuda.preferred_blas_library (cublas, cublaslt)
usage: python alt2.py --configs old=simt,new=cut:3/1073741824+4/8192:rulesB32.py --dtype fp32 --ms 4 --cycles 5 --out x.csv"""

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
p.add_argument("--configs", required=True)
p.add_argument("--dtype", default="fp16")
p.add_argument("--ms", required=True)
p.add_argument("--shapes", default=",".join(map(str, range(40))))
p.add_argument("--cycles", type=int, default=5)
p.add_argument("--blocksize", type=int, default=64)
p.add_argument("--out", required=True)
a = p.parse_args()
cfgs = [tuple(s.split("=", 1)) for s in a.configs.split(",")]
dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]
bs = a.blocksize
default_blas = torch.backends.cuda.preferred_blas_library()


def resolve(spec, M, N, K):
    """-> (kind, kernel env, tile, blas)"""
    if spec == "simt":
        return ("fused", "simt", "", None)
    if spec.startswith("table:"):
        return ("fused", "wmma", pick(load(spec[6:]), M, N, K, kfix=False), None)
    if spec.startswith("cut:"):
        _, cut, rules = spec.split(":", 2)
        rows = [tuple(map(int, r.split("/"))) for r in cut.split("+")]
        wm = next((N >= mn for mx, mn in rows if M <= mx), True)
        return ("fused", "wmma", pick(load(rules), M, N, K, kfix=False), None) if wm else ("fused", "simt", "", None)
    if spec.startswith("dequant"):
        return ("dequant", "", "", spec.split(":", 1)[1] if ":" in spec else None)
    raise ValueError(spec)


torch.manual_seed(0)
out = open(a.out, "w", newline="")
w = csv.DictWriter(
    out, fieldnames=["shape", "N", "K", "M", "dtype", "blocksize", "cfg", "kernel", "tile", "blas", "cycle", "us"]
)
w.writeheader()
for si in map(int, a.shapes.split(",")):
    name, N, K = SHAPES[si]
    W = torch.randn(N, K, device="cuda", dtype=dt)
    q, s = F.quantize_4bit(W, blocksize=bs, quant_type="nf4", compress_statistics=False)
    del W
    wbytes = q.numel() + s.absmax.numel() * 4
    nc = max(2, math.ceil(384e6 / wbytes))
    copies = [(q.clone().view(-1, 1), s.absmax.clone()) for _ in range(nc)]
    reps = min(max(20, 2 * nc // 3), 120)
    for M in map(int, a.ms.split(",")):
        A = torch.randn(M, K, device="cuda", dtype=dt)
        fk = [(lambda b=b, x=x: _gemm_4bit_kernel_impl(A, b, (N, K), x, bs, "nf4")) for b, x in copies]
        fd = [(lambda b=b, x=x: _dequant_linear_fallback(A, b, (N, K), x, bs, "nf4")) for b, x in copies]
        rs = [(n, resolve(spec, M, N, K)) for n, spec in cfgs]

        def run(r, timed):
            kind, kern, tile, blas = r
            torch.backends.cuda.preferred_blas_library(blas or default_blas)
            if kind == "dequant":
                fns = fd
            else:
                os.environ["BNB_GEMM4BIT_KERNEL"] = kern
                if tile:
                    os.environ["BNB_WMMA_CFG"] = tile
                else:
                    os.environ.pop("BNB_WMMA_CFG", None)
                fns = fk
            for f in fns[:2]:
                f()
            return timed_rotation(fns, reps) if timed else None

        for _, r in rs:
            run(r, False)  # warm-up pass
        for cyc in range(a.cycles):
            for n, r in rs if cyc % 2 == 0 else rs[::-1]:
                us = run(r, True)
                w.writerow(
                    {
                        "shape": name,
                        "N": N,
                        "K": K,
                        "M": M,
                        "dtype": a.dtype,
                        "blocksize": bs,
                        "cfg": n,
                        "kernel": r[0] if r[0] == "dequant" else r[1],
                        "tile": r[2],
                        "blas": str(torch.backends.cuda.preferred_blas_library()),
                        "cycle": cyc,
                        "us": us,
                    }
                )
        out.flush()
    print(f"done {name}", flush=True)
    del copies
    torch.cuda.empty_cache()
torch.backends.cuda.preferred_blas_library(default_blas)
