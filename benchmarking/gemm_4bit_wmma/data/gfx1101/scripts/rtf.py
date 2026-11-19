"""rtf.py --dtype fp32 --ms 64,128 --cycles 3 --out x.csv: default dispatch with BNB_WMMA_ROW_TILES_FIRST=0 vs 1
(dev build; read per launch), interleaved per (shape, M) in alternating order, on the same weight copies."""

import argparse
import csv
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import timed_rotation
from shapes import SHAPES

from bitsandbytes.backends.cuda.ops import _gemm_4bit_kernel_impl
import bitsandbytes.functional as F

p = argparse.ArgumentParser()
p.add_argument("--dtype", default="fp32")
p.add_argument("--ms", required=True)
p.add_argument("--cycles", type=int, default=3)
p.add_argument("--out", required=True)
a = p.parse_args()
dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]
torch.manual_seed(0)
w = csv.DictWriter(open(a.out, "w", newline=""), fieldnames=["shape", "N", "K", "M", "dtype", "cfg", "cycle", "us"])
w.writeheader()
for name, N, K in SHAPES:
    W = torch.randn(N, K, device="cuda", dtype=dt)
    q, s = F.quantize_4bit(W, blocksize=64, quant_type="nf4", compress_statistics=False)
    del W
    nc = max(2, math.ceil(384e6 / (q.numel() + s.absmax.numel() * 4)))
    copies = [(q.clone().view(-1, 1), s.absmax.clone()) for _ in range(nc)]
    reps = min(max(20, 2 * nc // 3), 120)
    for M in map(int, a.ms.split(",")):
        A = torch.randn(M, K, device="cuda", dtype=dt)
        fns = [(lambda b=b, x=x: _gemm_4bit_kernel_impl(A, b, (N, K), x, 64, "nf4")) for b, x in copies]
        cfgs = ["0", "1"]
        for cyc in range(a.cycles):
            for c in cfgs if cyc % 2 == 0 else cfgs[::-1]:
                os.environ["BNB_WMMA_ROW_TILES_FIRST"] = c
                for f in fns[:2]:
                    f()
                w.writerow(
                    {
                        "shape": name,
                        "N": N,
                        "K": K,
                        "M": M,
                        "dtype": a.dtype,
                        "cfg": "rtf" + c,
                        "cycle": cyc,
                        "us": timed_rotation(fns, reps),
                    }
                )
    del copies
    torch.cuda.empty_cache()
