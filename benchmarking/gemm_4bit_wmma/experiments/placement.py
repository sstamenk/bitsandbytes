"""Placement sensitivity of the 4-bit GEMM: same shape, same kernel, weights at different addresses.

Each trial allocates a fresh pool, carves the rotating weight copies (>= 384 MB, as in bench.py) out of it at a
byte offset, and times the kernel. Two sweeps:
  offsets: one pool per offset in {0, 256, 512, ..., 2 MB} (probes channel/bank interleave address bits)
  fresh:   repeated fresh pools at offset 0 (probes physical page placement)
Kernel selection uses the dev override BNB_GEMM4BIT_KERNEL (simt|wmma), so run with a -DBNB_WMMA_DEV=ON build.

usage: BNB_GEMM4BIT_KERNEL=wmma python experiments/placement.py --shape 13 --m 32 --dtype fp16 --out p.csv
"""

import argparse
import csv
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from bench import timed_rotation
from shapes import SHAPES

from bitsandbytes.backends.cuda.ops import _gemm_4bit_kernel_impl
import bitsandbytes.functional as F

p = argparse.ArgumentParser()
p.add_argument("--shape", type=int, required=True)
p.add_argument("--m", type=int, required=True)
p.add_argument("--dtype", default="fp16")
p.add_argument("--fresh", type=int, default=12, help="fresh-pool trials at offset 0")
p.add_argument("--reps", type=int, default=120)
p.add_argument("--out", required=True)
args = p.parse_args()

dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
name, N, K = SHAPES[args.shape]
torch.manual_seed(0)
W = torch.randn(N, K, device="cuda", dtype=dt)
q, s = F.quantize_4bit(W, blocksize=64, quant_type="nf4", compress_statistics=False)
del W
qb = q.view(torch.uint8).reshape(-1)
wbytes = qb.numel()
ncopies = max(2, math.ceil(384e6 / wbytes))
A = torch.randn(args.m, K, device="cuda", dtype=dt)
kernel = os.environ.get("BNB_GEMM4BIT_KERNEL", "auto")

MAX_OFF = 2 << 20
stride = (wbytes + MAX_OFF + 4095) // 4096 * 4096  # each copy gets its own slot, room for any offset


def trial(offset):
    pool = torch.empty(ncopies * stride, dtype=torch.uint8, device="cuda")
    copies = []
    for i in range(ncopies):
        v = pool[i * stride + offset : i * stride + offset + wbytes]
        v.copy_(qb)
        copies.append((v.view(-1, 1), s.absmax.clone()))
    fns = [(lambda b=b, a=a: _gemm_4bit_kernel_impl(A, b, (N, K), a, 64, "nf4")) for b, a in copies]
    for f in fns[:3]:
        f()
    us = timed_rotation(fns, max(args.reps, ncopies))
    base = pool.data_ptr()
    del copies, fns, pool
    torch.cuda.empty_cache()
    return us, base


rows = []
offsets = [0] + [1 << b for b in range(8, 22)]  # 0, 256 B .. 2 MB
for off in offsets:
    us, base = trial(off)
    rows.append({"sweep": "offset", "offset": off, "trial": 0, "base_mod_2M": base % (2 << 20), "us": us})
    print(f"{name} M={args.m} {args.dtype} {kernel} offset {off:8d}: {us:8.1f} us", flush=True)
for i in range(args.fresh):
    us, base = trial(0)
    rows.append({"sweep": "fresh", "offset": 0, "trial": i, "base_mod_2M": base % (2 << 20), "us": us})
    print(f"{name} M={args.m} {args.dtype} {kernel} fresh {i:2d}: {us:8.1f} us", flush=True)

with open(args.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["shape", "N", "K", "M", "dtype", "kernel", "sweep", "offset", "trial",
                                      "base_mod_2M", "us"])
    w.writeheader()
    for r in rows:
        w.writerow({"shape": name, "N": N, "K": K, "M": args.m, "dtype": args.dtype, "kernel": kernel, **r})
ts = [r["us"] for r in rows]
print(f"SUMMARY {name} M={args.m} {kernel}: min {min(ts):.1f} median {sorted(ts)[len(ts)//2]:.1f} max {max(ts):.1f} "
      f"(max/min {max(ts)/min(ts):.2f})")
