"""sweep1103.py: sweep.py (in-process WMMA tile sweep, -DBNB_WMMA_DEV=ON, BNB_GEMM4BIT_KERNEL=wmma) with
- a warm-up (--warmup seconds of timed rotations on shape 0 with the default tile, discarded) before any timing,
  against the first-in-process high readings seen on llama3-70b gate/up at M <= 3 (report_gfx1102.md);
- kch = 2 candidates also for K % 128 != 0 (prescale-aware: cfg kch is KCH_MAX; the launcher lowers the K step to 1
  but keeps the fp16 prescaled kernel at M >= 96), so rule tables can be looked up without lowering kch;
- --repcap (max timed calls per measurement, default 120 as sweep.py);
- --cool: idle seconds before each shape (added after the machine went down mid-sweep at edge 91-105 C).
- --minmt (default 32, as sweep.py: tiles under 32 rows skipped at M >= 64) and --mtmax (only tiles with at most that many rows).
Rows: shape,N,K,M,cfg,us (one dtype per file), written as measured."""

import argparse
import csv
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.getcwd())
from bench import timed_rotation
from shapes import SHAPES
from sweep_table import ROWS, TABLE, smem_ok

from bitsandbytes.backends.cuda.ops import _gemm_4bit_kernel_impl
import bitsandbytes.functional as F

p = argparse.ArgumentParser()
p.add_argument("--shapes", default=",".join(map(str, range(40))))
p.add_argument("--ms", default="1,8,16,32,64,128,256")
p.add_argument("--dtype", default="fp16")
p.add_argument("--cfgs", default="auto")
p.add_argument("--out", default="sweep.csv")
p.add_argument("--reps", type=int, default=20)
p.add_argument("--repcap", type=int, default=120)
p.add_argument("--warmup", type=float, default=5.0)
p.add_argument("--minmt", type=int, default=32, help="at M >= 64 skip tiles with fewer rows (sweep.py: 32)")
p.add_argument("--mtmax", type=int, default=1 << 20, help="only tiles with at most this many rows")
p.add_argument("--cool", type=float, default=0.0, help="idle seconds before each shape (thermal cooldown)")
args = p.parse_args()
dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
PLANES = 3 if args.dtype == "fp32" else 1


def cands(M, K):
    if args.cfgs != "auto":
        return [tuple(map(int, c.split(","))) for c in args.cfgs.split(";")]
    out = []
    for wm, wn, wmp, wnp, ks in TABLE:
        mt = wmp * ROWS * wm
        if mt > max(ROWS, 2 * M) or (M >= 64 and mt < args.minmt) or mt > args.mtmax:
            continue
        for kch in (1, 2):
            if K % 64 == 0 and smem_ok(wm, wn, wmp, wnp, ks, kch, PLANES):
                out.append((wm, wn, wmp, wnp, ks, kch))
    return out


def setup(si):
    name, N, K = SHAPES[si]
    W = torch.randn(N, K, device="cuda", dtype=dt)
    q, s = F.quantize_4bit(W, blocksize=64, quant_type="nf4", compress_statistics=False)
    del W
    wbytes = q.numel() + s.absmax.numel() * 4
    ncopies = max(2, math.ceil(384e6 / wbytes))
    return name, N, K, ncopies, [(q.clone().view(-1, 1), s.absmax.clone()) for _ in range(ncopies)]


torch.manual_seed(0)
if args.warmup > 0:
    os.environ.pop("BNB_WMMA_CFG", None)
    name, N, K, nc, copies = setup(0)
    A = torch.randn(1, K, device="cuda", dtype=dt)
    fns = [(lambda b=b, a=a: _gemm_4bit_kernel_impl(A, b, (N, K), a, 64, "nf4")) for b, a in copies]
    t0, ws = time.time(), []
    while time.time() - t0 < args.warmup:
        ws.append(timed_rotation(fns, 20))
    print("warmup", len(ws), " ".join(f"{x:.0f}" for x in ws[:8]), "... last", f"{ws[-1]:.0f}", flush=True)
    del copies
    torch.cuda.empty_cache()

out_f = open(args.out, "w", newline="")
writer = csv.DictWriter(out_f, fieldnames=["shape", "N", "K", "M", "cfg", "us"])
writer.writeheader()
for si in map(int, args.shapes.split(",")):
    if args.cool > 0:
        torch.cuda.synchronize()
        time.sleep(args.cool)
    name, N, K, ncopies, copies = setup(si)
    if args.cool > 0:  # re-warm on this shape after the idle period (default tile, discarded)
        os.environ.pop("BNB_WMMA_CFG", None)
        A = torch.randn(1, K, device="cuda", dtype=dt)
        fns = [(lambda b=b, a=a: _gemm_4bit_kernel_impl(A, b, (N, K), a, 64, "nf4")) for b, a in copies]
        t0 = time.time()
        while time.time() - t0 < 2.0:
            timed_rotation(fns, 20)
    for M in map(int, args.ms.split(",")):
        A = torch.randn(M, K, device="cuda", dtype=dt)
        fns = [(lambda b=b, a=a: _gemm_4bit_kernel_impl(A, b, (N, K), a, 64, "nf4")) for b, a in copies]
        res = []
        for c in cands(M, K):
            os.environ["BNB_WMMA_CFG"] = ",".join(map(str, c))
            for f in fns[:2]:
                f()
            us = timed_rotation(fns, min(max(args.reps, 2 * ncopies // 3), args.repcap))
            res.append((us, c))
            writer.writerow({"shape": name, "N": N, "K": K, "M": M, "cfg": ",".join(map(str, c)), "us": us})
        out_f.flush()
        res.sort()
        print(
            f"{name[:30]:30} M={M:4} best {res[0][0]:8.1f} {res[0][1]}  2nd {res[1][0]:8.1f} {res[1][1]}  worst {res[-1][0]:8.1f}",
            flush=True,
        )
    del copies
    torch.cuda.empty_cache()
out_f.close()
