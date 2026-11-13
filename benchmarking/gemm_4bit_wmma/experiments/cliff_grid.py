"""Does the CP firmware crawler-capacity completion delay (FWDEV-240463) hit the WMMA 4-bit GEMM?

Times one forced tile at N giving grid sizes around a target block count (same K, M), weights streamed from DRAM
as in bench.py, and reports us per N-block so that a fixed completion delay stands out at the band.
usage: BNB_GEMM4BIT_KERNEL=wmma BNB_WMMA_CFG=wm,wn,wmp,wnp,ks,kch python experiments/cliff_grid.py --k 4096 --m 1 \
       --nt 16 --mt-blocks 1 --blocks 760:772 [--dtype fp16]
"""
import argparse, math, os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from bench import timed_rotation
from bitsandbytes.backends.cuda.ops import _gemm_4bit_kernel_impl
import bitsandbytes.functional as F

p = argparse.ArgumentParser()
p.add_argument("--k", type=int, required=True); p.add_argument("--m", type=int, required=True)
p.add_argument("--nt", type=int, required=True, help="N columns per block of the forced tile")
p.add_argument("--blocks", required=True, help="lo:hi total grid blocks to test")
p.add_argument("--mt-blocks", type=int, default=1, help="row tiles (grid y) at this M")
p.add_argument("--dtype", default="fp16"); p.add_argument("--rounds", type=int, default=3)
a = p.parse_args()
dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]
lo, hi = map(int, a.blocks.split(":"))
res = {}
for rnd in range(a.rounds):
    for g in range(lo, hi + 1):
        if g % a.mt_blocks: continue
        N = g // a.mt_blocks * a.nt
        W = torch.randn(N, a.k, device="cuda", dtype=dt)
        q, s = F.quantize_4bit(W, blocksize=64, quant_type="nf4", compress_statistics=False); del W
        nc = max(2, math.ceil(384e6 / (q.numel() + s.absmax.numel() * 4)))
        cp = [(q.clone().view(-1, 1), s.absmax.clone()) for _ in range(nc)]
        A = torch.randn(a.m, a.k, device="cuda", dtype=dt)
        fns = [(lambda b=b, sc=sc: _gemm_4bit_kernel_impl(A, b, (N, a.k), sc, 64, "nf4")) for b, sc in cp]
        for f in fns[:3]: f()
        res.setdefault(g, []).append(timed_rotation(fns, max(120, nc)))
        del cp, fns; torch.cuda.empty_cache()
for g in sorted(res):
    us = sorted(res[g])[len(res[g]) // 2]
    print(f"grid {g:5d} N={g // a.mt_blocks * a.nt:6d}: {us:8.2f} us  {us / g * 1000:7.2f} ns/block")
