"""Correctness of every tile in the dev launcher's BNB_WMMA_CFG table (the sweep candidates).
Requires a -DBNB_WMMA_DEV=ON build; run with BNB_GEMM4BIT_KERNEL=wmma."""

import itertools
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep_table import ROWS, TABLE, smem_ok

from bitsandbytes.backends.cuda.ops import _gemm_4bit_kernel_impl
import bitsandbytes.functional as F

torch.manual_seed(0)
cases = [(1, 300, 256), (13, 100, 704), (40, 777, 1024), (70, 129, 512)]
tol = {torch.float16: 5e-3, torch.bfloat16: 2e-2, torch.float32: 1e-5}
fails = n = 0
for (M, N, K), dt, bs, nested in itertools.product(
    cases, [torch.float16, torch.bfloat16, torch.float32], [32, 64, 256], [False, True]
):
    if K % bs:
        continue
    A = torch.randn(M, K, device="cuda", dtype=dt)
    W = torch.randn(N, K, device="cuda", dtype=dt)
    q, s = F.quantize_4bit(W, blocksize=bs, quant_type="nf4", compress_statistics=nested)
    b = torch.randn(N, device="cuda", dtype=dt)
    ref = A.double() @ F.dequantize_4bit(q, s).double().t() + b.double()
    kw = dict(absmax_8bit=s.absmax, absmax_code=s.state2.code, absmax_offset=s.offset) if nested else {}
    absmax = s.state2.absmax if nested else s.absmax
    planes = 3 if dt == torch.float32 else 1
    for cfg in TABLE:
        for kch in (1, 2):
            if K % (64 * kch) or not smem_ok(*cfg, kch, planes):
                continue
            os.environ["BNB_WMMA_CFG"] = ",".join(map(str, (*cfg, kch)))
            out = _gemm_4bit_kernel_impl(A, q.view(-1, 1), (N, K), absmax, bs, "nf4", b, **kw)
            rel = ((out.double() - ref).norm() / ref.norm()).item()
            n += 1
            if not rel < tol[dt]:
                fails += 1
                print(f"FAIL cfg={cfg},{kch} M={M} N={N} K={K} {dt} bs={bs} nested={nested} rel={rel:.3g}")
print(f"rows/wmma={ROWS} checked={n} fails={fails}")
