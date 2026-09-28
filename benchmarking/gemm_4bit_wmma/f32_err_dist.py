import os
import statistics as st

import torch

from bitsandbytes.backends.cuda.ops import _gemm_4bit_kernel_impl
import bitsandbytes.functional as F

torch.manual_seed(0)
res = {"simt": [], "wmma": [], "torch": []}
for M in (1, 4, 16, 24, 40, 100):
    for bs in (32, 64):
        for qt in ("nf4", "fp4"):
            N, K = 1000, 1024
            A = torch.randn(M, K, device="cuda")
            B = torch.randn(N, K, device="cuda") / K**0.5
            q, s = F.quantize_4bit(B, blocksize=bs, quant_type=qt)
            Bd = F.dequantize_4bit(q, s)
            ref = A.double() @ Bd.double().t()
            outs = {}
            for k in ("simt", "wmma"):
                os.environ["BNB_GEMM4BIT_KERNEL"] = k
                outs[k] = _gemm_4bit_kernel_impl(A, q.view(-1, 1), (N, K), s.absmax, bs, qt)
            outs["torch"] = torch.nn.functional.linear(A, Bd)
            for k, o in outs.items():
                e = (o.double() - ref).abs()
                res[k].append(((e.max() / ref.abs().mean()).item(), (e.mean() / ref.abs().mean()).item()))
for k, v in res.items():
    mx = [a for a, b in v]
    mn = [b for a, b in v]
    print(
        f"{k:6} max-metric: median {st.median(mx):.2e} worst {max(mx):.2e} | mean-metric: median {st.median(mn):.2e} worst {max(mn):.2e}"
    )
