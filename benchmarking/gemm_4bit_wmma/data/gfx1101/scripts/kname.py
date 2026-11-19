"""kname.py: which kernel (and WMMA tile template arguments) the default dispatch launches, plus relative error vs a
float64-free fp32 reference. Shapes/M chosen so the gfx1100 and gfx1101 tables pick different tiles."""

import torch
from torch.profiler import ProfilerActivity, profile

from bitsandbytes.backends.cuda.ops import _gemm_4bit_kernel_impl
import bitsandbytes.functional as F

print(torch.cuda.get_device_properties(0).gcnArchName)
cases = [
    (torch.float16, 1, 4096, 4096),
    (torch.float16, 16, 4096, 4096),
    (torch.float16, 40, 4096, 4096),
    (torch.float16, 128, 4096, 4096),
    (torch.bfloat16, 40, 4096, 4096),
    (torch.float32, 1, 4096, 4096),
    (torch.float32, 8, 4096, 4096),
    (torch.float32, 16, 4096, 4096),
    (torch.float32, 128, 4096, 4096),
]
for dt, M, N, K in cases:
    W = torch.randn(N, K, device="cuda", dtype=dt)
    q, s = F.quantize_4bit(W, blocksize=64, quant_type="nf4", compress_statistics=False)
    A = torch.randn(M, K, device="cuda", dtype=dt)
    _gemm_4bit_kernel_impl(A, q.view(-1, 1), (N, K), s.absmax, 64, "nf4")
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        out = _gemm_4bit_kernel_impl(A, q.view(-1, 1), (N, K), s.absmax, 64, "nf4")
        torch.cuda.synchronize()
    ref = A.float() @ F.dequantize_4bit(q, s).float().t()
    names = sorted({e.name for e in p.events() if e.device_type.name == "CUDA"})
    print(str(dt)[6:], f"M={M} N={N} K={K}", f"relerr={((out.float() - ref).norm() / ref.norm()).item():.2e}", names)
