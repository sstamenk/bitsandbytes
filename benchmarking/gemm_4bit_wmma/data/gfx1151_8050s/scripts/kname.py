# ruff: noqa
import torch
from torch.profiler import ProfilerActivity, profile

from bitsandbytes.backends.cuda.ops import _gemm_4bit_kernel_impl
import bitsandbytes.functional as F

for dt in (torch.float16, torch.bfloat16, torch.float32):
    W = torch.randn(4096, 4096, device="cuda", dtype=dt)
    q, s = F.quantize_4bit(W, blocksize=64, quant_type="nf4", compress_statistics=False)
    for M in (1, 16, 128):
        A = torch.randn(M, 4096, device="cuda", dtype=dt)
        _gemm_4bit_kernel_impl(A, q.view(-1, 1), (4096, 4096), s.absmax, 64, "nf4")
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            out = _gemm_4bit_kernel_impl(A, q.view(-1, 1), (4096, 4096), s.absmax, 64, "nf4")
            torch.cuda.synchronize()
        ref = A.float() @ F.dequantize_4bit(q, s).float().t()
        names = sorted({e.name for e in p.events() if e.device_type.name == "CUDA"})
        print(dt, M, "relerr=%.2e" % ((out.float() - ref).norm() / ref.norm()).item(), [n[:110] for n in names])
