"""kname.py: kernel names the default dispatch launches (torch.profiler), gfx1103, NF4 bs64, N=K=4096."""

import torch

from bitsandbytes.backends.cuda.ops import _gemm_4bit_kernel_impl
import bitsandbytes.functional as F

for dt in (torch.float16, torch.bfloat16, torch.float32):
    q, s = F.quantize_4bit(
        torch.randn(4096, 4096, device="cuda", dtype=dt), blocksize=64, quant_type="nf4", compress_statistics=False
    )
    for M in (1, 4, 16, 128):
        A = torch.randn(M, 4096, device="cuda", dtype=dt)
        _gemm_4bit_kernel_impl(A, q.view(-1, 1), (4096, 4096), s.absmax, 64, "nf4")
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
            _gemm_4bit_kernel_impl(A, q.view(-1, 1), (4096, 4096), s.absmax, 64, "nf4")
            torch.cuda.synchronize()
        names = [e.name for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA]
        print(str(dt).split(".")[1], M, names[:2] if names else "none")
