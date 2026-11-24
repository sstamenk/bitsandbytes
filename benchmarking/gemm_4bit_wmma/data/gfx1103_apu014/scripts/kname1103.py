"""kname1103.py: kernel names the default dispatch launches (torch.profiler), NF4 blocksize 64, for (N, K, M) cases
covering the gfx1103 tile-table leaves. The profiler sometimes returns no event; a case is retried up to 5 times."""

import torch

from bitsandbytes.backends.cuda.ops import _gemm_4bit_kernel_impl
import bitsandbytes.functional as F

CASES = [(4096, 4096, M) for M in (1, 2, 3, 4, 8, 16, 24, 40, 64, 96, 128, 512)] + [
    (2048, 8192, 1),
    (10240, 2560, 24),
    (4096, 14336, 40),
    (2816, 2112, 128),
    (512, 2048, 128),
]
for dt in (torch.float16, torch.bfloat16, torch.float32):
    for N, K, M in CASES:
        q, s = F.quantize_4bit(
            torch.randn(N, K, device="cuda", dtype=dt), blocksize=64, quant_type="nf4", compress_statistics=False
        )
        A = torch.randn(M, K, device="cuda", dtype=dt)
        names = []
        for _ in range(5):
            _gemm_4bit_kernel_impl(A, q.view(-1, 1), (N, K), s.absmax, 64, "nf4")
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
                _gemm_4bit_kernel_impl(A, q.view(-1, 1), (N, K), s.absmax, 64, "nf4")
                torch.cuda.synchronize()
            names = [e.name for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA]
            if names:
                break
        print(str(dt).split(".")[1], f"N={N} K={K} M={M}", names[0].split("(")[0] if names else "none")
