# Correctness of the gfx1102 rule tiles: default dispatch vs a float64 reference over the leaves of every bucket
# (N < 1024, K < 2560 / 3840 / 6144 / 8192 / 14336 and wider, K % 128 != 0), blocksizes and nested absmax.
import torch

import bitsandbytes as bnb
from bitsandbytes.backends.cuda.ops import _gemm_4bit_kernel_impl
import bitsandbytes.functional as F

torch.manual_seed(0)
worst = {}
SH = (
    (512, 2048),
    (2816, 704),
    (2816, 2112),
    (3072, 3072),
    (4096, 4096),
    (5120, 6144),
    (1024, 14336),
    (6144, 2560),
    (10240, 1024),
    (28672, 1024),
)
for dt in (torch.float16, torch.bfloat16, torch.float32):
    for N, K in SH:
        W = torch.randn(N, K, device="cuda", dtype=dt)
        for bs in (32, 64, 128):
            if K % bs:
                continue
            for nested in (False, True):
                q, s = F.quantize_4bit(W, blocksize=bs, quant_type="nf4", compress_statistics=nested)
                Wd = F.dequantize_4bit(q, s).double()
                for M in (1, 2, 4, 5, 8, 9, 16, 17, 24, 32, 33, 48, 49, 64, 65, 96, 128, 129, 384, 512):
                    A = torch.randn(M, K, device="cuda", dtype=dt)
                    if nested:
                        out = bnb.matmul_4bit(A, q.t(), quant_state=s)
                    else:
                        out = _gemm_4bit_kernel_impl(A, q.view(-1, 1), (N, K), s.absmax, bs, "nf4")
                    ref = A.double() @ Wd.t()
                    e = ((out.double() - ref).norm() / ref.norm()).item()
                    worst[str(dt)] = max(worst.get(str(dt), (0,)), (e, N, K, M, bs, nested))
                    assert torch.isfinite(out).all(), (dt, N, K, M, bs, nested)
for k, v in worst.items():
    e, N, K, M, bs, nested = v
    print(k, f"max rel err {e:.2e} at N={N} K={K} M={M} bs={bs} nested={nested}")
