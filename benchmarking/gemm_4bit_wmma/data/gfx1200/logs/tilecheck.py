# Correctness of the gfx1200 rule tiles: fused (default dispatch) vs float64 reference over N/M buckets.
import torch

from bitsandbytes.backends.cuda.ops import _gemm_4bit_kernel_impl
import bitsandbytes.functional as F

torch.manual_seed(0)
worst = {}
for dt in (torch.float16, torch.bfloat16, torch.float32):
    for N, K in ((512, 2048), (2816, 704), (4096, 1536), (5376, 2048), (10240, 1024), (28672, 1024), (36864, 512)):
        W = torch.randn(N, K, device="cuda", dtype=dt)
        for bs in (32, 64, 128):
            for nested in (False, True):
                if K % bs:
                    continue
                q, s = F.quantize_4bit(W, blocksize=bs, quant_type="nf4", compress_statistics=nested)
                Wd = F.dequantize_4bit(q, s).double()
                for M in (1, 2, 5, 8, 16, 17, 24, 32, 40, 96, 100, 128, 129, 160, 200, 384, 512):
                    A = torch.randn(M, K, device="cuda", dtype=dt)
                    if nested:
                        out = F.gemv_4bit(A, q.t(), state=s) if M == 1 else None
                        import bitsandbytes as bnb

                        out = bnb.matmul_4bit(A, q.t(), quant_state=s)
                    else:
                        out = _gemm_4bit_kernel_impl(A, q.view(-1, 1), (N, K), s.absmax, bs, "nf4")
                    ref = A.double() @ Wd.t()
                    e = ((out.double() - ref).norm() / ref.norm()).item()
                    k = str(dt)
                    worst[k] = max(worst.get(k, (0,)), (e, N, K, M, bs, nested))
                    assert torch.isfinite(out).all(), (dt, N, K, M, bs)
print(worst)
