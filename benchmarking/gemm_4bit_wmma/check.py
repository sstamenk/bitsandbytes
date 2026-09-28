import itertools
import os

import torch

from bitsandbytes.backends.cuda.ops import _gemm_4bit_kernel_impl
import bitsandbytes.functional as F

torch.manual_seed(0)
kern = os.environ.get("BNB_GEMM4BIT_KERNEL", "")
fails = 0
cases = [
    (1, 256, 128),
    (5, 100, 192),
    (9, 4096, 4096),
    (16, 1000, 512),
    (17, 777, 2048),
    (33, 512, 1024),
    (64, 3072, 1024),
    (100, 129, 768),
    (256, 512, 704),
    (40, 300, 1152),
    (8, 6000, 1024),
    (3, 5000, 704),
    (16, 6000, 512),
    (24, 5000, 1152),
    (60, 4200, 768),
    (100, 5000, 768),
    (200, 5000, 256),
]
for (M, N, K), dt, qt, bs, nested, bias in itertools.product(
    cases, [torch.float16, torch.bfloat16], ["nf4", "fp4"], [32, 64, 256], [False, True], [False, True]
):
    if K % bs:
        continue
    A = torch.randn(M, K, device="cuda", dtype=dt)
    W = torch.randn(N, K, device="cuda", dtype=dt)
    q, s = F.quantize_4bit(W, blocksize=bs, quant_type=qt, compress_statistics=nested)
    b = torch.randn(N, device="cuda", dtype=dt) if bias else None
    Wd = F.dequantize_4bit(q, s).float()
    ref = A.float() @ Wd.t() + (b.float() if bias else 0)
    if nested:
        out = _gemm_4bit_kernel_impl(
            A,
            q.view(-1, 1),
            (N, K),
            s.state2.absmax,
            bs,
            qt,
            b,
            absmax_8bit=s.absmax,
            absmax_code=s.state2.code,
            absmax_offset=s.offset,
        )
    else:
        out = _gemm_4bit_kernel_impl(A, q.view(-1, 1), (N, K), s.absmax, bs, qt, b)
    err = (out.float() - ref).abs().max().item()
    tol = (0.02 if dt == torch.float16 else 0.1) * ref.abs().max().item() / 8 + 1e-2
    rel = ((out.float() - ref).norm() / ref.norm()).item()
    if not (err < tol and rel < (5e-3 if dt == torch.float16 else 2e-2)):
        fails += 1
        print(
            f"FAIL M={M} N={N} K={K} {dt} {qt} bs={bs} nested={nested} bias={bias} maxerr={err:.4g} tol={tol:.4g} rel={rel:.3g}"
        )
print(f"kernel={kern or 'auto'} fails={fails}")
