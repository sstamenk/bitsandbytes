"""fp32 accuracy: WMMA bf16x3 vs SIMT vs torch dequant+F.linear, all against a float64 reference."""

import itertools
import os

import torch

from bitsandbytes.backends.cuda.ops import _gemm_4bit_kernel_impl
import bitsandbytes.functional as F

torch.manual_seed(0)


def run(kern, A, q, s, N, K, bs, qt, b, nested):
    os.environ["BNB_GEMM4BIT_KERNEL"] = kern
    if nested:
        return _gemm_4bit_kernel_impl(
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
    return _gemm_4bit_kernel_impl(A, q.view(-1, 1), (N, K), s.absmax, bs, qt, b)


worst = {"wmma": 0, "simt": 0, "torch": 0}
fails = 0
n = 0
cases = [
    (2, 256, 128),
    (5, 100, 192),
    (9, 4096, 4096),
    (16, 1000, 512),
    (17, 777, 2048),
    (33, 512, 1024),
    (64, 3072, 1024),
    (100, 129, 768),
    (256, 512, 704),
    (40, 300, 1152),
]
for (M, N, K), qt, bs, nested, bias in itertools.product(
    cases, ["nf4", "fp4"], [32, 64, 256], [False, True], [False, True]
):
    if K % bs:
        continue
    A = torch.randn(M, K, device="cuda")
    W = torch.randn(N, K, device="cuda") / K**0.5
    q, s = F.quantize_4bit(W, blocksize=bs, quant_type=qt, compress_statistics=nested)
    b = torch.randn(N, device="cuda") if bias else None
    Bd = F.dequantize_4bit(q, s)
    ref = A.double() @ Bd.double().t() + (b.double() if bias else 0)
    scale = ref.abs().mean().item()
    outs = {
        "wmma": run("wmma", A, q, s, N, K, bs, qt, b, nested),
        "simt": run("simt", A, q, s, N, K, bs, qt, b, nested),
        "torch": torch.nn.functional.linear(A, Bd, b),
    }
    errs = {k: ((v.double() - ref).abs().max() / scale).item() for k, v in outs.items()}
    n += 1
    for k in errs:
        worst[k] = max(worst[k], errs[k])
    if errs["wmma"] > 4 * max(errs["simt"], errs["torch"], 1e-7):
        fails += 1
        print(f"WORSE M={M} N={N} K={K} {qt} bs={bs} nested={nested} bias={bias} {errs}")
print(
    f"{n} cases; worst max|err|/mean|ref|: "
    + ", ".join(f"{k} {v:.2e}" for k, v in worst.items())
    + f"; wmma >4x worse in {fails}"
)
