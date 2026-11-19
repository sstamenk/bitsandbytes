"""blas_probe.py: fp32/fp16 F.linear throughput on this GPU with torch's default, rocBLAS ("cublas") and hipBLASLt
("cublaslt") backends, to see whether the slow fp32 dequant+F.linear baseline is a BLAS-library effect."""

import torch
import torch.nn.functional as Fn

print(
    torch.cuda.get_device_properties(0).gcnArchName,
    torch.__version__,
    "default blas:",
    torch.backends.cuda.preferred_blas_library(),
)


def t(f, n=20):
    for _ in range(3):
        f()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        f()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n


for lib in ("default", "cublas", "cublaslt"):
    if lib != "default":
        torch.backends.cuda.preferred_blas_library(lib)
    for dt in (torch.float32, torch.float16):
        for N, K in ((14336, 4096), (4096, 14336), (8192, 8192)):
            W = torch.randn(N, K, device="cuda", dtype=dt)
            for M in (1, 64, 384):
                A = torch.randn(M, K, device="cuda", dtype=dt)
                ms = t(lambda: Fn.linear(A, W))
                print(
                    f"{lib:8} {str(dt)[6:]:8} N={N:5} K={K:5} M={M:3} {ms:8.3f} ms {2 * M * N * K / ms / 1e9:7.2f} TFLOP/s"
                )
