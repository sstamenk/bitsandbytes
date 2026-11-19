"""Which BLAS runs fp32/fp16 F.linear on this build, and its speed, per torch preferred_blas_library setting."""

import time

import torch

print("torch", torch.__version__, "default preferred blas:", torch.backends.cuda.preferred_blas_library())
for lib in ("cublas", "cublaslt"):
    torch.backends.cuda.preferred_blas_library(lib)
    for dt in (torch.float32, torch.float16):
        a = torch.randn(512, 8192, device="cuda", dtype=dt)
        w = torch.randn(8192, 8192, device="cuda", dtype=dt)
        for _ in range(3):
            torch.nn.functional.linear(a, w)
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(10):
            torch.nn.functional.linear(a, w)
        torch.cuda.synchronize()
        dtm = (time.time() - t) / 10
        print(
            lib,
            torch.backends.cuda.preferred_blas_library(),
            dt,
            f"{dtm * 1e3:.2f} ms",
            f"{2 * 512 * 8192 * 8192 / dtm / 1e12:.2f} TFLOPS",
        )
maps = open("/proc/self/maps").read()
print("loaded:", sorted({ln.split()[-1] for ln in maps.splitlines() if "blas" in ln.lower() and "/" in ln}))
