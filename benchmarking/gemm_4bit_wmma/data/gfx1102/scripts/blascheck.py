import time

import torch

print("preferred blas:", torch.backends.cuda.preferred_blas_library())
a = torch.randn(512, 4096, device="cuda")
w = torch.randn(4096, 4096, device="cuda")
torch.nn.functional.linear(a, w)
torch.cuda.synchronize()
maps = open("/proc/self/maps").read()
print("loaded:", sorted({ln.split()[-1] for ln in maps.splitlines() if "blas" in ln.lower() and "/" in ln}))

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
    print(dt, f"{dtm * 1e3:.2f} ms", f"{2 * 512 * 8192 * 8192 / dtm / 1e12:.2f} TFLOPS")
