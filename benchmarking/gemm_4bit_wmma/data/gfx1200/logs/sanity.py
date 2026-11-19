import time

import torch
from torch.profiler import ProfilerActivity, profile

import bitsandbytes as bnb
import bitsandbytes.functional as F

print(torch.cuda.get_device_name(0), bnb.__file__)
for dt in (torch.float16, torch.float32):
    W = torch.randn(4096, 4096, device="cuda", dtype=dt)
    q, s = F.quantize_4bit(W, blocksize=64, quant_type="nf4")
    for M in (1, 8, 64):
        A = torch.randn(M, 4096, device="cuda", dtype=dt)
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            out = bnb.matmul_4bit(A, q.t(), quant_state=s)
            torch.cuda.synchronize()
        ref = A.float() @ F.dequantize_4bit(q, s).float().t()
        names = sorted({e.name[:70] for e in p.events() if e.device_type.name == "CUDA"})
        print(dt, M, f"relerr {((out.float() - ref).norm() / ref.norm()).item():.2e}", names)
# fp32 BLAS speed check
a = torch.randn(16, 2048, device="cuda")
b = torch.randn(2048, 2048, device="cuda")
for _ in range(3):
    torch.nn.functional.linear(a, b)
torch.cuda.synchronize()
t = time.time()
for _ in range(10):
    torch.nn.functional.linear(a, b)
torch.cuda.synchronize()
print("fp32 16x2048x2048 ms", (time.time() - t) * 100)
