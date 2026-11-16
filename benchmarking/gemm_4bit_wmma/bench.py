"""DRAM-streaming 4-bit GEMM sweep: SIMT vs WMMA vs dequant+BLAS over the 40 LLM shapes.
Kernel "auto" times the fused path with the default C dispatch (SIMT or WMMA as the library picks).

Kernel selection inside the fused path is forced with the dev-only env var
BNB_GEMM4BIT_KERNEL (needs a -DBNB_WMMA_DEV=ON build), which the C dispatch reads once per process, so each fused
kernel runs in its own subprocess.
"""

import argparse
import csv
import json
import os
import subprocess
import sys

import torch


def timed_rotation(fns, reps):
    """Median per-call time (us) of back-to-back calls cycling through fns.

    A sleep kernel holds the GPU until every launch is queued, so CPU launch
    latency never lands inside the timed window."""
    evs = [torch.cuda.Event(enable_timing=True) for _ in range(reps + 1)]
    torch.cuda._sleep(int(5e7))
    for i in range(reps):
        evs[i].record()
        fns[i % len(fns)]()
    evs[reps].record()
    torch.cuda.synchronize()
    ts = sorted(evs[i].elapsed_time(evs[i + 1]) * 1000.0 for i in range(reps))
    return ts[len(ts) // 2]


def run_worker(args):
    """Weights rotate through enough copies (>= 384 MB total, 4x the 96 MB
    Infinity Cache) that every call streams its weights from DRAM, like
    layer-by-layer decoding."""
    import math

    from shapes import SHAPES

    from bitsandbytes.backends.cuda.ops import _dequant_linear_fallback, _gemm_4bit_kernel_impl
    import bitsandbytes.functional as F

    torch.manual_seed(0)
    dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    Ms = [int(m) for m in args.ms.split(",")]
    shapes = SHAPES if not args.shapes else [SHAPES[int(i)] for i in args.shapes.split(",")]
    out = []
    for name, N, K in shapes:
        W = torch.randn(N, K, device="cuda", dtype=dt)
        q, s = F.quantize_4bit(W, blocksize=args.blocksize, quant_type="nf4", compress_statistics=False)
        del W
        wbytes = q.numel() + s.absmax.numel() * 4
        ncopies = max(2, math.ceil(384e6 / wbytes))
        copies = [(q.clone().view(-1, 1), s.absmax.clone()) for _ in range(ncopies)]
        reps = min(max(args.reps, ncopies), 200)
        for M in Ms:
            A = torch.randn(M, K, device="cuda", dtype=dt)
            if args.kernel == "dequant":
                fns = [
                    (lambda b=b, a=a: _dequant_linear_fallback(A, b, (N, K), a, args.blocksize, "nf4"))
                    for b, a in copies
                ]
            else:
                fns = [
                    (lambda b=b, a=a: _gemm_4bit_kernel_impl(A, b, (N, K), a, args.blocksize, "nf4"))
                    for b, a in copies
                ]
            for f in fns[:3]:
                f()
            us = timed_rotation(fns, reps)
            out.append({"shape": name, "N": N, "K": K, "M": M, "dtype": args.dtype, "kernel": args.kernel, "us": us})
        del copies
        torch.cuda.empty_cache()
    json.dump(out, sys.stdout)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dtypes", default="fp16,bf16")
    p.add_argument("--kernels", default="simt,wmma,dequant")
    p.add_argument("--ms", default="1,2,4,8,12,16,24,32,48,64,96,128,192,256")
    p.add_argument("--shapes", default="")
    p.add_argument("--blocksize", type=int, default=64)
    p.add_argument("--reps", type=int, default=40)
    p.add_argument("--out", default="results.csv")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--dtype")
    p.add_argument("--kernel")
    args = p.parse_args()
    if args.worker:
        return run_worker(args)
    rows = []
    for dtype in args.dtypes.split(","):
        for kernel in args.kernels.split(","):
            env = dict(os.environ)
            env.pop("BNB_GEMM4BIT_KERNEL", None)  # "auto" times the default C dispatch
            if kernel in ("simt", "wmma"):
                env["BNB_GEMM4BIT_KERNEL"] = kernel
            cmd = [
                sys.executable,
                __file__,
                "--worker",
                "--dtype",
                dtype,
                "--kernel",
                kernel,
                "--ms",
                args.ms,
                "--shapes",
                args.shapes,
                "--blocksize",
                str(args.blocksize),
                "--reps",
                str(args.reps),
            ]
            res = subprocess.run(
                cmd, env=env, capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__))
            )
            if res.returncode != 0:
                print(res.stderr[-3000:])
                sys.exit(1)
            rows += json.loads(res.stdout.strip().splitlines()[-1])
            print(f"done {dtype} {kernel}", file=sys.stderr)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
