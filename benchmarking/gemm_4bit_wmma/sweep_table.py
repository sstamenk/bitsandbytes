"""Candidate WMMA tiles (the BNB_WMMA_DEV_TRY table compiled into the dev launcher) and the LDS budget check."""

import os
import re

import torch

# Output rows per WMMA: gfx11 runs two 8-row half-wave products, gfx12 one 16-row product.
ROWS = 16 if torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx12") else 8

_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../csrc/gemm_4bit_wmma.cu")).read()
TABLE = sorted(
    {tuple(map(int, t)) for t in re.findall(r"BNB_WMMA_DEV_TRY\((\d+), (\d+), (\d+), (\d+), (\d+)\)", _src)}
)


def smem_ok(wm, wn, wmp, wnp, ks, kch, planes):
    mt, nt = wmp * ROWS * wm, wnp * 16 * wn
    return max(2 * planes * ks * mt * (64 * kch + 8) * 2, ks * mt * nt * 4 if ks > 1 else 0) <= 48 * 1024
