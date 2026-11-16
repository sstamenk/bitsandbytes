"""Candidate WMMA tiles (wmma_dev_shapes, compiled into the -DBNB_WMMA_DEV=ON launcher) and the LDS budget check."""

import os
import re

import torch

# Output rows per WMMA on the device being tuned (the current one): gfx11 runs two 8-row half-wave products, gfx120x one
# 16-row product. Same test as the C launcher (gfx125x has other WMMAs and no WMMA kernel here).
ROWS = 16 if torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.startswith("gfx120") else 8

_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../csrc/gemm_4bit_wmma.cu")).read()
_dev = _src[_src.index("wmma_dev_shapes[][5] = {") : _src.index("};", _src.index("wmma_dev_shapes[][5] = {"))]
TABLE = sorted({tuple(map(int, t)) for t in re.findall(r"\{(\d+), (\d+), (\d+), (\d+), (\d+)\}", _dev)})


def smem_ok(wm, wn, wmp, wnp, ks, kch, planes):
    mt, nt = wmp * ROWS * wm, wnp * 16 * wn
    return max(2 * planes * ks * mt * (64 * kch + 8) * 2, ks * mt * nt * 4 if ks > 1 else 0) <= 48 * 1024
