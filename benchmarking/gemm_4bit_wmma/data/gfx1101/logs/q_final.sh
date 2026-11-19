#!/bin/bash
# final code (worktree tune/gfx1101): release build, correctness tests, kernel names
bash ~/bnbtune/build.sh srcfin 0 && bash ~/bnbtune/pytest.sh srcfin pytest_final.log
(cd ~/bnbtune/srcfin && PYTHONPATH=~/bnbtune/srcfin python ~/bnbtune/kname.py 2>&1 | grep -v -e Warning -e USDT) > ~/bnbtune/kname_final.txt
python3 - <<"PY"
import sys; sys.path.insert(0, "/home/ubuntu/bnbtune/srcfin")
import torch
from bitsandbytes.backends.cuda.ops import _gemm_4bit_use_custom_rocm as u
for dt in (torch.float16, torch.bfloat16, torch.float32):
    print(dt, [m for m in (8, 128, 129, 384, 448, 480, 481, 512) if u(0, dt, m, 4096, 4096)], "K%64!=0:", [m for m in (8, 9) if u(0, dt, m, 4096, 4000)])
PY
