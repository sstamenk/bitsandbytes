#!/bin/bash
# Astra round 1 checks. (1) fp32 M=128, shapes 7 (qwen3-14b o/q) and 19 (gemma3-4b gate/up): A (src) and B (srcb)
# auto, three paired passes A/B, B/A, A/B, sleep pre-roll 0.1, plus kernel names. (2) blocksize 32 fused (final table)
# vs dequant interleaved, M=416,448,480, fp16 and bf16, 3 cycles, B build.
export BNB_SLEEP_SCALE=0.1
D=~/bnbtune/data/r1; mkdir -p $D
run() { (cd ~/bnbtune/$1/benchmarking/gemm_4bit_wmma && PYTHONPATH=~/bnbtune/fastsite:~/bnbtune/$1 python bench.py --kernels auto --dtypes fp32 --shapes 7,19 --ms 128 --reps 40 --out $D/fp32m128_$2.csv); }
for p in "src:p1A srcb:p1B" "srcb:p2B src:p2A" "src:p3A srcb:p3B"; do for x in $p; do [ -s $D/fp32m128_${x#*:}.csv ] || run ${x%%:*} ${x#*:}; done; done
for d in src srcb; do (cd ~/bnbtune/$d && PYTHONPATH=~/bnbtune/$d python - <<'PY'
import torch, bitsandbytes.functional as F
from bitsandbytes.backends.cuda.ops import _gemm_4bit_kernel_impl
from torch.profiler import profile, ProfilerActivity
for N, K in ((5120, 5120), (10240, 2560)):
    W = torch.randn(N, K, device="cuda"); q, s = F.quantize_4bit(W, blocksize=64, quant_type="nf4", compress_statistics=False)
    A = torch.randn(128, K, device="cuda"); _gemm_4bit_kernel_impl(A, q.view(-1, 1), (N, K), s.absmax, 64, "nf4"); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        _gemm_4bit_kernel_impl(A, q.view(-1, 1), (N, K), s.absmax, 64, "nf4"); torch.cuda.synchronize()
    print(N, K, sorted({e.name.split("(")[0] for e in p.events() if e.device_type.name == "CUDA"}))
PY
) > $D/kname_fp32m128_$d.txt 2>&1; done
cd ~/bnbtune/srcb/benchmarking/gemm_4bit_wmma; export PYTHONPATH=~/bnbtune/srcb
for dt in fp16 bf16; do
  [ -s $D/cap_bs32_alt_$dt.csv.done ] || { python ~/bnbtune/fastsleep.py alt.py --tables A=rules/gfx11_f16.py --dequant --blocksize 32 --dtype $dt --ms 416,448,480 --cycles 3 --out $D/cap_bs32_alt_$dt.csv > $D/cap_bs32_alt_$dt.log && touch $D/cap_bs32_alt_$dt.csv.done; }
done
echo done
