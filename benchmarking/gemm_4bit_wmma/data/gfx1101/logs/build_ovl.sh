#!/bin/bash
# build the hipBLASLt fp32-fix overlay (ROCm/rocm-libraries#12465) for gfx1101 (CPU-only; run between timing windows)
. ~/bnbtune/env.sh
pip() { uv pip "$@"; }; export -f pip
cd ~/bnbtune && bash ~/bnbtune/src/benchmarking/gemm_4bit_wmma/hipblaslt_fix_overlay.sh ~/bnbtune/hipblaslt_fix gfx1101
