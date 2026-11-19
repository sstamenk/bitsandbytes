#!/bin/bash
# build the hipBLASLt fp32-fix overlay (ROCm/rocm-libraries#12465) for gfx1102 from locally exported sources (no timing)
. ~/bnbtune/env.sh
export SRC_TGZ=~/bnbtune/ovlsrc/src.tar.gz PR_DIFF=~/bnbtune/ovlsrc/pr12465.diff
bash ~/bnbtune/hipblaslt_fix_overlay_local.sh ~/bnbtune/hipblaslt_fix gfx1102 > ~/bnbtune/logs/ovl_build.log 2>&1
echo "ovl rc=$?"
