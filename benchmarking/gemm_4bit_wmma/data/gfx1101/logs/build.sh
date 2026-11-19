#!/bin/bash
# build.sh <srcdir> <dev 0|1>
. ~/bnbtune/env.sh
d=$1; dev=$2
cd ~/bnbtune/$d && rm -rf build bitsandbytes/*.so
extra=""; [ "$dev" = 1 ] && extra="-DBNB_WMMA_DEV=ON"
cmake -B build -G "Unix Makefiles" -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH=gfx1101 $extra -DCMAKE_HIP_COMPILER_ROCM_ROOT=$ROCM_PATH > ~/bnbtune/build_$d.log 2>&1 && cmake --build build -j24 >> ~/bnbtune/build_$d.log 2>&1
echo "build $d rc=$?"
