#!/bin/bash
# hipblaslt_fix_overlay.sh <out dir> [gfx arch, default gfx1201]: build the active ROCm nightly venv's exact hipBLASLt host libraries plus the
# fp32 fix (ROCm/rocm-libraries#12465) and write <out dir>/env.sh, which loads them in place of the wheel's
# copies. The wheel's device objects and every other ROCm component stay unchanged. Needed on nightlies from
# 10.2.0a20260913 until the fix ships: without it, every small torch fp32 GEMM on gfx12 takes ~5-20 ms.
# Requires: an activated venv with rocm[devel], git, network access. Host code only; the arch selects the wheel's
# device objects the overlay loads (e.g. gfx1100 on a W7900).
# Re-running is safe: the checkout is reset to the pin and the fix re-applied each time. The fix is applied as the
# PR's current diff against its merge base, so a rebased or force-pushed PR still works.
set -euo pipefail
out=$(realpath -m "$1"); arch=${2:-gfx1201}; mkdir -p "$out"; work=$out/work; mkdir -p "$work"
site=$(python -c "import site; print(site.getsitepackages()[0])")
devel=$site/_rocm_sdk_devel
pin=$(python -c "import json; d = json.load(open('$site/_rocm_sdk_core/share/therock/therock_manifest.json')); \
print([s['pin_sha'] for s in d['submodules'] if s['submodule_name'] == 'rocm-libraries'][0])")
pip install -q "cmake<4" ninja

# rocm-libraries at the wheel's pin, sparse checkout of hipBLASLt (+ the shared dirs it builds from), plus the fix.
src=$work/rocm-libraries
if [ ! -d "$src" ]; then
  git clone -q --filter=blob:none --no-checkout https://github.com/ROCm/rocm-libraries.git "$src"
  git -C "$src" sparse-checkout set projects/hipblaslt shared
fi
git -C "$src" fetch -q origin "$pin" +pull/12465/head:pr12465
git -C "$src" checkout -q --detach --force "$pin"
git -C "$src" reset -q --hard "$pin"
git -C "$src" clean -qfd
fix=$(git -C "$src" rev-parse pr12465)
git -C "$src" diff "$(git -C "$src" merge-base "$pin" pr12465)" pr12465 -- projects/hipblaslt | git -C "$src" apply --index

# Build-only dependencies of the rocroller CMake package: static PIC yaml-cpp, header-only fmt and msgpack.
[ -d "$work/yaml-src" ] || git clone -q --depth 1 -b 0.8.0 https://github.com/jbeder/yaml-cpp.git "$work/yaml-src"
cmake -G Ninja -S "$work/yaml-src" -B "$work/yaml-build" -DCMAKE_BUILD_TYPE=Release -DYAML_BUILD_SHARED_LIBS=OFF \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DYAML_CPP_BUILD_TESTS=OFF -DYAML_CPP_BUILD_TOOLS=OFF \
  -DCMAKE_INSTALL_PREFIX="$work/yaml" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 > /dev/null
cmake --build "$work/yaml-build" > /dev/null && cmake --install "$work/yaml-build" > /dev/null
[ -d "$work/fmt-src" ] || git clone -q --depth 1 -b 11.1.4 https://github.com/fmtlib/fmt.git "$work/fmt-src"
mkdir -p "$work/fmt-cmake"
cat > "$work/fmt-cmake/fmtConfig.cmake" <<CFG
if(NOT TARGET fmt::fmt-header-only)
    add_library(fmt::fmt-header-only INTERFACE IMPORTED)
    set_target_properties(fmt::fmt-header-only PROPERTIES
        INTERFACE_COMPILE_DEFINITIONS "FMT_HEADER_ONLY=1" INTERFACE_INCLUDE_DIRECTORIES "$work/fmt-src/include")
endif()
set(fmt_FOUND TRUE)
CFG
[ -d "$work/msgpack-src" ] || git clone -q --depth 1 -b cpp-6.1.1 https://github.com/msgpack/msgpack-c.git "$work/msgpack-src"
cmake -S "$work/msgpack-src" -B "$work/msgpack-build" -DMSGPACK_USE_BOOST=OFF -DMSGPACK_BUILD_TESTS=OFF \
  -DMSGPACK_BUILD_EXAMPLES=OFF -DCMAKE_INSTALL_PREFIX="$work/msgpack" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 > /dev/null
cmake --install "$work/msgpack-build" > /dev/null

# Host-only hipBLASLt against the venv's shared origami/rocroller/roctx, like the wheel. No build RPATH, so the
# library cannot pick up a build-tree libtensilelite-host (that hides the fix).
cmake -G Ninja -S "$src/projects/hipblaslt" -B "$work/build" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$devel/lib/llvm/bin/clang" -DCMAKE_CXX_COMPILER="$devel/lib/llvm/bin/clang++" \
  -DCMAKE_PREFIX_PATH="$devel;$site/_rocm_sdk_libraries;$site/_rocm_sdk_core;$work/yaml;$work/msgpack" \
  -Dfmt_DIR="$work/fmt-cmake" -DROCM_PATH="$devel" -DGPU_TARGETS="$arch" \
  -DHIPBLASLT_ENABLE_DEVICE=OFF -DHIPBLASLT_ENABLE_EXTOPS=OFF -DHIPBLASLT_ENABLE_MATRIX_TRANSFORM=OFF \
  -DHIPBLASLT_ENABLE_CLIENT=OFF -DHIPBLASLT_ENABLE_THEROCK=ON -DHIPBLASLT_ENABLE_FETCH=OFF \
  -DHIPBLASLT_ENABLE_MXDATAGENERATOR=OFF -DTENSILELITE_BUILD_TESTING=OFF -DCMAKE_SKIP_RPATH=ON \
  -DCMAKE_LINKER_TYPE=LLD > "$work/configure.log"
cmake --build "$work/build" > "$work/build.log"
cp -L "$work/build/library/libhipblaslt.so.1" "$work/build/tensilelite/libtensilelite-host.so.1" "$out/"

# rocm_sdk preloads hipBLASLt by full path, which would map the wheel's copy next to an LD_PRELOADead one;
# redirect the preload instead. The overlay finds its kernels through HIPBLASLT_TENSILE_LIBPATH.
mkdir -p "$out/bootstrap"
cat > "$out/bootstrap/sitecustomize.py" <<'PY'
"""Redirect the ROCm wheel's hipBLASLt preload to the fixed overlay in BNB_HIPBLASLT_OVERLAY (ROCm/rocm-libraries#12465)."""

import ctypes
import os
from pathlib import Path

import rocm_sdk

_overlay = Path(os.environ["BNB_HIPBLASLT_OVERLAY"])
ctypes.CDLL(str(_overlay / "libtensilelite-host.so.1"), mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL(str(_overlay / "libhipblaslt.so.1"), mode=ctypes.RTLD_GLOBAL)
_find = rocm_sdk.find_libraries


def _find_libraries(*names):
    out = []
    for n in names:
        out += [_overlay / "libhipblaslt.so.1"] if n == "hipblaslt" else _find(n)
    return out


rocm_sdk.find_libraries = _find_libraries
PY
cat > "$out/env.sh" <<ENV
# source to load the fixed hipBLASLt host libraries : rocm-libraries $pin + ROCm/rocm-libraries#12465 at $fix
export BNB_HIPBLASLT_OVERLAY=$out
export PYTHONPATH=$out/bootstrap\${PYTHONPATH:+:\$PYTHONPATH}
export HIPBLASLT_TENSILE_LIBPATH=$site/_rocm_sdk_libraries/lib/hipblaslt/library/$arch
ENV
echo "built: source $out/env.sh"
