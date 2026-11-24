"""pickcheck.py <header> f16.py f32.py: compare the C++ wmma_pick of the header's gfx1103 tables (pickdump.cpp) with
rules1103.pick on the candidate files, and the gfx1100 tables with rules1103's parse of the header, for the 40 shapes
and M = 1..600."""

import os
import subprocess
import sys
import tempfile

here = os.path.dirname(os.path.abspath(__file__))
hdr = os.path.abspath(sys.argv[1])
os.environ["RULES_HDR"] = hdr
sys.path.insert(0, here)
sys.path.insert(0, os.environ.get("BENCH_DIR", os.path.join(here, "..", "..", "..")))  # shapes.py
from rules1103 import load, pick  # noqa: E402
from shapes import SHAPES  # noqa: E402

exe = os.path.join(tempfile.mkdtemp(), "pickdump")
subprocess.check_call(
    ["g++", "-std=c++17", "-x", "c++", "-include", hdr, os.path.join(here, "pickdump.cpp"), "-o", exe]
)
inp = "".join(f"{n} {k}\n" for _, n, k in SHAPES)
f = tempfile.NamedTemporaryFile("w", delete=False)
f.write(inp)
f.close()
out = subprocess.run([exe], stdin=open(f.name), capture_output=True, text=True, check=True).stdout
T = {
    "g1103_f16": load(sys.argv[2]),
    "g1103_f32": load(sys.argv[3]),
    "g1100_f16": load("g1100_f16"),
    "g1100_f32": load("g1100_f32"),
}
n = bad = 0
for line in out.splitlines():
    name, M, N, K, tile = line.split()
    n += 1
    if pick(T[name], int(M), int(N), int(K)) != tile:
        bad += 1
        if bad < 10:
            print("MISMATCH", line, pick(T[name], int(M), int(N), int(K)))
print(f"{n} picks compared, {bad} mismatches")
