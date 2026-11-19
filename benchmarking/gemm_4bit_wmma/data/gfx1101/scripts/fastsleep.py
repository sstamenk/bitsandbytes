"""fastsleep.py script.py [args]: run a harness script in-process with torch.cuda._sleep scaled by BNB_SLEEP_SCALE
(default 0.1). bench.timed_rotation holds the GPU with _sleep(5e7) before queueing the timed launches; on gfx1101 that
is ~0.5 s per measurement (the sleep counter ticks at ~100 MHz), i.e. most of the run time. 0.1 leaves ~50 ms, still far
more than the CPU needs to queue <= 200 launches; the timed window is unchanged."""

import os
import runpy
import sys

import torch

_scale = float(os.environ.get("BNB_SLEEP_SCALE", "0.1"))
_orig = torch.cuda._sleep
torch.cuda._sleep = lambda cycles: _orig(int(cycles * _scale))
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0])))
runpy.run_path(sys.argv[0], run_name="__main__")
