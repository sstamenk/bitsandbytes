"""Scale torch.cuda._sleep by BNB_SLEEP_SCALE in every Python process (bench.py runs its timings in subprocesses).
See fastsleep.py: the harness's _sleep(5e7) pre-roll is ~0.5 s per measurement on gfx1101; the timed window is unchanged."""

import os

if os.environ.get("BNB_SLEEP_SCALE"):
    import torch

    _scale = float(os.environ["BNB_SLEEP_SCALE"])
    _orig = torch.cuda._sleep
    torch.cuda._sleep = lambda cycles: _orig(int(cycles * _scale))
