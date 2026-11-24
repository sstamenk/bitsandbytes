"""clk_summary.py logs/*.clk: per job, over the samples with busy >= 50: sclk level share, current sclk min/median/max
(MHz), edge temperature median/max (C), package power median/max (W), and the perf level seen."""

import os
import re
import statistics as st
import sys

for f in sys.argv[1:]:
    rows = []
    for line in open(f):
        m = {k: v for k, v in re.findall(r"(\w+)=(\S+)", line)}
        if "busy" not in m or not m["busy"].isdigit():
            continue
        lv = re.search(r"sclk=(\d): (\d+)Mhz \*", line)
        try:
            rows.append(
                (
                    int(m["busy"]),
                    int(m.get("cur_sclk", 0)) / 1e6,
                    int(m.get("temp", 0)) / 1e3,
                    int(m.get("ppt", 0)) / 1e6,
                    m.get("lvl", "?"),
                    lv.group(2) if lv else "?",
                )
            )
        except ValueError:
            pass
    b = [r for r in rows if r[0] >= 50]
    if not b:
        print(f"{os.path.basename(f):14} no busy samples ({len(rows)} samples)")
        continue
    s = [r[1] for r in b]
    t = [r[2] for r in b]
    p = [r[3] for r in b]
    lv = {x: sum(r[5] == x for r in b) for x in sorted({r[5] for r in b})}
    print(
        f"{os.path.basename(f):14} n={len(rows):5} busy={len(b):5} sclk {min(s):.0f}/{st.median(s):.0f}/{max(s):.0f} MHz (level {lv}) temp {st.median(t):.0f}/{max(t):.0f} C ppt {st.median(p):.1f}/{max(p):.1f} W perf levels {sorted({r[4] for r in rows})}"
    )
