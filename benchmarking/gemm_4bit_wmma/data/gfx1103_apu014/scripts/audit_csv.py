"""audit_csv.py <root> [--live]: provenance of every measurement CSV against the job metadata.

root = the data directory (pinned/, logs/) or, with --live, the rsync of the machine's ~/bnbtune (data/, logs/).
For every CSV: the queue job whose interval (logs/<job>.meta start .. end) contains the time of its last row write
(mtime in the live tree; logs/csv_mtimes.txt, written from it by mkstage.sh, in the data directory), the tree and the
perf level read back at the job's start, the busy clock samples of that job up to that time that are not at 2599 MHz
(logs/<job>.clk), and the GPU resets (logs/dmesg_gpu_resets.txt) inside the job: before the CSV was finished (R) or
after it (r). Flags: L = level at start not `high`, U = unpinned busy sample before the CSV was finished, R, r,
N = no job found. Exit status 1 if a flagged CSV is not in ALLOWED."""

import datetime as dt
import glob
import os
import re
import sys

ALLOWED = {  # flagged CSVs kept on purpose, and why
    "alt_s1/fp32_lo_part13_before_gpu_reset.csv": "rows written before the 02:02 reset; cross-check only",
    "alt_s1/fp32_hi.csv": "after the 02:02 reset, clock not held; listed in NOTES as not used",
    "alt_s1/fp32_x.csv": "after the 02:02 reset, clock not held; listed in NOTES as not used",
    "alt_s1/fp32_d.csv": "after the 02:02 reset, clock not held; listed in NOTES as not used",
    "alt_s1/fp16.csv": "earlier candidate S1; finished 1 h before the reset of its job (02:02)",
    "alt_s1/fp16_x.csv": "earlier candidate S1; finished before the reset of its job (02:02)",
    "alt_s1/fp16_m1.csv": "earlier candidate S1; finished before the reset of its job (02:02)",
    "alt_s1/bf16.csv": "earlier candidate S1; finished before the reset of its job (02:02)",
    "alt_s1/bf16_x.csv": "earlier candidate S1; finished before the reset of its job (02:02)",
    "alt_s1/bf16_m1.csv": "earlier candidate S1; finished before the reset of its job (02:02)",
    "r1/fp32_fq_part39_before_gpu_reset.csv": "rows written before the 10:52 reset; cross-check only",
}
LIVE = {"altF": "altF3", "alt_s7": "altF2", "alt_s1": "altF", "evalF": "evalF3", "altF_5c": "altF3s5"}
root = sys.argv[1]
live = "--live" in sys.argv
utc = dt.timezone.utc


def ts(s):
    return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=utc).timestamp()


jobs = []
for m in sorted(glob.glob(f"{root}/logs/*.meta")):
    txt = open(m).read()
    a = re.search(r"^(\S+Z) start (\S+) level=(\S*) tree=(\S+)", txt, re.M)
    e = re.findall(r"^(\S+Z) (?:end rc=\S+|GPU RESET DURING JOB)", txt, re.M)
    if not a:
        continue
    name = os.path.basename(m)[:-5]
    clk = []
    day = a.group(1)[:10]
    prev = ""
    for line in open(m[:-5] + ".clk") if os.path.exists(m[:-5] + ".clk") else []:
        c = re.match(r"(\d\d:\d\d:\d\d) sclk=\d+: (\d+)Mhz .* busy=(\d+)", line)
        if not c:
            continue
        if c.group(1) < prev:  # past midnight
            day = (dt.date.fromisoformat(day) + dt.timedelta(days=1)).isoformat()
        prev = c.group(1)
        clk.append((ts(f"{day}T{c.group(1)}Z"), int(c.group(2)), int(c.group(3))))
    jobs.append((ts(a.group(1)), ts(e[-1]) if e else float("inf"), name, a.group(3), a.group(4), clk))
jobs.sort(key=lambda x: x[0])
resets = [
    dt.datetime.strptime(x, "%a %b %d %H:%M:%S %Y").replace(tzinfo=utc).timestamp()
    for x in re.findall(r"^\[(.*?)\] .*GPU reset begin", open(f"{root}/logs/dmesg_gpu_resets.txt").read(), re.M)
]
if live:
    files = {}
    for d in "sweep alt altV cap evalA evalA_fix evalH r1 altF alt_s7 alt_s1 evalF altF_5c".split():
        for f in sorted(glob.glob(f"{root}/data/{LIVE.get(d, d)}/*.csv")):
            if not f.endswith(".raw.csv"):
                files[f"{d}/{os.path.basename(f)}"] = os.path.getmtime(f)
    if not os.path.exists(f"{root}/data/evalF3/fp32_auto.csv"):
        for k in ("fp32_auto.csv", "fp32_wmma.csv"):
            files[f"evalF/{k}"] = os.path.getmtime(f"{root}/data/evalF2/{k}")
else:
    files = {a: float(b) for a, b in (line.split() for line in open(f"{root}/logs/csv_mtimes.txt"))}
if "--mtimes" in sys.argv:
    for k, v in sorted(files.items()):
        print(k, f"{v:.3f}")
    sys.exit(0)
bad = 0
fmt = lambda t: dt.datetime.fromtimestamp(t, utc).strftime("%d %H:%M:%S")
print(
    f"{len(files)} CSVs, {len(jobs)} jobs, {len(resets)} GPU resets (dmesg times, about +-1 s against the file times)"
)
print(f"{'csv':52} {'last write (UTC)':16} {'job':22} {'tree':6} {'level':5} flags")
for k, t in sorted(files.items()):
    j = [x for x in jobs if x[0] <= t <= x[1] + 2]
    flags, note = "", ""
    if not j:
        flags = "N"
        j = [(0, 0, "-", "-", "-", [])]
    s, e, name, lvl, tree, clk = j[0]  # a file written at a job boundary belongs to the job that ended
    if name != "-":
        if lvl != "high":
            flags += "L"
        un = [c for c in clk if c[0] <= t and c[2] >= 50 and c[1] != 2599]
        if un:
            flags += "U"
            note += f" {len(un)} busy samples not at 2599 MHz;"
        before = [r for r in resets if s <= r <= t]
        after = [r for r in resets if t < r <= e + 2]
        if before:
            flags += "R"
            note += f" reset {fmt(before[0])[3:]} before the last write;"
        if after:
            flags += "r"
            note += f" finished {after[0] - t:.0f} s before the reset at {fmt(after[0])[3:]};"
    if flags:
        ok = ALLOWED.get(k)
        note += f" ALLOWED: {ok}" if ok else " NOT ALLOWED"
        bad += ok is None
    print(f"{k:52} {fmt(t):16} {name:22} {tree:6} {lvl:5} {flags or '-'}{note}")
print(f"{bad} flagged CSVs that are not in the allowed list")
sys.exit(1 if bad else 0)
