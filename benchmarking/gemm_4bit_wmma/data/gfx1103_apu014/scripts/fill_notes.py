"""fill_notes.py <template> <out>: NOTES.md from stage_NOTES.md; the values that depend on the last runs are in
notes_values.py next to this file."""

import os
import sys

KEYS = """FP32CAP_384 FP32CAP_416 FP32CAP_448 FP32CAP_480 FP32CAP_512 FP32CAP_SRC
FP32CAP_TEXT FQ_CONSEQ RESET_COUNT RESET_LATER CLOCKS_TEXT HOOKS_TEXT AUDIT_TEXT""".split()
V = {}
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "notes_values.py")).read(), V)
s = open(sys.argv[1]).read()
for k in sorted(KEYS, key=len, reverse=True):
    assert k in s, k
    s = s.replace(k, V[k])
for k in KEYS:
    assert k not in s, k
open(sys.argv[2], "w").write(s)
