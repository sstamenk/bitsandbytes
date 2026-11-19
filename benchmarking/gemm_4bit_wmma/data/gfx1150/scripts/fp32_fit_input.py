"""fp32_fit_input.py sweep_fp32.csv sweep_fp32_mt16.csv > fit.csv: the input derive_cfg3.py / rules_eval.py used for fp32.

The main sweep plus the 16-row-tile supplement, without the supplement's kch = 2 rows of K % 128 != 0 shapes (forced
with --cfgs, they ran as kch = 1 and duplicate the kch = 1 rows) and without rows the main sweep already has."""

import csv
import sys

a = list(csv.DictReader(open(sys.argv[1])))
b = [r for r in csv.DictReader(open(sys.argv[2])) if not (int(r["K"]) % 128 and r["cfg"].endswith(",2"))]
seen = {(r["shape"], r["M"], r["cfg"]) for r in a}
w = csv.DictWriter(sys.stdout, fieldnames=list(a[0]))
w.writeheader()
w.writerows(a + [r for r in b if (r["shape"], r["M"], r["cfg"]) not in seen])
