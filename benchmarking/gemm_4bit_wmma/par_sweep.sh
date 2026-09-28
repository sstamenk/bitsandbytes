#!/bin/bash
# par_sweep.sh <tag> <gpu list> <shape list> [extra sweep args] : one sweep process per GPU, shapes round-robin
tag=$1; gpus=($2); IFS=',' read -ra shapes <<< "$3"; shift 3
here=$(cd "$(dirname "$0")" && pwd)
[ -n "$BNB_VENV" ] && source "$BNB_VENV/bin/activate"
export PYTHONPATH=${PYTHONPATH:-$(cd "$here/../.." && pwd)}
cd "$here"
pids=()
for gi in "${!gpus[@]}"; do
  sub=""
  for si in "${!shapes[@]}"; do (( si % ${#gpus[@]} == gi )) && sub="$sub,${shapes[$si]}"; done
  [ -z "$sub" ] && continue
  HIP_VISIBLE_DEVICES=${gpus[$gi]} BNB_GEMM4BIT_KERNEL=wmma python sweep.py --shapes ${sub#,} --out ${tag}_g${gpus[$gi]}.csv "$@" > ${tag}_g${gpus[$gi]}.log 2>&1 &
  pids+=($!)
done
wait "${pids[@]}"
python3 - "$tag" <<'PY'
import csv, glob, sys
rows = []
for f in sorted(glob.glob(f"{sys.argv[1]}_g*.csv")): rows += list(csv.DictReader(open(f)))
with open(f"{sys.argv[1]}.csv", "w", newline="") as out_f:
    w = csv.DictWriter(out_f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("merged", len(rows), "rows")
PY
