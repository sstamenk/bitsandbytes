#!/bin/bash
# fitbucket.sh sweep.csv "12,16" [feature]: joint 1-3 leaf fit over the given M values (derive_cfg3_safe, all merged)
f=$1; ms=$2; feat=${3:-auto}
tmp=$(mktemp); python3 -c "
import csv,sys
ms=set(map(int,'$ms'.split(',')))
r=csv.reader(open('$f')); w=csv.writer(sys.stdout); h=next(r); w.writerow(h)
i=h.index('M')
[w.writerow(x) for x in r if int(x[i]) in ms]" > $tmp
python3 derive_cfg3_safe.py $tmp --score total --merge 10 --feature $feat | grep -v "^    M=" ; rm $tmp
