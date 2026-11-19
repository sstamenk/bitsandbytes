#!/bin/bash
# fitb.sh sweep.csv "1,2,4,8" [feature]: joint fit for one M bucket (all its swept M forced into one bucket)
python3 -c "
import csv,sys
ms=set(sys.argv[2].split(','))
r=csv.DictReader(open(sys.argv[1])); w=csv.DictWriter(open('/tmp/fitb.csv','w',newline=''),fieldnames=r.fieldnames); w.writeheader()
[w.writerow(x) for x in r if x['M'] in ms]" "$1" "$2"
python3 /home/master/develop/bnb-wmma-tune/gfx1102_work/derive_cfg3_safe.py /tmp/fitb.csv --score total --merge 100 --feature ${3:-N}
