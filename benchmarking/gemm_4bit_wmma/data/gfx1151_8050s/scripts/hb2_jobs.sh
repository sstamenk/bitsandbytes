#!/bin/bash
cd ~/bnbtune8050
bash run8060.sh alt60_fp16 python alt.py --tables A=f16,B=rulesB.py,J2=rulesJ2.py --dtype fp16 --ms 17,24,40,48,96,128,192,256,384,512 --cycles 5 --out ~/bnbtune8050/data/alt60_fp16.csv
bash run8060.sh alt60_bf16 python alt.py --tables A=f16,B=rulesB.py,J2=rulesJ2.py --dtype bf16 --ms 17,24,40,48,96,128,192,256 --cycles 3 --out ~/bnbtune8050/data/alt60_bf16.csv
echo ALLDONE >> ~/bnbtune8050/logs/jobs.done
