#!/bin/bash
# q_evalF.sh <outdir> <mset> <dtypes> [kernels...]: bench of the tree job.sh runs in (cwd = its benchmarking dir).
# Shape 22 (phi3-mini down) is timed first as a warm-up and again in its place (dedup.py keeps the later rows);
# shape 0 (llama3-70b gate/up) is timed last. mset: dense (M = 1-512), short (1-64), hi (576-1024), cap (320-1024).
D=$1; MSET=$2; DTS=$3; shift 3; mkdir -p $D
SH=22,$(seq -s, 1 39),0
case $MSET in
  dense) MS=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,22,24,26,28,30,32,36,40,44,48,52,56,60,64,72,80,88,96,104,112,120,128,144,160,176,192,208,224,240,256,288,320,352,384,416,448,480,512;;
  short) MS=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,22,24,26,28,30,32,36,40,44,48,52,56,60,64;;
  hi) MS=576,640,704,768,896,1024;;
  *) MS=$MSET;;
esac
for dt in ${DTS//,/ }; do
  for k in "$@"; do
    [ -s $D/${dt}_$k.csv ] || { echo "$(date -u +%T) $dt $k"; rm -f $D/${dt}_$k.raw.csv; python bench.py --dtypes $dt --kernels $k --shapes $SH --ms $MS --out $D/${dt}_$k.raw.csv && python ~/bnbtune/bin/dedup.py $D/${dt}_$k.raw.csv $D/${dt}_$k.csv; }
  done
done
echo "$(date -u +%T) done"
