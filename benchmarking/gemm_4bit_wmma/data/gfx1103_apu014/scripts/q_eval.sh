#!/bin/bash
# q_eval.sh <outdir> [kernels...]: dense-M bench of the tree job.sh runs in (cwd = its benchmarking dir).
# Shape 0 (llama3-70b gate/up) is timed last in every worker (first-in-process high readings, report_gfx1102.md).
D=$1; shift; mkdir -p $D
SH=$(seq -s, 1 39),0
MS=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,22,24,26,28,30,32,36,40,44,48,52,56,60,64,72,80,88,96,104,112,120,128,144,160,176,192,208,224,240,256,288,320,352,384,416,448,480,512
MSS=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,22,24,26,28,30,32,36,40,44,48,52,56,60,64
for dt in fp16 bf16 fp32; do
  for k in "$@"; do
    ms=$MS; [ $k = simt ] && ms=$MSS; [ $k = wmma ] && [ $dt != fp32 ] && continue; [ $k = wmma ] && ms=$MSS
    [ -s $D/${dt}_$k.csv ] || { echo "$(date -u +%T) $dt $k"; python bench.py --dtypes $dt --kernels $k --shapes $SH --ms $ms --out $D/${dt}_$k.csv; }
  done
done
echo "$(date -u +%T) done"
