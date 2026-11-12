"""Same-process alternating prescale/base timing with identical real NF4 allocations."""
import argparse,csv,math,os,statistics,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from bench import timed_rotation
from shapes import SHAPES
import bitsandbytes.functional as F
from bitsandbytes.backends.cuda.ops import _gemm_4bit_kernel_impl
p=argparse.ArgumentParser();p.add_argument("--shapes",required=True);p.add_argument("--out",required=True);p.add_argument("--dtype",default="fp16");p.add_argument("--ms",default="96,128,192,256");p.add_argument("--rounds",type=int,default=5);args=p.parse_args()
dt={"fp16":torch.float16,"bf16":torch.bfloat16}[args.dtype];rows=[];torch.manual_seed(0)
for si in map(int,args.shapes.split(",")):
 name,N,K=SHAPES[si];W=torch.randn(N,K,device="cuda",dtype=dt);q,s=F.quantize_4bit(W,blocksize=64,quant_type="nf4");del W
 nc=max(2,math.ceil(384e6/(q.numel()+4*s.absmax.numel())));copies=[(q.clone().view(-1,1),s.absmax.clone()) for _ in range(nc)]
 for M in map(int,args.ms.split(",")):
  A=torch.randn(M,K,device="cuda",dtype=dt);fns=[lambda b=b,a=a:_gemm_4bit_kernel_impl(A,b,(N,K),a,64,"nf4") for b,a in copies];times={"0":[],"1":[]}
  for it in range(args.rounds):
   for mode in (["0","1"] if it%2==0 else ["1","0"]):
    os.environ["BNB_WMMA_PRESCALE"]=mode
    for f in fns[:3]:f()
    times[mode].append(timed_rotation(fns,min(max(40,nc),200)))
  a,b=map(statistics.median,[times["0"],times["1"]]);row=dict(shape=name,N=N,K=K,M=M,dtype=args.dtype,base_us=a,variant_us=b,speedup=a/b,base_min=min(times["0"]),base_max=max(times["0"]),variant_min=min(times["1"]),variant_max=max(times["1"]));rows.append(row);print(row,flush=True)
 del copies;torch.cuda.empty_cache()
with open(args.out,"w",newline="") as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
