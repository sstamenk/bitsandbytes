import os,torch
import bitsandbytes.functional as F
from bitsandbytes.backends.cuda.ops import _gemm_4bit_kernel_impl
for frac in [.01,.05,.1,.2,.5]:
 torch.manual_seed(71);M,N,K=96,4096,512
 scales=torch.rand(N,K//64,device="cuda")*15504+50000
 scales[torch.rand_like(scales)<frac]=1000
 W=(torch.rand(N,K,device="cuda")*2-1).reshape(N,K//64,64)*scales[:,:,None]
 W[:,:,0]=scales
 W=W.reshape(N,K).half();A=torch.randn(M,K,device="cuda").half()*1e-5
 q,s=F.quantize_4bit(W,blocksize=64,quant_type="nf4",compress_statistics=True)
 amax=F.dequantize_blockwise(s.absmax,s.state2)+s.offset
 outputs=[]
 for mode in ["0","1"]:
  os.environ["BNB_WMMA_PRESCALE"]=mode
  out=_gemm_4bit_kernel_impl(A,q.view(-1,1),(N,K),s.state2.absmax,64,"nf4",absmax_8bit=s.absmax,absmax_code=s.state2.code,absmax_offset=s.offset)
  outputs.append(out)
 finite=torch.isfinite(outputs[0])&torch.isfinite(outputs[1])
 print(frac,"decoded_scale_max",float(amax.max()),"base_nonfinite",int((~torch.isfinite(outputs[0])).sum()),"scaled_nonfinite",int((~torch.isfinite(outputs[1])).sum()),"finite_relative_difference",float((outputs[0].float()[finite]-outputs[1].float()[finite]).norm()/outputs[0].float()[finite].norm()),flush=True)
