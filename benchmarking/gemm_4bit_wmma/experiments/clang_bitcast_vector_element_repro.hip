#include <hip/hip_runtime.h>
typedef unsigned int u4 __attribute__((ext_vector_type(4)));
// Expected: out[i] = bit pattern of element i as float, i.e. out = in.
__global__ void k(const u4* in, float* out) {
  u4 v = in[threadIdx.x];
  for (int e = 0; e < 4; e++) out[threadIdx.x * 4 + e] = __builtin_bit_cast(float, v[e]);
}
