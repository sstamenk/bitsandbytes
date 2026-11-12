// Probe the gfx11 / gfx115x wave32 WMMA half-wave behaviour the kernel relies on
// (csrc/gemm_4bit_wmma.cu, "Half-wave WMMA on gfx11"):
//   half-wave h (lanes 16h..16h+15) computes output rows 2j+h (accumulator element j)
//   from its own B copy and from the A rows held by its lanes c with c % 2 == h.
// One wave per A slot (lane, elem): A is one-hot, B holds a unique id per slot. Every
// non-zero output is then exactly the id of the B slot it was paired with. The model:
//   A slot (16h + r, e), r % 2 == h  ->  D[lane 16h + c][elem (r-h)/2] = id(B lane 16h + c, elem e)
//   A slot with r % 2 != h            ->  contributes nothing.
// hipcc --offload-arch=gfx1151 -O2 wmma_gfx11_halfwave_probe.hip -o probe && ./probe
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>

typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef short v16s __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));

// B ids 1..512 are exact in fp16 (<= 2048). bf16 has 8 mantissa bits, so it uses id % 128 + 1
// (ids repeat across lanes; the checker compares against the expected slot's id).
template <bool BF16> __device__ __host__ inline int b_id(int slot) { return BF16 ? slot % 128 + 1 : slot + 1; }

template <bool BF16> __global__ void probe(float* out) {
    const int a_slot = blockIdx.x, lane = threadIdx.x;
    v8f c = {}, d;
    if constexpr (BF16) {
        v16s a, b;
        for (int e = 0; e < 16; e++) {
            a[e] = (a_slot == lane * 16 + e) ? (short)0x3f80 : (short)0;
            const float f = (float)b_id<true>(lane * 16 + e);
            b[e] = (short)(__float_as_uint(f) >> 16);
        }
        d = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
    } else {
        v16h a, b;
        for (int e = 0; e < 16; e++) {
            a[e] = (a_slot == lane * 16 + e) ? (_Float16)1 : (_Float16)0;
            b[e] = (_Float16)b_id<false>(lane * 16 + e);
        }
        d = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
    }
    for (int j = 0; j < 8; j++) out[(a_slot * 32 + lane) * 8 + j] = d[j];
}

template <bool BF16> int run(const char* arch) {
    float* d_out;
    const size_t n = 512 * 32 * 8;
    hipMalloc(&d_out, n * sizeof(float));
    probe<BF16><<<512, 32>>>(d_out);
    std::vector<float> h(n);
    hipMemcpy(h.data(), d_out, n * sizeof(float), hipMemcpyDeviceToHost);
    hipFree(d_out);
    int bad = 0;
    for (int a = 0; a < 512; a++) {
        const int la = a / 16, e = a % 16, hh = la / 16, r = la % 16;
        for (int L = 0; L < 32; L++)
            for (int j = 0; j < 8; j++) {
                float expect = 0;
                if (r % 2 == hh && L / 16 == hh && j == (r - hh) / 2) expect = (float)b_id<BF16>((16 * hh + L % 16) * 16 + e);
                const float got = h[(a * 32 + L) * 8 + j];
                if (got != expect && bad++ < 6)
                    printf("  A L%02d e%02d -> D L%02d e%d = %g, expected %g\n", la, e, L, j, got, expect);
            }
    }
    printf("%s %s: %d mismatching outputs (of %d) -> %s\n", arch, BF16 ? "bf16" : "f16 ", bad, (int)n,
           bad ? "MODEL DOES NOT HOLD" : "half-wave model holds");
    return bad;
}

int main() {
    hipDeviceProp_t p;
    hipGetDeviceProperties(&p, 0);
    return (run<false>(p.gcnArchName) + run<true>(p.gcnArchName)) != 0;
}
