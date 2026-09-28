// AMD WMMA 4-bit GEMM kernel for RDNA3 (gfx110x, wave32).
// Computes C[M,N] = A[M,K] @ B[N,K]^T + bias with fused 4-bit dequantization.

#include <cstdint>
#include <cstring>
#include <type_traits>

#include "common.cuh"
#include "gemm_4bit_common.cuh"
#include "gemm_4bit_wmma.cuh"

// RDNA3.5 (gfx115x) shares the gfx11 WMMA encoding but has not been validated
// with the half-wave K split below, so it keeps the SIMT path for now.
#define BNB_WMMA_RDNA3 IS_RDNA3

namespace {

// K elements per warp per 64-K unit: each half-wave takes 32 (two 16-K WMMA subs).
// A main-loop iteration consumes KCH units, i.e. 64*KCH K per warp.
constexpr int WMMA_KC = 64;

// Native vector types: HIP's uint4 is a class, and copying it through private
// arrays lowers to memcpy that stays in scratch memory.
using wmma_v2u = uint32_t __attribute__((ext_vector_type(2)));
using wmma_v4u = uint32_t __attribute__((ext_vector_type(4)));
using wmma_v8f = float __attribute__((ext_vector_type(8)));
using wmma_v8u = uint32_t __attribute__((ext_vector_type(8)));
using wmma_v16h = _Float16 __attribute__((ext_vector_type(16)));
using wmma_v16s = short __attribute__((ext_vector_type(16)));

/// @brief One gfx11 v_wmma_f32_16x16x16_{f16,bf16} on 16-bit operand bits (bf16 for
/// T = bnb_bfloat16 and for the split fp32 path).
///
/// gfx11 executes a wave32 WMMA as two independent 8x16x16 products: lanes 0-15
/// produce the even output rows from their own B copy and the A rows held by the
/// even lanes, lanes 16-31 produce the odd rows from their own B copy and the odd
/// lanes' A rows. The ISA asks for replicated operands, but because each half only
/// reads its own copy, the kernel feeds the two halves different K slices and sums
/// the halves once at the end. Every weight nibble is decoded exactly once.
template <typename T> __device__ __forceinline__ wmma_v8f wmma_16x16x16(wmma_v8u a, wmma_v8u b, wmma_v8f c) {
#if BNB_WMMA_RDNA3
    if constexpr (std::is_same_v<T, half>) {
        return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
            __builtin_bit_cast(wmma_v16h, a), __builtin_bit_cast(wmma_v16h, b), c
        );
    } else {
        return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
            __builtin_bit_cast(wmma_v16s, a), __builtin_bit_cast(wmma_v16s, b), c
        );
    }
#else
    return c;
#endif
}

// fp32 operands are split into this many bf16 parts (~24 mantissa bits).
constexpr int WMMA_F32_PARTS = 3;

// Split an fp32 value into P bf16 parts (bit patterns in the low 16 bits) whose sum
// is x to ~2^-(8P) relative: every part but the last truncates, so each remainder
// is exact in fp32; the last rounds to nearest even. Inf/NaN keep the lower parts at
// zero so they propagate through the leading part alone.
template <int P> __device__ __forceinline__ void split_bf16(float x, uint32_t (&parts)[P]) {
    const bool finite = (__builtin_bit_cast(uint32_t, x) & 0x7f800000u) != 0x7f800000u;
    float r = x;
#pragma unroll
    for (int p = 0; p < P - 1; p++) {
        const uint32_t bits = __builtin_bit_cast(uint32_t, r);
        parts[p] = bits >> 16;
        r = r - __builtin_bit_cast(float, bits & 0xffff0000u);
    }
    const uint32_t bits = __builtin_bit_cast(uint32_t, r);
    parts[P - 1] = (bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16;
    if (!finite) {
#pragma unroll
        for (int p = 1; p < P; p++)
            parts[p] = 0u;
    }
}

// Add the value held by the lane 16 positions away (the other half-wave).
__device__ __forceinline__ float half_wave_sum(float v) {
#if BNB_WMMA_RDNA3
    const uint32_t other = __builtin_amdgcn_permlanex16(
        __builtin_bit_cast(uint32_t, v), __builtin_bit_cast(uint32_t, v), 0x76543210u, 0xfedcba98u, false, false
    );
    return v + __builtin_bit_cast(float, other);
#else
    return v;
#endif
}

// LDS bytes for the A stages (two stages of `planes` 16-bit tiles: 1, or 2 for the
// fp32 hi/lo split) or, reusing the same space, the split-K reduction tile,
// whichever is larger. The staged A rows are padded by 8 elements: keeps 16-byte
// alignment and spreads rows across LDS banks.
constexpr int wmma_smem_bytes(int mt, int nt, int ksplit, int kch, int planes) {
    const int a_bytes = 2 * planes * ksplit * mt * (WMMA_KC * kch + 8) * 2;
    const int red_bytes = ksplit > 1 ? ksplit * mt * nt * 4 : 0;
    return a_bytes > red_bytes ? a_bytes : red_bytes;
}

} // namespace

/// @brief Fused 4-bit dequantize + WMMA GEMM for RDNA3 (bf16, fp16 and fp32).
///
/// fp32 has no WMMA operand type, so A and the centroids are split into three bf16
/// parts (hi + mid + lo, ~24 mantissa bits) and each tile accumulates the six
/// products whose part indices sum to < 3 in fp32 -- six WMMAs per 16-bit one,
/// with fp32-level accuracy.
///
/// Block: KSPLIT x WARPS_M x WARPS_N warps; block tile MT = WARPS_M*8*WM rows x
/// NT = WARPS_N*16*WN columns. Each warp owns an (8*WM) x (16*WN) sub-tile; the
/// KSPLIT warp groups split K in interleaved KC = 64*KCH chunks and are reduced
/// through LDS in a fixed order (deterministic). Each WMMA covers 8 real rows x 16
/// columns x 32 K (16 K per half-wave).
///
/// Per iteration the block stages the A tile of its KSPLIT current chunks in LDS
/// (double-buffered, shared by all warps of a K group). Each lane (half h =
/// lane/16, c = lane%16) streams its own B columns straight from global memory:
///   B: column n, K [k0 + 32*KCH*h, k0 + 32*KCH*(h+1)) -> KCH 16-byte loads,
///      decoded through a 256-entry LDS LUT (byte -> packed pair of centroids).
///   A: row c/2 of each 8-row sub-tile, same K range, read from LDS.
///   Scale: absmax for that column/K range, applied to the fp32 accumulator after
///      the chunk's WMMAs (centroids enter the WMMA unscaled).
/// The next iteration's A and B loads are issued before the current chunk is
/// consumed; A is issued first so storing it to LDS never waits on the B loads.
///
/// Requires K % (64*KCH) == 0 and blocksize >= 32.
///
/// @tparam T       Input/output dtype (`bnb_bfloat16`, `half`, or `float`)
/// @tparam WM      8-row M sub-tiles per warp
/// @tparam WN      16-column N sub-tiles per warp
/// @tparam WARPS_M Warps along M
/// @tparam WARPS_N Warps along N
/// @tparam KSPLIT  Warp groups along K
/// @tparam KCH     64-K units per main-loop iteration
/// @tparam SG      32-K units sharing one absmax (1, or KCH when blocksize >= 32*KCH):
///                 each group's WMMAs chain into one fp32 tile that is scaled once.
/// @tparam RING    Weight-stream register slots (2 or 3): loads run RING-1 iterations ahead.
template <typename T, int WM, int WN, int WARPS_M, int WARPS_N, int KSPLIT, int KCH, int SG, int RING>
__global__ void __launch_bounds__(KSPLIT* WARPS_M* WARPS_N * 32) gemm_4bit_wmma(
    // clang-format off
    const T*       __restrict__ A,             // inputs [M, K]
    const uint8_t* __restrict__ B,             // packed 4-bit weights [N, K/2]
    const float*   __restrict__ absmax,        // fp32 absmax [N, K/blocksize] or
                                               //   [ceil(N*K/(blocksize*256))] when nested
    const uint8_t* __restrict__ absmax_8bit,   // [N, K/blocksize] uint8 compressed absmax;
                                               //   nullptr = non-nested
    const float*   __restrict__ absmax_code,   // [256] codebook for 8bit absmax
    const float*   __restrict__ absmax_offset, // scalar; nullptr = non-nested
    T*             __restrict__ C,             // [M, N]
    const T*       __restrict__ bias,          // [N] optional, nullptr = no bias
    int M, int N, int K,                       // problem shape
    int blocksize,                             // elements per quantization block
    int quant_type                             // 1 = FP4, 2 = NF4
    // clang-format on
) {
#if BNB_WMMA_RDNA3
    constexpr int MT = WARPS_M * 8 * WM;
    constexpr int NT = WARPS_N * 16 * WN;
    constexpr int NUM_THREADS = KSPLIT * WARPS_M * WARPS_N * 32;
    constexpr int KC = WMMA_KC * KCH; // K per warp per iteration
    constexpr int A_STRIDE = KC + 8;  // LDS row stride (elements)
    constexpr bool F32 = std::is_same_v<T, float>;
    constexpr int PLANES = F32 ? WMMA_F32_PARTS : 1; // fp32: one bf16 tile per split part
    constexpr int A_PLANE = KSPLIT * MT * A_STRIDE;  // elements per A tile
    constexpr int A_STAGE = PLANES * A_PLANE;        // elements per A stage

    // byte -> {centroid[hi nibble] (lower K), centroid[lo nibble] (higher K)} as
    // 16-bit operand bits; fp32 keeps one table per bf16 split part.
    __shared__ uint32_t lut[PLANES][256];
    // Codebook for the nested (8-bit) absmax, so the per-chunk scale needs no
    // dependent global load.
    __shared__ float code_lds[256];
    // A stages [2][KSPLIT][MT][A_STRIDE]; reused for the split-K reduction.
    __shared__ __attribute__((aligned(16))) uint8_t smem[wmma_smem_bytes(MT, NT, KSPLIT, KCH, PLANES)];
    uint16_t* a_lds = reinterpret_cast<uint16_t*>(smem);

    const bool nested = absmax_8bit != nullptr;
    {
        const float* centroids = (quant_type == 1) ? FP4_LUT_F32 : NF4_LUT_F32;
        for (int i = threadIdx.x; i < 256; i += NUM_THREADS) {
            if constexpr (F32) {
                uint32_t c0[PLANES], c1[PLANES];
                split_bf16<PLANES>(centroids[i >> 4], c0);
                split_bf16<PLANES>(centroids[i & 0x0f], c1);
#pragma unroll
                for (int p = 0; p < PLANES; p++)
                    lut[p][i] = c0[p] | (c1[p] << 16);
            } else {
                const T lo = static_cast<T>(centroids[i >> 4]);
                const T hi = static_cast<T>(centroids[i & 0x0f]);
                lut[0][i] = (uint32_t)(*reinterpret_cast<const uint16_t*>(&lo)) |
                            ((uint32_t)(*reinterpret_cast<const uint16_t*>(&hi)) << 16);
            }
            if (nested)
                code_lds[i] = __ldg(&absmax_code[i]);
        }
    }

    const float absmax_offset_f = nested ? __ldg(absmax_offset) : 0.0f;
    const int blocksize_log2 = __ffs(blocksize) - 1;
    const int blocks_per_row = K >> blocksize_log2;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warp_n = warp % WARPS_N;
    const int warp_m = (warp / WARPS_N) % WARPS_M;
    const int warp_k = warp / (WARPS_N * WARPS_M);
    const int half_id = lane >> 4;
    const int col = lane & 15;

    const int bn = blockIdx.x * NT;
    const int bm = blockIdx.y * MT;

    // Per-lane weight rows (clamped so tail lanes read valid memory; never stored).
    const uint8_t* b_row[WN];
    int blk_row[WN];
#pragma unroll
    for (int wn = 0; wn < WN; wn++) {
        const int n = min(bn + (warp_n * WN + wn) * 16 + col, N - 1);
        b_row[wn] = B + (size_t)n * (K / 2);
        blk_row[wn] = n * blocks_per_row;
    }

    const int num_chunks = K / KC;
    const int num_iters = (num_chunks + KSPLIT - 1) / KSPLIT;

    // Blocks start at different K chunks so that concurrently resident blocks
    // stream different K offsets of their rows. With power-of-two row strides
    // (K/2 bytes), lock-stepped blocks otherwise camp on a few DRAM channels.
    const int chunk_rot = blockIdx.x % num_chunks;
    // First K element of warp group g's chunk at iteration i. Past-the-end chunks
    // (last iteration when KSPLIT does not divide num_chunks) alias a valid chunk:
    // they are loaded but never consumed, which keeps every load unconditional.
    auto chunk_valid = [&](int i, int g) { return i * KSPLIT + g < num_chunks; };
    auto chunk_k0 = [&](int i, int g) {
        int ch = min(i * KSPLIT + g, num_chunks - 1) + chunk_rot;
        ch -= ch >= num_chunks ? num_chunks : 0;
        return ch * KC;
    };

    // Cooperative A tile copy: KSPLIT x MT rows x KC/VEC_ELEMS 16-byte vectors per
    // stage (fp32 vectors are split into the hi/lo tiles on the way to LDS). Rows
    // past M are clamped to a valid row: WMMA output rows only depend on their own A
    // row, and those rows are never stored.
    constexpr int VEC_ELEMS = 16 / (int)sizeof(T);
    constexpr int ROW_VECS = KC / VEC_ELEMS;
    constexpr int A_VECS = KSPLIT * MT * ROW_VECS;
    constexpr int A_VECS_PER_THREAD = (A_VECS + NUM_THREADS - 1) / NUM_THREADS;
    wmma_v4u a_stage[A_VECS_PER_THREAD];
    auto issue_a = [&](int i) {
#pragma clang loop unroll(full)
        for (int j = 0; j < A_VECS_PER_THREAD; j++) {
            const int v = threadIdx.x + j * NUM_THREADS;
            if (A_VECS % NUM_THREADS == 0 || v < A_VECS) {
                const int part = v % ROW_VECS;
                const int row = (v / ROW_VECS) % MT;
                const int g = (v / ROW_VECS) / MT;
                a_stage[j] = *reinterpret_cast<const wmma_v4u*>(
                    A + (size_t)min(bm + row, M - 1) * K + chunk_k0(i, g) + part * VEC_ELEMS
                );
            }
        }
    };
    auto store_a = [&](int buf) {
#pragma clang loop unroll(full)
        for (int j = 0; j < A_VECS_PER_THREAD; j++) {
            const int v = threadIdx.x + j * NUM_THREADS;
            if (A_VECS % NUM_THREADS == 0 || v < A_VECS) {
                const int part = v % ROW_VECS;
                const int rg = v / ROW_VECS; // g * MT + row
                uint16_t* dst = &a_lds[buf * A_STAGE + rg * A_STRIDE + part * VEC_ELEMS];
                if constexpr (F32) {
                    uint32_t parts[4][PLANES];
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        // Copy the element out first: __builtin_bit_cast on a vector element
                        // lvalue reads element 0 with this clang.
                        const uint32_t bits = a_stage[j][e];
                        split_bf16<PLANES>(__builtin_bit_cast(float, bits), parts[e]);
                    }
#pragma unroll
                    for (int p = 0; p < PLANES; p++)
                        *reinterpret_cast<wmma_v2u*>(dst + p * A_PLANE) =
                            wmma_v2u{parts[0][p] | (parts[1][p] << 16), parts[2][p] | (parts[3][p] << 16)};
                } else {
                    *reinterpret_cast<wmma_v4u*>(dst) = a_stage[j];
                }
            }
        }
    };

    // This warp's B columns for one iteration: KCH 32-K units per column, one
    // absmax per SG units.
    constexpr int NG = KCH / SG; // scale groups per iteration

    struct BStage {
        wmma_v4u b[WN][KCH];
        float scale[WN][NG];   // absmax, or the fp32 state2 absmax when nested
        uint32_t qidx[WN][NG]; // nested: 8-bit absmax code index
    };

    auto issue_b = [&](int i, BStage& st) {
        const int k_half = chunk_k0(i, warp_k) + half_id * 32 * KCH;
#pragma unroll
        for (int wn = 0; wn < WN; wn++) {
#pragma unroll
            for (int u = 0; u < KCH; u++)
                st.b[wn][u] = *reinterpret_cast<const wmma_v4u*>(b_row[wn] + (k_half + u * 32) / 2);
#pragma unroll
            for (int g = 0; g < NG; g++) {
                const int blk = blk_row[wn] + ((k_half + g * SG * 32) >> blocksize_log2);
                st.qidx[wn][g] = nested ? absmax_8bit[blk] : 0u;
                st.scale[wn][g] = __ldg(&absmax[nested ? blk >> 8 : blk]);
            }
        }
    };

    wmma_v8f acc[WM][WN];
#pragma unroll
    for (int wm = 0; wm < WM; wm++)
#pragma unroll
        for (int wn = 0; wn < WN; wn++)
            acc[wm][wn] = wmma_v8f{};

    auto compute = [&](int buf, const BStage& st) {
        const uint16_t* a_base =
            a_lds + buf * A_STAGE + (warp_k * MT + warp_m * 8 * WM + (col >> 1)) * A_STRIDE + half_id * 32 * KCH;
#pragma unroll
        for (int g = 0; g < NG; g++) {
            float scale[WN];
#pragma unroll
            for (int wn = 0; wn < WN; wn++)
                scale[wn] = nested ? code_lds[st.qidx[wn][g]] * st.scale[wn][g] + absmax_offset_f : st.scale[wn][g];

            // Issue order keeps WM*WN independent WMMA chains in flight; the group's
            // 2*SG WMMAs per chain accumulate unscaled and are scaled once. A
            // fragments stream from LDS one 8-row sub-tile at a time.
            wmma_v8f tmp[WM][WN];
#pragma unroll
            for (int u = g * SG; u < (g + 1) * SG; u++) {
                wmma_v8u b_frag[PLANES][WN][2];
#pragma unroll
                for (int wn = 0; wn < WN; wn++)
#pragma unroll
                    for (int s = 0; s < 2; s++)
#pragma unroll
                        for (int i = 0; i < 8; i++) {
                            const int byte = (st.b[wn][u][s * 2 + i / 4] >> (8 * (i % 4))) & 0xff;
#pragma unroll
                            for (int p = 0; p < PLANES; p++)
                                b_frag[p][wn][s][i] = lut[p][byte];
                        }
#pragma unroll
                for (int s = 0; s < 2; s++)
#pragma unroll
                    for (int wm = 0; wm < WM; wm++) {
                        const uint16_t* ap = a_base + wm * 8 * A_STRIDE + u * 32 + s * 16;
                        const wmma_v8u a_frag = *reinterpret_cast<const wmma_v8u*>(ap);
                        const bool first = u == g * SG && s == 0;
#pragma unroll
                        for (int wn = 0; wn < WN; wn++) {
                            wmma_v8f c = first ? wmma_v8f{} : tmp[wm][wn];
                            if constexpr (F32) {
                                // Products of split parts whose indices sum to < 3, smallest
                                // first; the leading hi*hi product comes last (below).
                                const wmma_v8u a_mid = *reinterpret_cast<const wmma_v8u*>(ap + A_PLANE);
                                const wmma_v8u a_lo = *reinterpret_cast<const wmma_v8u*>(ap + 2 * A_PLANE);
                                c = wmma_16x16x16<T>(a_lo, b_frag[0][wn][s], c);
                                c = wmma_16x16x16<T>(a_frag, b_frag[2][wn][s], c);
                                c = wmma_16x16x16<T>(a_mid, b_frag[1][wn][s], c);
                                c = wmma_16x16x16<T>(a_mid, b_frag[0][wn][s], c);
                                c = wmma_16x16x16<T>(a_frag, b_frag[1][wn][s], c);
                            }
                            tmp[wm][wn] = wmma_16x16x16<T>(a_frag, b_frag[0][wn][s], c);
                        }
                    }
            }
#pragma unroll
            for (int wm = 0; wm < WM; wm++)
#pragma unroll
                for (int wn = 0; wn < WN; wn++)
                    acc[wm][wn] += tmp[wm][wn] * scale[wn];
        }
    };

    // The per-warp weight stream runs RING-1 iterations ahead in a register ring.
    // The loop is unrolled by RING so slots are addressed statically: copying a
    // prefetched slot would force a wait on its (still in-flight) loads. A stays
    // double-buffered in LDS and is issued before the weights, so storing it never
    // waits on the newer weight loads.
    static_assert(RING == 2 || RING == 3, "RING must be 2 or 3");
    BStage b_ring[RING];
    issue_a(0);
#pragma unroll
    for (int r = 0; r < RING - 1; r++)
        if (r < num_iters)
            issue_b(r, b_ring[r]);
    store_a(0);
    __syncthreads();

    auto step = [&](int it, const BStage& cur, BStage& ahead) {
        const bool has_next = it + 1 < num_iters;
        if (has_next)
            issue_a(it + 1);
        if (it + RING - 1 < num_iters)
            issue_b(it + RING - 1, ahead);
        if (chunk_valid(it, warp_k))
            compute(it & 1, cur);
        if (has_next)
            store_a((it + 1) & 1);
        __syncthreads();
    };
    for (int it = 0; it < num_iters; it += RING) {
#pragma unroll
        for (int r = 0; r < RING; r++)
            if (it + r < num_iters)
                step(it + r, b_ring[r], b_ring[(r + RING - 1) % RING]);
    }

    // Combine the two half-wave K slices: every lane now holds full-K sums for
    // rows j (element j) and column `col`.
#pragma unroll
    for (int wm = 0; wm < WM; wm++)
#pragma unroll
        for (int wn = 0; wn < WN; wn++)
#pragma unroll
            for (int j = 0; j < 8; j++)
                acc[wm][wn][j] = half_wave_sum(acc[wm][wn][j]);

    const int m_warp = warp_m * 8 * WM; // warp's first row within the block tile
    const int n_warp = warp_n * 16 * WN;
    if constexpr (KSPLIT == 1) {
        if (half_id == 0) {
#pragma unroll
            for (int wn = 0; wn < WN; wn++) {
                const int n = bn + n_warp + wn * 16 + col;
                if (n >= N)
                    continue;
                const float bias_f = bias ? static_cast<float>(bias[n]) : 0.0f;
#pragma unroll
                for (int wm = 0; wm < WM; wm++)
#pragma unroll
                    for (int j = 0; j < 8; j++) {
                        const int m = bm + m_warp + wm * 8 + j;
                        if (m < M)
                            C[(size_t)m * N + n] = static_cast<T>(acc[wm][wn][j] + bias_f);
                    }
            }
        }
    } else {
        // Partial tiles [KSPLIT][MT][NT] in LDS (the A stages are no longer needed;
        // the loop's trailing barrier ordered all reads before these writes).
        float* red = reinterpret_cast<float*>(smem);
        if (half_id == 0) {
#pragma unroll
            for (int wm = 0; wm < WM; wm++)
#pragma unroll
                for (int wn = 0; wn < WN; wn++)
#pragma unroll
                    for (int j = 0; j < 8; j++)
                        red[(warp_k * MT + m_warp + wm * 8 + j) * NT + n_warp + wn * 16 + col] = acc[wm][wn][j];
        }
        __syncthreads();

        for (int o = threadIdx.x; o < MT * NT; o += NUM_THREADS) {
            const int m = bm + o / NT;
            const int n = bn + o % NT;
            if (m >= M || n >= N)
                continue;
            float sum = 0.0f;
#pragma unroll
            for (int g = 0; g < KSPLIT; g++)
                sum += red[g * MT * NT + o];
            if (bias)
                sum += static_cast<float>(bias[n]);
            C[(size_t)m * N + n] = static_cast<T>(sum);
        }
    }
#endif // BNB_WMMA_RDNA3
}

namespace {

// Cached per device: whether it is an RDNA3 (gfx110x) GPU.
bool wmma_device_supported() {
    static int cache[16] = {}; // 0 = unknown, 1 = supported, 2 = not supported
    int dev = 0;
    (void)hipGetDevice(&dev);
    if (dev < 16 && cache[dev] != 0)
        return cache[dev] == 1;
    hipDeviceProp_t prop;
    const bool supported =
        hipGetDeviceProperties(&prop, dev) == hipSuccess && std::strncmp(prop.gcnArchName, "gfx110", 6) == 0;
    if (dev < 16)
        cache[dev] = supported ? 1 : 2;
    return supported;
}

/// Launch one tile configuration. KCH_MAX is lowered to 1 when K is not a multiple
/// of 128; one absmax covers each half-wave's per-iteration K slice when
/// blocksize >= 32 * KCH.
template <typename T, int WM, int WN, int WARPS_M, int WARPS_N, int KSPLIT, int KCH_MAX>
void launch_wmma_tile(
    const T* A, const uint8_t* B, const float* absmax, const uint8_t* absmax_8bit, const float* absmax_code,
    const float* absmax_offset, T* C, const T* bias, int M, int N, int K, int blocksize, int quant_type,
    bnb_stream_t stream
) {
    constexpr int MT = WARPS_M * 8 * WM;
    constexpr int NT = WARPS_N * 16 * WN;
    constexpr int THREADS = KSPLIT * WARPS_M * WARPS_N * 32;
    static_assert(
        wmma_smem_bytes(MT, NT, KSPLIT, KCH_MAX, std::is_same_v<T, float> ? WMMA_F32_PARTS : 1) <= 48 * 1024,
        "tile exceeds the LDS budget"
    );
    const dim3 grid((N + NT - 1) / NT, (M + MT - 1) / MT);
    auto launch = [&](auto kernel) {
        kernel<<<grid, THREADS, 0, stream>>>(
            A, B, absmax, absmax_8bit, absmax_code, absmax_offset, C, bias, M, N, K, blocksize, quant_type
        );
    };
    // Two iterations of weight prefetch, except for the largest warp tiles where the
    // extra register slot would spill.
    constexpr int RING = WM * WN >= 8 ? 2 : 3;
    if constexpr (KCH_MAX == 2) {
        if (K % 128 == 0) {
            if (blocksize >= 64)
                launch(gemm_4bit_wmma<T, WM, WN, WARPS_M, WARPS_N, KSPLIT, 2, 2, RING>);
            else
                launch(gemm_4bit_wmma<T, WM, WN, WARPS_M, WARPS_N, KSPLIT, 2, 1, RING>);
            return;
        }
    }
    launch(gemm_4bit_wmma<T, WM, WN, WARPS_M, WARPS_N, KSPLIT, 1, 1, RING>);
}

} // namespace

bool gemm_4bit_wmma_supported() { return wmma_device_supported(); }

/// @brief Host launcher for the RDNA3 WMMA 4-bit GEMM kernel.
///
/// Tile selection was calibrated on gfx1100 (W7900) over 40 LLM linear-layer
/// shapes: one tile per M bucket and N side, within ~2-7% (geomean) of the best
/// swept tile per shape. Narrow N keeps 16-column warp tiles and deeper split-K
/// so enough warps stream B; wide N uses 32-column warp tiles and 128-K iterations.
/// @tparam T Input/output dtype (`bnb_bfloat16` or `half`)
template <typename T>
void launch_gemm_4bit_wmma(
    // clang-format off
    const T*       A,             // inputs [M, K]
    const uint8_t* B,             // packed 4-bit weights [N, K/2]
    const float*   absmax,        // fp32 absmax [N*K/blocksize] or [ceil(N*K/(blocksize*256))] when nested
    const uint8_t* absmax_8bit,   // [N*K/blocksize] uint8 compressed absmax; nullptr = non-nested
    const float*   absmax_code,   // [256] codebook for 8bit absmax
    const float*   absmax_offset, // scalar; nullptr = non-nested
    T*             C,             // output [M, N]
    const T*       bias,          // [N] optional, nullptr = no bias
    int M, int N, int K,          // problem shape
    int blocksize,                // elements per quantization block
    int quant_type,               // 1 = FP4, 2 = NF4
    bnb_stream_t stream           // HIP stream
    // clang-format on
) {
    // clang-format off
#define LAUNCH_WMMA(WM, WN, WARPS_M, WARPS_N, KSPLIT, KCH_MAX)                                           \
    launch_wmma_tile<T, WM, WN, WARPS_M, WARPS_N, KSPLIT, KCH_MAX>(                                      \
        A, B, absmax, absmax_8bit, absmax_code, absmax_offset, C, bias, M, N, K, blocksize, quant_type, stream)


    if constexpr (std::is_same_v<T, float>) {
        // fp32 (three LDS A tiles per stage): calibrated separately, within ~2-4%
        // (geomean) of the best swept tile per shape.
        if (M <= 8) {
            if (N < 5120) LAUNCH_WMMA(1, 1, 1, 2, 4, 1);
            else          LAUNCH_WMMA(1, 2, 1, 2, 2, 2);
        } else if (M <= 16) {
            if (N < 5120) LAUNCH_WMMA(1, 1, 1, 2, 4, 1);
            else          LAUNCH_WMMA(2, 2, 1, 2, 2, 1);
        } else if (M <= 32) {
            if (N < 2560) LAUNCH_WMMA(1, 2, 1, 2, 2, 2);
            else          LAUNCH_WMMA(2, 2, 1, 2, 2, 1);
        } else {
            LAUNCH_WMMA(4, 1, 1, 4, 1, 1);
        }
        return;
    } else
    if (M <= 8) {
        if (N < 4608) LAUNCH_WMMA(1, 1, 1, 1, 8, 1);
        else          LAUNCH_WMMA(1, 2, 1, 1, 4, 2);
    } else if (M <= 16) {
        if (N < 5120) LAUNCH_WMMA(2, 1, 1, 2, 8, 1);
        else          LAUNCH_WMMA(2, 2, 1, 2, 4, 2);
    } else if (M <= 32) {
        if (N < 4096) LAUNCH_WMMA(2, 1, 1, 2, 8, 1);
        else          LAUNCH_WMMA(4, 1, 1, 4, 2, 2);
    } else if (M <= 64) {
        if (N < 3072) LAUNCH_WMMA(4, 1, 1, 2, 4, 1);
        else          LAUNCH_WMMA(4, 1, 1, 4, 2, 2);
    } else if (M <= 128) {
        if (N < 4096) LAUNCH_WMMA(4, 1, 1, 4, 2, 1);
        else          LAUNCH_WMMA(4, 2, 1, 4, 1, 2);
    } else {
        if (N < 3072) LAUNCH_WMMA(4, 2, 1, 2, 2, 1);
        else          LAUNCH_WMMA(4, 2, 1, 4, 1, 2);
    }
#undef LAUNCH_WMMA
    // clang-format on
}

template void launch_gemm_4bit_wmma<bnb_bfloat16>(
    const bnb_bfloat16*, const uint8_t*, const float*, const uint8_t*, const float*, const float*, bnb_bfloat16*,
    const bnb_bfloat16*, int, int, int, int, int, bnb_stream_t
);
template void launch_gemm_4bit_wmma<half>(
    const half*, const uint8_t*, const float*, const uint8_t*, const float*, const float*, half*, const half*, int, int,
    int, int, int, bnb_stream_t
);
template void launch_gemm_4bit_wmma<float>(
    const float*, const uint8_t*, const float*, const uint8_t*, const float*, const float*, float*, const float*, int,
    int, int, int, int, bnb_stream_t
);
