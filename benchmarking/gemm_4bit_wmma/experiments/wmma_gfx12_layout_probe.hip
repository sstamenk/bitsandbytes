// Probe the gfx12 (RDNA4) wave32 WMMA operand/accumulator layouts empirically.
// One wave per (A one-hot slot, B one-hot slot) pair: A has a single 1.0 at (lane_a, elem_a), B a
// single 1.0 at (lane_b, elem_b). D is non-zero at exactly one (lane, elem) iff both slots hold the
// same K index, giving: A slot -> (row, k), B slot -> (col, k), D slot -> (row, col).
// hipcc --offload-arch=gfx1201 -O2 wmma_gfx12_layout_probe.hip -o probe && ./probe
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>

typedef _Float16 v8h __attribute__((ext_vector_type(8)));
typedef short v8s __attribute__((ext_vector_type(8)));
typedef float v8f __attribute__((ext_vector_type(8)));

template <bool BF16> __global__ void probe(int* out) {
    const int pair = blockIdx.x; // a_slot * 256 + b_slot
    const int a_slot = pair >> 8, b_slot = pair & 255;
    const int lane = threadIdx.x;
    v8f c = {};
    v8f d;
    if constexpr (BF16) {
        v8s a = {}, b = {};
        for (int e = 0; e < 8; e++) {
            a[e] = (a_slot == lane * 8 + e) ? (short)0x3f80 : (short)0;
            b[e] = (b_slot == lane * 8 + e) ? (short)0x3f80 : (short)0;
        }
        d = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
    } else {
        v8h a = {}, b = {};
        for (int e = 0; e < 8; e++) {
            a[e] = (a_slot == lane * 8 + e) ? (_Float16)1 : (_Float16)0;
            b[e] = (b_slot == lane * 8 + e) ? (_Float16)1 : (_Float16)0;
        }
        d = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
    }
    for (int e = 0; e < 8; e++) {
        const float x = d[e];
        if (x != 0.0f)
            atomicAdd(&out[pair * 2], 1), out[pair * 2 + 1] = lane * 8 + e;
    }
}

template <bool BF16> int run() {
    int* d_out;
    hipMalloc(&d_out, 65536 * 2 * sizeof(int));
    hipMemset(d_out, 0, 65536 * 2 * sizeof(int));
    probe<BF16><<<65536, 32>>>(d_out);
    std::vector<int> h(65536 * 2);
    hipMemcpy(h.data(), d_out, h.size() * sizeof(int), hipMemcpyDeviceToHost);
    // Group A slots by the K they share with B slots; D slot tells row (from A) and col (from B).
    // Canonicalize: assign K ids by connected components of "a_slot pairs with b_slot".
    std::vector<int> a_k(256, -1), b_k(256, -1), a_row(256, -1), b_col(256, -1);
    std::vector<int> d_of(256 * 256, -1);
    int bad = 0, nk = 0;
    for (int p = 0; p < 65536; p++) {
        if (h[p * 2] > 1) bad++;
        if (h[p * 2] == 1) d_of[p] = h[p * 2 + 1];
    }
    // K id of a_slot = smallest b_slot it hits (bucket), row/col resolved via D position below.
    std::vector<int> kid_of_bmin(256, -1);
    for (int a = 0; a < 256; a++) {
        int bmin = -1;
        for (int b = 0; b < 256; b++) if (d_of[a * 256 + b] >= 0) { bmin = b; break; }
        if (bmin < 0) { printf("A slot %d hits nothing\n", a); bad++; continue; }
        if (kid_of_bmin[bmin] < 0) kid_of_bmin[bmin] = nk++;
        a_k[a] = kid_of_bmin[bmin];
    }
    for (int b = 0; b < 256; b++) {
        int amin = -1;
        for (int a = 0; a < 256; a++) if (d_of[a * 256 + b] >= 0) { amin = a; break; }
        if (amin >= 0) b_k[b] = a_k[amin];
    }
    printf("%s: %d distinct K ids, %d anomalies\n", BF16 ? "bf16" : "f16", nk, bad);
    // D slot -> (a_slot, b_slot) pairs: rows are the classes of A slots with same K that land on
    // different D lanes/elems. Print raw: for lane, elem of A: which D slots it can light up.
    printf("A slot (lane,elem) -> K id, and the set of D slots it lights (first 2):\n");
    for (int a = 0; a < 256; a++) {
        int ds[2] = {-1, -1}, n = 0;
        for (int b = 0; b < 256 && n < 2; b++) if (d_of[a * 256 + b] >= 0) ds[n++] = d_of[a * 256 + b];
        printf("  A L%02d e%d: k%02d D{L%02d e%d, L%02d e%d}\n", a / 8, a % 8, a_k[a], ds[0] / 8, ds[0] % 8, ds[1] / 8, ds[1] % 8);
    }
    printf("B slot (lane,elem) -> K id, and the set of D slots it lights (first 2):\n");
    for (int b = 0; b < 256; b++) {
        int ds[2] = {-1, -1}, n = 0;
        for (int a = 0; a < 256 && n < 2; a++) if (d_of[a * 256 + b] >= 0) ds[n++] = d_of[a * 256 + b];
        printf("  B L%02d e%d: k%02d D{L%02d e%d, L%02d e%d}\n", b / 8, b % 8, b_k[b], ds[0] / 8, ds[0] % 8, ds[1] / 8, ds[1] % 8);
    }
    hipFree(d_out);
    return bad;
}

int main() {
    run<false>();
    run<true>();
}
