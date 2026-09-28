#pragma once
// Launcher declarations for the AMD WMMA 4-bit GEMM kernel.
// T must be bnb_bfloat16, half or float (fp32 is emulated with split bf16 operands). Compiled on ROCm only; the kernel
// body is only emitted for RDNA3 (gfx110x) and RDNA4 (gfx120x) targets. Callers must check gemm_4bit_wmma_arch() and
// K % 64 == 0 before launching. absmax_8bit == nullptr selects the non-nested (standard) path.

#include <cstdint>

#include "compat.cuh" // CUDA/HIP runtime headers + bnb_stream_t

// WMMA generation of the current device: 11 (RDNA3, gfx110x), 12 (RDNA4, gfx120x), or 0 if the kernel does not
// support it.
int gemm_4bit_wmma_arch();

template <typename T>
void launch_gemm_4bit_wmma(
    const T* A, const uint8_t* B, const float* absmax, const uint8_t* absmax_8bit, const float* absmax_code,
    const float* absmax_offset, T* C, const T* bias, int M, int N, int K, int blocksize, int quant_type,
    bnb_stream_t stream
);
