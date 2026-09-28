#pragma once
// Launcher declarations for the AMD WMMA 4-bit GEMM kernel.
// T must be bnb_bfloat16, half or float (fp32 is emulated with split bf16 operands). Compiled on ROCm only; the kernel
// body is only emitted for RDNA3 (gfx110x) targets. Callers must check gemm_4bit_wmma_supported() and K % 64 == 0
// before launching. absmax_8bit == nullptr selects the non-nested (standard) path.

#include <cstdint>

#include "compat.cuh" // CUDA/HIP runtime headers + bnb_stream_t

// True if the current device can run the WMMA kernel (RDNA3, gfx110x).
bool gemm_4bit_wmma_supported();

template <typename T>
void launch_gemm_4bit_wmma(
    const T* A, const uint8_t* B, const float* absmax, const uint8_t* absmax_8bit, const float* absmax_code,
    const float* absmax_offset, T* C, const T* bias, int M, int N, int K, int blocksize, int quant_type,
    bnb_stream_t stream
);
