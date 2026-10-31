#pragma once

// Warp-wide reduction intrinsics (__reduce_max_sync, __reduce_min_sync,
// __reduce_or_sync) are not declared by the CUDA 12.x device runtime headers,
// which is the toolkit lineage that still compiles sm_70. CUDA 13 provides
// them, so upstream code uses them freely. Provide drop-in equivalents built
// from __shfl_xor_sync butterflies (available since Volta).
//
//   reduce_max_sync(mask, s32) CUDA docs:
//     "32-bit signed integer value of val. All values must be in the range
//      [0, 2^31-1] or [-2^31, 0] for a given call."
// The sampling call sites pair a sign-bit split with nonnegative magnitudes,
// which satisfies that contract; the butterfly below matches exactly.

#include <cuda_runtime.h>
#include <cstdint>

namespace ninfer::ops {

__device__ __forceinline__ unsigned reduce_max_sync(unsigned mask, unsigned value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const unsigned other = __shfl_xor_sync(mask, value, offset);
        value                = other > value ? other : value;
    }
    return value;
}

__device__ __forceinline__ int reduce_max_sync(unsigned mask, int value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const int other = __shfl_xor_sync(mask, value, offset);
        value           = other > value ? other : value;
    }
    return value;
}

__device__ __forceinline__ int reduce_min_sync(unsigned mask, int value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const int other = __shfl_xor_sync(mask, value, offset);
        value           = other < value ? other : value;
    }
    return value;
}

__device__ __forceinline__ unsigned reduce_or_sync(unsigned mask, unsigned value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value |= __shfl_xor_sync(mask, value, offset);
    return value;
}

} // namespace ninfer::ops
