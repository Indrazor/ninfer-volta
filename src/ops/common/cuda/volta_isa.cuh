#pragma once

// Volta (sm_70) emulation tier for the Ampere+ tensor-core ISA used by the common
// mma/ldmatrix hub. Upstream kernels are written against the Ampere warp-tile
// contract (m16n8k16 bf16/f16, m16n8k32 s8/fp8, m16n8k64 e2m1, m16n8k8 tf32);
// Volta has none of these instructions natively (only m8n8k4 f16 HMMA). This
// header reproduces the *exact per-lane fragment semantics* of the PTX
// instructions with shuffles + FFMA/IMAD so whole kernels compile and produce
// correct results on sm_70. Throughput is not the goal here: it is the
// correctness floor that lets the v100 target link. Hot paths are replaced by
// Volta-native transplanted kernels (see PORTING.md).
//
// Fragment layouts follow the PTX ISA for mma.sync.aligned .m16n8kN .row.col
// (groupID = lane>>2, tid = lane&3):
//   C: c0(c1) row groupID (groupID+8), col 2*tid (2*tid+1)
//   A(f16 k16): a0 (groupID, 2t..2t+1)  a1 (groupID+8, 2t..2t+1)
//               a2 (groupID, 2t+8..)    a3 (groupID+8, 2t+8..)
//   B(k16):     b0 (col=groupID, k=2t..2t+1)  b1 (col=groupID, k=2t+8..)
//   k32 (s8/fp8): a0 (groupID, 4t..4t+3), a1 row+8, a2 k+16, a3 row+8 k+16;
//                 b0 (col=groupID, 4t..4t+3), b1 +16.
//   k64 (e2m1): a0 (groupID, 8t..8t+7) ... a2 k+32; b0 (col=groupID, 8t..+7), b1 +32.
//   k8 (tf32):  a0 (groupID, t) a1 (groupID+8, t) a2 (groupID, t+4) a3 (row+8, t+4);
//               b0 (col=groupID, k=t) b1 (col, t+4).
// The shared identity: for output column j the needed B elements for a lane's
// own k live in the B-owning lane (j*4 + tid), i.e. s0 = (2*tid)*4 + tid for
// col 2*tid and s1 = s0 + 1 for col 2*tid+1.
//
// ldmatrix is reproduced with ld.shared: address inputs are shared::cta .b32
// values produced by smem_addr(), exactly as PTX consumes them.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace ninfer::ops::common::volta {

static __device__ __noinline__ std::uint32_t ld_shared_u32(std::uint32_t addr) {
    std::uint32_t v;
    asm("ld.shared.b32 %0, [%1];" : "=r"(v) : "r"(addr));
    return v;
}
static __device__ __noinline__ std::uint16_t ld_shared_u16(std::uint32_t addr) {
    std::uint16_t v;
    asm("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(addr));
    return v;
}

// ---------------- ldmatrix ----------------

static __device__ __noinline__ void ldmatrix_x2(std::uint32_t& r0, std::uint32_t& r1,
                                            std::uint32_t addr) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned rr   = lane >> 2;
    // Addresses are supplied by one lane per row of each matrix: matrix 0 rows
    // come from lanes 0..7, matrix 1 from lanes 8..15.
    r0 = ld_shared_u32(__shfl_sync(0xffffffffu, addr, static_cast<int>(rr)) + (lane & 3u) * 4u);
    r1 = ld_shared_u32(__shfl_sync(0xffffffffu, addr, static_cast<int>(8u + rr)) + (lane & 3u) * 4u);
}

static __device__ __noinline__ void ldmatrix_x4(std::uint32_t& r0, std::uint32_t& r1, std::uint32_t& r2,
                                            std::uint32_t& r3, std::uint32_t addr) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned rr   = lane >> 2;
    const unsigned cp   = (lane & 3u) * 4u;
    r0 = ld_shared_u32(__shfl_sync(0xffffffffu, addr, static_cast<int>(rr)) + cp);
    r1 = ld_shared_u32(__shfl_sync(0xffffffffu, addr, static_cast<int>(8u + rr)) + cp);
    r2 = ld_shared_u32(__shfl_sync(0xffffffffu, addr, static_cast<int>(16u + rr)) + cp);
    r3 = ld_shared_u32(__shfl_sync(0xffffffffu, addr, static_cast<int>(24u + rr)) + cp);
}

static __device__ __noinline__ std::uint32_t ldmatrix_trans_one(std::uint32_t addr, unsigned lane,
                                                            unsigned base) {
    // Transposed distribution: lane holds source rows (lane%4)*2 and +1 at
    // column lane/4; the two rows' addresses come from lanes base+(lane%4)*2, +1.
    const unsigned r0 = (lane & 3u) * 2u;
    const unsigned c  = (lane >> 2) * 2u;
    const std::uint16_t lo = ld_shared_u16(
        __shfl_sync(0xffffffffu, addr, static_cast<int>(base + r0)) + c);
    const std::uint16_t hi = ld_shared_u16(
        __shfl_sync(0xffffffffu, addr, static_cast<int>(base + r0 + 1u)) + c);
    return (static_cast<std::uint32_t>(hi) << 16) | lo;
}

static __device__ __noinline__ void ldmatrix_x2_trans(std::uint32_t& r0, std::uint32_t& r1,
                                                  std::uint32_t addr) {
    const unsigned lane = threadIdx.x & 31u;
    r0 = ldmatrix_trans_one(addr, lane, 0u);
    r1 = ldmatrix_trans_one(addr, lane, 8u);
}

static __device__ __noinline__ void ldmatrix_x4_trans(std::uint32_t& r0, std::uint32_t& r1,
                                                  std::uint32_t& r2, std::uint32_t& r3,
                                                  std::uint32_t addr) {
    const unsigned lane = threadIdx.x & 31u;
    r0 = ldmatrix_trans_one(addr, lane, 0u);
    r1 = ldmatrix_trans_one(addr, lane, 8u);
    r2 = ldmatrix_trans_one(addr, lane, 16u);
    r3 = ldmatrix_trans_one(addr, lane, 24u);
}

// ---------------- format helpers ----------------

static __device__ __noinline__ float2 unpack_f16x2(std::uint32_t reg) {
    const __half2 h = *reinterpret_cast<const __half2*>(&reg);
    return __half22float2(h);
}
static __device__ __noinline__ float2 unpack_bf16x2(std::uint32_t reg) {
    const __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(&reg);
    return __bfloat1622float2(b);
}
static __device__ __noinline__ float decode_e2m1_nibble(std::uint32_t nib) {
    const float mag = (nib & 7u) == 0u ? 0.0F
                    : (nib & 7u) == 1u ? 0.5F
                    : (nib & 7u) == 2u ? 1.0F
                    : (nib & 7u) == 3u ? 1.5F
                    : (nib & 7u) == 4u ? 2.0F
                    : (nib & 7u) == 5u ? 3.0F
                    : (nib & 7u) == 6u ? 4.0F
                                      : 6.0F;
    return (nib & 8u) ? -mag : mag;
}
static __device__ __noinline__ float decode_e4m3_byte(std::uint32_t b) {
    const unsigned s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 7u;
    float v;
    if (e == 0u) {
        v = static_cast<float>(m) * 0x1p-9F;  // subnormal
    } else {
        // (m + 8) * 2^(e - 10) via direct exponent-bit construction; e in 1..14
        // (15 is inf/NaN, never produced by SATFINITE paths).
        const int biased     = static_cast<int>(e) - 10 + 127;
        const unsigned bits  = static_cast<unsigned>(biased) << 23;
        v                    = (static_cast<float>(m) + 8.0F) * __uint_as_float(bits);
    }
    return s ? -v : v;
}
static __device__ __noinline__ float decode_ue8m0_byte(std::uint32_t b) {
    // ue8m0: power-of-two exponent, biased by 127.
    return b == 0u ? 0.0F
                   : (b >= 127u ? static_cast<float>(1u << static_cast<int>(b) - 127)
                                : 1.0F / static_cast<float>(1u << (127 - static_cast<int>(b))));
}

// ---------------- mma.sync emulation ----------------

static __device__ __noinline__ void mma_f16(float& c0, float& c1, float& c2, float& c3,
                                        std::uint32_t a0, std::uint32_t a1, std::uint32_t a2,
                                        std::uint32_t a3, std::uint32_t b0, std::uint32_t b1) {
    const unsigned tid = threadIdx.x & 3;
    const int s0 = static_cast<int>((2u * tid) * 4u + tid), s1 = s0 + 1;
    const float2 A0 = unpack_f16x2(a0), A1 = unpack_f16x2(a1), A2 = unpack_f16x2(a2),
                 A3 = unpack_f16x2(a3);
    const float2 B0a = unpack_f16x2(__shfl_sync(0xffffffffu, b0, s0));
    const float2 B1a = unpack_f16x2(__shfl_sync(0xffffffffu, b1, s0));
    const float2 B0b = unpack_f16x2(__shfl_sync(0xffffffffu, b0, s1));
    const float2 B1b = unpack_f16x2(__shfl_sync(0xffffffffu, b1, s1));
    c0 += A0.x * B0a.x + A0.y * B0a.y + A2.x * B1a.x + A2.y * B1a.y;
    c1 += A0.x * B0b.x + A0.y * B0b.y + A2.x * B1b.x + A2.y * B1b.y;
    c2 += A1.x * B0a.x + A1.y * B0a.y + A3.x * B1a.x + A3.y * B1a.y;
    c3 += A1.x * B0b.x + A1.y * B0b.y + A3.x * B1b.x + A3.y * B1b.y;
}

static __device__ __noinline__ void mma_bf16(float& c0, float& c1, float& c2, float& c3,
                                         std::uint32_t a0, std::uint32_t a1, std::uint32_t a2,
                                         std::uint32_t a3, std::uint32_t b0, std::uint32_t b1) {
    const unsigned tid = threadIdx.x & 3;
    const int s0 = static_cast<int>((2u * tid) * 4u + tid), s1 = s0 + 1;
    const float2 A0 = unpack_bf16x2(a0), A1 = unpack_bf16x2(a1), A2 = unpack_bf16x2(a2),
                 A3 = unpack_bf16x2(a3);
    const float2 B0a = unpack_bf16x2(__shfl_sync(0xffffffffu, b0, s0));
    const float2 B1a = unpack_bf16x2(__shfl_sync(0xffffffffu, b1, s0));
    const float2 B0b = unpack_bf16x2(__shfl_sync(0xffffffffu, b0, s1));
    const float2 B1b = unpack_bf16x2(__shfl_sync(0xffffffffu, b1, s1));
    c0 += A0.x * B0a.x + A0.y * B0a.y + A2.x * B1a.x + A2.y * B1a.y;
    c1 += A0.x * B0b.x + A0.y * B0b.y + A2.x * B1b.x + A2.y * B1b.y;
    c2 += A1.x * B0a.x + A1.y * B0a.y + A3.x * B1a.x + A3.y * B1a.y;
    c3 += A1.x * B0b.x + A1.y * B0b.y + A3.x * B1b.x + A3.y * B1b.y;
}

static __device__ __noinline__ void mma_tf32(float& c0, float& c1, float& c2, float& c3, float a0,
                                         float a1, float a2, float a3, float b0, float b1) {
    const unsigned tid = threadIdx.x & 3;
    const int s0 = static_cast<int>((2u * tid) * 4u + tid), s1 = s0 + 1;
    const float B0a = __shfl_sync(0xffffffffu, b0, s0), B1a = __shfl_sync(0xffffffffu, b1, s0);
    const float B0b = __shfl_sync(0xffffffffu, b0, s1), B1b = __shfl_sync(0xffffffffu, b1, s1);
    c0 += a0 * B0a + a2 * B1a;
    c1 += a0 * B0b + a2 * B1b;
    c2 += a1 * B0a + a3 * B1a;
    c3 += a1 * B0b + a3 * B1b;
}

static __device__ __noinline__ int s8_byte(std::uint32_t reg, unsigned i) {
    return static_cast<signed char>((reg >> (8 * i)) & 0xFFu);
}
static __device__ __noinline__ void mma_s8(int& c0, int& c1, int& c2, int& c3, std::uint32_t a0,
                                       std::uint32_t a1, std::uint32_t a2, std::uint32_t a3,
                                       std::uint32_t b0, std::uint32_t b1) {
    const unsigned tid = threadIdx.x & 3;
    const int s0 = static_cast<int>((2u * tid) * 4u + tid), s1 = s0 + 1;
    const std::uint32_t B0a = __shfl_sync(0xffffffffu, b0, s0);
    const std::uint32_t B1a = __shfl_sync(0xffffffffu, b1, s0);
    const std::uint32_t B0b = __shfl_sync(0xffffffffu, b0, s1);
    const std::uint32_t B1b = __shfl_sync(0xffffffffu, b1, s1);
#pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        const int a0i = s8_byte(a0, i), a1i = s8_byte(a1, i), a2i = s8_byte(a2, i),
                  a3i = s8_byte(a3, i);
        c0 += a0i * s8_byte(B0a, i) + a2i * s8_byte(B1a, i);
        c1 += a0i * s8_byte(B0b, i) + a2i * s8_byte(B1b, i);
        c2 += a1i * s8_byte(B0a, i) + a3i * s8_byte(B1a, i);
        c3 += a1i * s8_byte(B0b, i) + a3i * s8_byte(B1b, i);
    }
}

static __device__ __noinline__ float e4m3_byte(std::uint32_t reg, unsigned i) {
    return decode_e4m3_byte((reg >> (8 * i)) & 0xFFu);
}

static __device__ __noinline__ void mma_fp8_e4m3(float& c0, float& c1, float& c2, float& c3,
                                             std::uint32_t a0, std::uint32_t a1, std::uint32_t a2,
                                             std::uint32_t a3, std::uint32_t b0, std::uint32_t b1) {
    const unsigned tid = threadIdx.x & 3;
    const int s0 = static_cast<int>((2u * tid) * 4u + tid), s1 = s0 + 1;
    const std::uint32_t B0a = __shfl_sync(0xffffffffu, b0, s0);
    const std::uint32_t B1a = __shfl_sync(0xffffffffu, b1, s0);
    const std::uint32_t B0b = __shfl_sync(0xffffffffu, b0, s1);
    const std::uint32_t B1b = __shfl_sync(0xffffffffu, b1, s1);
#pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        const float a0i = e4m3_byte(a0, i), a1i = e4m3_byte(a1, i), a2i = e4m3_byte(a2, i),
                    a3i = e4m3_byte(a3, i);
        c0 += a0i * e4m3_byte(B0a, i) + a2i * e4m3_byte(B1a, i);
        c1 += a0i * e4m3_byte(B0b, i) + a2i * e4m3_byte(B1b, i);
        c2 += a1i * e4m3_byte(B0a, i) + a3i * e4m3_byte(B1a, i);
        c3 += a1i * e4m3_byte(B0b, i) + a3i * e4m3_byte(B1b, i);
    }
}

// scale_vec::1X: one UE8M0 scale (low byte) per mma. Row scale for row lane/4 comes
// from quad lane 0, row lane/4+8 from quad lane 1; column scale for column j from
// lane 4*j (tid 0).
static __device__ __noinline__ void mma_fp8_e4m3_scaled(float& c0, float& c1, float& c2, float& c3,
                                                    std::uint32_t a0, std::uint32_t a1,
                                                    std::uint32_t a2, std::uint32_t a3,
                                                    std::uint32_t b0, std::uint32_t b1,
                                                    std::uint32_t scale_a, std::uint32_t scale_b) {
    const unsigned lane = threadIdx.x & 31u, tid = lane & 3u;
    float acc0 = 0.0F, acc1 = 0.0F, acc2 = 0.0F, acc3 = 0.0F;
    mma_fp8_e4m3(acc0, acc1, acc2, acc3, a0, a1, a2, a3, b0, b1);
    const float sa_even =
        decode_ue8m0_byte(__shfl_sync(0xffffffffu, scale_a, static_cast<int>(lane & ~3u)));
    const float sa_odd =
        decode_ue8m0_byte(__shfl_sync(0xffffffffu, scale_a, static_cast<int>((lane & ~3u) + 1u)));
    const int sb0 = static_cast<int>((2u * tid) * 4u);
    const float sb_even = decode_ue8m0_byte(__shfl_sync(0xffffffffu, scale_b, sb0));
    const float sb_odd  = decode_ue8m0_byte(__shfl_sync(0xffffffffu, scale_b, sb0 + 4));
    c0 += acc0 * sa_even * sb_even;
    c1 += acc1 * sa_even * sb_odd;
    c2 += acc2 * sa_odd * sb_even;
    c3 += acc3 * sa_odd * sb_odd;
}

// m16n8k64 e2m1 with scale_vec::4X: four UE4M3 sub-block scales (16 k each) packed
// in sfa/sfb. This lane's a0/a1 hold k=8t..8t+7 (sub-block t/2), a2/a3 hold
// k=8t+32..39 (sub-block 2+t/2); sfb must come from the same B-owning lane as b0/b1.
static __device__ __noinline__ float e2m1_nibble(std::uint32_t reg, unsigned i) {
    return decode_e2m1_nibble((reg >> (4 * i)) & 0xFu);
}
static __device__ __noinline__ void mma_nvfp4_e2m1(float& c0, float& c1, float& c2, float& c3,
                                               std::uint32_t a0, std::uint32_t a1, std::uint32_t a2,
                                               std::uint32_t a3, std::uint32_t b0, std::uint32_t b1,
                                               std::uint32_t sfa, std::uint32_t sfb) {
    const unsigned tid = threadIdx.x & 3;
    const int s0 = static_cast<int>((2u * tid) * 4u + tid), s1 = s0 + 1;
    const std::uint32_t B0a = __shfl_sync(0xffffffffu, b0, s0);
    const std::uint32_t B1a = __shfl_sync(0xffffffffu, b1, s0);
    const std::uint32_t B0b = __shfl_sync(0xffffffffu, b0, s1);
    const std::uint32_t B1b = __shfl_sync(0xffffffffu, b1, s1);
    const float sfb0 = decode_e4m3_byte((__shfl_sync(0xffffffffu, sfb, s0) >> (8u * (tid / 2u))) &
                                        0xFFu);
    const float sfb0h = decode_e4m3_byte(
        (__shfl_sync(0xffffffffu, sfb, s0) >> (8u * (2u + tid / 2u))) & 0xFFu);
    const float sfb1 = decode_e4m3_byte((__shfl_sync(0xffffffffu, sfb, s1) >> (8u * (tid / 2u))) &
                                        0xFFu);
    const float sfb1h = decode_e4m3_byte(
        (__shfl_sync(0xffffffffu, sfb, s1) >> (8u * (2u + tid / 2u))) & 0xFFu);
    const float sfa0 = decode_e4m3_byte((sfa >> (8u * (tid / 2u))) & 0xFFu);
    const float sfa0h = decode_e4m3_byte((sfa >> (8u * (2u + tid / 2u))) & 0xFFu);
    const float sfa1 = decode_e4m3_byte((sfa >> (8u * (tid / 2u))) & 0xFFu);
    const float sfa1h = decode_e4m3_byte((sfa >> (8u * (2u + tid / 2u))) & 0xFFu);
    float dot00 = 0.0F, dot01 = 0.0F, dot10 = 0.0F, dot11 = 0.0F;  // k8t block per output
    float e00 = 0.0F, e01 = 0.0F, e10 = 0.0F, e11 = 0.0F;          // k8t+32 block per output
#pragma unroll
    for (unsigned i = 0; i < 8; ++i) {
        const float a0i = e2m1_nibble(a0, i), a1i = e2m1_nibble(a1, i), a2i = e2m1_nibble(a2, i),
                    a3i = e2m1_nibble(a3, i);
        dot00 += a0i * e2m1_nibble(B0a, i);
        dot01 += a0i * e2m1_nibble(B0b, i);
        dot10 += a1i * e2m1_nibble(B0a, i);
        dot11 += a1i * e2m1_nibble(B0b, i);
        e00 += a2i * e2m1_nibble(B1a, i);
        e01 += a2i * e2m1_nibble(B1b, i);
        e10 += a3i * e2m1_nibble(B1a, i);
        e11 += a3i * e2m1_nibble(B1b, i);
    }
    c0 += sfa0 * sfb0 * dot00 + sfa0h * sfb0h * e00;
    c1 += sfa1 * sfb1 * dot01 + sfa0h * sfb1h * e01;
    c2 += sfa0 * sfb0 * dot10 + sfa0h * sfb0h * e10;
    c3 += sfa1 * sfb1 * dot11 + sfa0h * sfb1h * e11;
}

// ---------------- cp.async ----------------
// On Volta a cp.async group is reproduced with a synchronous register-mediated
// copy; commit is a no-op and wait(N) is satisfied trivially because copies
// complete at issue. smem visibility for consumers relies on the
// __syncthreads()/barrier that always follows in these kernels.

static __device__ __noinline__ void cp_async_copy(void* smem_dst, const void* gmem_src,
                                              int bytes) {
    auto* dst = static_cast<char*>(smem_dst);
    const char* src = static_cast<const char*>(gmem_src);
    for (int i = 0; i < bytes; ++i) dst[i] = src[i];
}

template <int Bytes>
static __device__ __noinline__ void cp_async_zfill_emul(void* smem_dst, const void* gmem_src,
                                                    int src_bytes) {
    auto* dst = static_cast<char*>(smem_dst);
    const char* src = static_cast<const char*>(gmem_src);
    const int n = src_bytes < Bytes ? src_bytes : Bytes;
    for (int i = 0; i < Bytes; ++i) dst[i] = i < n ? src[i] : '\0';
}

}  // namespace ninfer::ops::common::volta
