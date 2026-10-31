#pragma once

#include "ops/common/cuda/memory.cuh"
#include "ops/common/cuda/volta_isa.cuh"

#include <cuda.h>
#include <cuda_pipeline.h>

#include <cstdint>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#define NINFER_VOLTA_ISA 1
#else
#define NINFER_VOLTA_ISA 0
#endif

namespace ninfer::ops::common {

// Volta tier: cp.async becomes a synchronous register-mediated copy; commit and
// wait are then trivially satisfied. Consumers rely on the __syncthreads() that
// follows every producer phase in these kernels.

template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void cp_async(void* smem_dst, const void* gmem_src) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "cp_async supports 4, 8, or 16 bytes");
#if NINFER_VOLTA_ISA
    volta::cp_async_copy(smem_dst, gmem_src, Bytes);
#else
    if constexpr (Policy == Cache::cg) {
        static_assert(Bytes == 16, "cp.async.cg requires a 16-byte copy");
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src), "n"(Bytes));
    }
#endif
}

template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void cp_async_zfill(void* smem_dst, const void* gmem_src,
                                               int src_bytes) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16,
                  "cp_async_zfill supports 4, 8, or 16 bytes");
#if NINFER_VOLTA_ISA
    volta::cp_async_zfill_emul<Bytes>(smem_dst, gmem_src, src_bytes);
#else
    if constexpr (Policy == Cache::cg) {
        static_assert(Bytes == 16, "cp.async.cg requires a 16-byte copy");
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src), "r"(src_bytes));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src), "n"(Bytes), "r"(src_bytes));
    }
#endif
}

template <Cache Policy>
__device__ __forceinline__ void cp_async_cache_hint(void* destination, const void* source,
                                                    int valid_bytes, std::uint64_t policy) {
#if NINFER_VOLTA_ISA
    (void)policy;
    volta::cp_async_zfill_emul<16>(destination, source, valid_bytes);
#else
    if constexpr (Policy == Cache::cg) {
        asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                     :
                     : "r"(smem_addr(destination)), "l"(source), "r"(valid_bytes), "l"(policy));
    } else {
        asm volatile("cp.async.ca.shared.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
                     :
                     : "r"(smem_addr(destination)), "l"(source), "r"(valid_bytes), "l"(policy));
    }
#endif
}

__device__ __forceinline__ void cp_commit() {
#if NINFER_VOLTA_ISA
    // synchronous copies: nothing outstanding
#else
    asm volatile("cp.async.commit_group;\n");
#endif
}

template <int Groups>
__device__ __forceinline__ void cp_wait() {
    static_assert(Groups >= 0 && Groups <= 7, "cp_wait group count must fit the PTX immediate");
#if NINFER_VOLTA_ISA
    // copies completed at issue
#else
    asm volatile("cp.async.wait_group %0;\n" : : "n"(Groups));
#endif
}

template <int Bytes>
__device__ __forceinline__ void pipe_copy(void* smem_dst, const void* gmem_src) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "pipe_copy supports 4, 8, or 16 bytes");
#if NINFER_VOLTA_ISA
    volta::cp_async_copy(smem_dst, gmem_src, Bytes);
#else
    __pipeline_memcpy_async(smem_dst, gmem_src, Bytes);
#endif
}

__device__ __forceinline__ void pipe_commit() {
#if NINFER_VOLTA_ISA
#else
    __pipeline_commit();
#endif
}

template <int Groups>
__device__ __forceinline__ void pipe_wait() {
    static_assert(Groups >= 0 && Groups <= 7, "pipe_wait group count must fit the PTX immediate");
#if NINFER_VOLTA_ISA
#else
    __pipeline_wait_prior(Groups);
#endif
}

// mbarrier and TMA are Hopper+ facilities. They compile to a trap on Volta: any
// kernel instantiated for v100 must not take a TMA/mbarrier data path (those
// live in rtx5090/spark target dirs; Volta gets transplanted kernels instead).

__device__ __forceinline__ void cta_mbarrier_init(std::uint64_t* barrier, std::uint32_t arrivals) {
#if NINFER_VOLTA_ISA
    (void)barrier; (void)arrivals; __trap();
#else
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                 :
                 : "r"(smem_addr(barrier)), "r"(arrivals)
                 : "memory");
#endif
}

__device__ __forceinline__ void cta_mbarrier_wait(std::uint64_t* barrier, std::uint32_t phase) {
#if NINFER_VOLTA_ISA
    (void)barrier; (void)phase; __trap();
#else
    constexpr std::uint32_t kSuspendTicks = 0x989680;
    asm volatile("{\n"
                 ".reg .pred done;\n"
                 "wait_loop_%=: \n"
                 "mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1, %2;\n"
                 "@done bra wait_done_%=;\n"
                 "bra wait_loop_%=;\n"
                 "wait_done_%=: \n"
                 "}\n"
                 :
                 : "r"(smem_addr(barrier)), "r"(phase), "r"(kSuspendTicks)
                 : "memory");
#endif
}

__device__ __forceinline__ void cta_mbarrier_arrive(std::uint64_t* barrier) {
#if NINFER_VOLTA_ISA
    (void)barrier; __trap();
#else
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" : : "r"(smem_addr(barrier)) : "memory");
#endif
}

__device__ __forceinline__ void cta_mbarrier_arrive_expect_tx(std::uint64_t* barrier,
                                                              std::uint32_t bytes) {
#if NINFER_VOLTA_ISA
    (void)barrier; (void)bytes; __trap();
#else
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :
                 : "r"(smem_addr(barrier)), "r"(bytes)
                 : "memory");
#endif
}

__device__ __forceinline__ void cta_mbarrier_fence_init() {
#if NINFER_VOLTA_ISA
#else
    asm volatile("fence.mbarrier_init.release.cluster;" : : : "memory");
#endif
}

// The caller owns tensor-map geometry, shared layout, and mbarrier transaction accounting.
__device__ __forceinline__ void tma_load_2d(void* destination, const CUtensorMap* map, int c0,
                                            int c1, std::uint64_t* barrier) {
#if NINFER_VOLTA_ISA
    (void)destination; (void)map; (void)c0; (void)c1; (void)barrier; __trap();
#else
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3}], [%4];"
                 :
                 : "r"(smem_addr(destination)), "l"(map), "r"(c0), "r"(c1), "r"(smem_addr(barrier))
                 : "memory");
#endif
}

__device__ __forceinline__ void tma_load_3d(void* destination, const CUtensorMap* map, int c0,
                                            int c1, int c2, std::uint64_t* barrier) {
#if NINFER_VOLTA_ISA
    (void)destination; (void)map; (void)c0; (void)c1; (void)c2; (void)barrier; __trap();
#else
    asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3, %4}], [%5];"
                 :
                 : "r"(smem_addr(destination)), "l"(map), "r"(c0), "r"(c1), "r"(c2),
                   "r"(smem_addr(barrier))
                 : "memory");
#endif
}

__device__ __forceinline__ void tma_load_4d(void* destination, const CUtensorMap* map, int c0,
                                            int c1, int c2, int c3, std::uint64_t* barrier) {
#if NINFER_VOLTA_ISA
    (void)destination; (void)map; (void)c0; (void)c1; (void)c2; (void)c3; (void)barrier; __trap();
#else
    asm volatile("cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3, %4, %5}], [%6];"
                 :
                 : "r"(smem_addr(destination)), "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(c3),
                   "r"(smem_addr(barrier))
                 : "memory");
#endif
}

} // namespace ninfer::ops::common
