# NInfer Volta (sm_70) port — roadmap

Volta port of [Neroued/ninfer](https://github.com/Neroued/ninfer) for Tesla V100 SXM2
(32 GB + 16 GB), rebuilt on current upstream. The historical community port
([geoffwatts/ninfer-v100](https://github.com/geoffwatts/ninfer-v100)) froze at 2026-09-14;
this repo re-applies its Volta work on top of live upstream instead of its stale base.
NInfer is Apache-2.0; geoffwatts' commit history is preserved verbatim on the
`geoffwatts-v100` branch for attribution and as the reference diff.

## Branch model

| Branch | Contents |
| --- | --- |
| `master` | pristine upstream sync — never commit here |
| `volta` | the port line (this branch) |
| `geoffwatts-v100` | frozen reference: geoffwatts tip `b37d0dd` (2026-09-14) |

## Verified divergence (measured 2026-10-31)

- Upstream `master` = `1882c6ad` (1,489 commits, active daily).
- Volta reference base = `b88c0f6f` (2026-09-09); since then **97 Volta commits**
  (90 new kernel files + 196 modified, 288 files, +20,984 / −2,893).
- Upstream moved **+421 commits** since that base (3,536 files, +415k / −202k — a
  full `src/ops` restructure).
- `git merge-tree` of the two tips: **449 conflicted paths** (149 in `src/ops`, many
  `CONFLICT (file location)` / `directory rename split`). A wholesale rebase or merge
  is not viable; the port is **re-applied per subsystem** against upstream's new layout.

## Upstream seams the port uses

- `cmake/NinferPlatform.cmake` hard-gates `NINFER_TARGET` to `rtx5090`/`dgx_spark`;
  31 CMake files branch on it; `sources.cmake` files put `${NINFER_TARGET}` **into
  source paths** (`dense/packed/${NINFER_TARGET}/uniform.cu`). Five op families already
  carry per-target dirs (`gated_residual`, `linear`, `linear_attention`, `moe`,
  `softmax_attention`) — `v100/` joins those.
- `src/core/platform.h.in` exports target flags + capabilities to 43 source files.
- This scaffold commit adds `NINFER_TARGET=v100` (sm_70, x86-64), a hard CUDA-13
  rejection (CUDA 13 removed Volta; the build stays on CUDA 12.9.1, see
  `Containerfile.volta`), `v100-release|dev|ops` presets, and the Volta capability flags.

## Volta constraints (from the reference port + rig measurements)

- **CUDA 12.9.1 max** (sm_70 removed in 13.x). Upstream kernels assume sm_120a
  (Blackwell MMA `.e2m1`, TMA): they must not compile for sm_70 — each `kernels/`
  family needs a Volta replacement in the target dir, not a guard.
- No FP4/FP8 tensor cores on Volta: NVFP4 is dequantized in software via WMMA FP16 /
  SIMT "QPN" kernels; weights are repacked device-side at load (immutable artifact).
- KV dtype on Volta: `int8` only (fp16 V storage existed in the reference port;
  nvfp4/k8v4 KV unavailable). 32 GB card ceiling: 184,320-token window
  (weights ~19.4 GiB + int8 KV).
- NInfer decode pulls ~220 W on V100 (dequant is compute-bound): it needs a 250 W
  cap on GPU1, unlike Strata/llama.cpp (180 W plateau). Power is part of the deploy.

## Baseline to beat (measured on this rig, b37d0dd = geoffwatts tip)

| Context | decode tok/s |
| --- | --- |
| 2K | ~64 |
| 8K | ~57 |
| 125K real | 34.9 (MTP accept 246/468), prefill ~320 |
| short prefill | ~1,050–1,100 (engine ceiling on Volta) |

Acceptance protocol for every milestone: 400-token medians, 3 runs, warm engine,
GPU1 capped 250 W, llama-swap isolated; artifact = Qwen3.8-27B NVFP4
(local v2 `11dbbbbbc…`, 23,719,496,192 B).

## Port mechanics (learned at M1, applies to every family)

- A kernel family = (a) shared manifest `sources.cmake` (target-agnostic),
  (b) shared template launch headers instantiated by per-shape `.cu` files,
  (c) per-target dirs `${NINFER_TARGET}/` added additively.
- Shape `.cu` files are **host dispatch only** (`select_q8_n5120_k6144(tokens)`
  returns a `Q8Launch` function pointer); the Ampere ISA enters each shape TU
  through the template launch headers (schedule → `cp.async`/`ldmatrix` bodies).
- Transplant pattern = **contract adapter**:
  1. exclude Ampere-only shared `.cu` from v100 builds
     (`if(NOT NINFER_TARGET_V100)` in the shared manifest);
  2. keep shape files and `select_*` signatures untouched;
  3. in shared template launch headers, branch on `NINFER_VOLTA`
     (compile-def the v100 target adds) to route to Volta kernels with the
     same `launch_*` contracts;
  4. Volta kernels come from the frozen `geoffwatts-v100` branch
     (`src/ops/**/volta*|*qpn*|*rowsplit*|.cu|.cuh`, ~280 files) adapted to
     current interfaces.
- CUDA-12.9 vs 13.4 API deltas handled once at M1 (`CUDART_VERSION` guards):
  `cudaMemcpyBatchAsync` (9-arg + const casts), `cudaStreamGetCaptureInfo_v2`,
  `__reduce_*_sync` → butterfly helpers in `ops/common/warp_reduce_compat.cuh`.

## Milestones

- **M1b done (2026-10-31): Volta ISA emulation tier.** New
  `src/ops/common/cuda/volta_isa.cuh` reproduces the exact per-lane fragment
  semantics of the Ampere+ instructions under `__CUDA_ARCH__ < 800`:
  `ldmatrix{.x2,.x4}{.trans}` via `ld.shared` + address shuffles;
  `mma.m16n8k16` f16/bf16, `m16n8k8` tf32, `m16n8k32` s8/e4m3,
  `m16n8k32.block_scale` ue8m0 and `m16n8k64.kind::mxf4nvf4` ue4m3 via
  shfl-gather + FFMA/IMAD (fragment layouts documented in the header).
  `cp.async{.ca,.cg,.zfill,cache_hint}` -> synchronous register copies
  (commit/wait trivially satisfied); `mbarrier`/TMA -> `__trap()` (Hopper
  data paths must never be instantiated for v100 kernels); L2
  `createpolicy` -> 0; `cvt.rn.bf16x2.f32` -> intrinsic; `tanh.approx` ->
  `tanhf`; e2m1 pack -> `__nv_cvt_float2_to_fp4x2` (soft in 12.9, returns
  `__nv_fp4x2_storage_t`). All hubs routed through
  `NINFER_VOLTA_ISA`; sm_80+ keeps original PTX byte-for-byte. Standalone
  probe (`mma/pipeline/nvfp4` include + use) compiles for sm_70: `HUB_OK`.
  This is the correctness floor; transplanted Volta kernels (M2) replace
  the hot paths for speed.


1. **M0 — scaffold + configure probe** *(done 2026-10-31)*: `v100` scaffolding +
   `tools/volta/scaffold_target_dirs.sh` mirror 189 placeholder files (143
   `.cu`/`.cuh`); first configure pass failed with exactly 16 missing per-target
   includes, after mirroring `cmake --preset v100-release` configures **clean**
   on CUDA 12.9.1. Full inventory + transplant mapping: `docs/volta/inventory.md`.
2. **M1 — green build with fallbacks** *(compat wave done 2026-10-31)*:
   CUDA 12.9 API parity landed (transfer.cu `batch_copy_async`, decode_graph.cpp
   version guards — core lib TU verified green, warp-reduce butterflies +
   8 call-site rewrites). Measured `v100-release` failure surface: **130 ops TUs,
   all ptxas ISA wall** (.bf16 ×363,964, ldmatrix ×270,024, mma .m16n8k16
   ×224,606, cp.async ×185,799, plus .m16n8k32/.block_scale tail). Family
   breakdown of failing TUs: linear/projection 25, linear/split 19,
   gated_delta_net 14, linear/add 14, linear/kernels 13, softmax_attention/dense
   11, linear/swiglu 10, linear/topk 6, moe/kernels 4, kimi_delta_attention 4,
   linear/pair 3, singles (sliding_window, qsa, sampling, kv_cache ×2,
   dynamic_grouped_conv ×2, decode_graph). Upstream sync: volta base =
   `1882c6ad` (0 behind as of 2026-10-31). Remaining M1 work: `NINFER_VOLTA`
   compile-def + manifest guards + stub/slow Volta sources at every demanded
   path; engine boots with a **v3 artifact** (upstream `tools/upgrade_ninfer_v2_to_v3.py`
   converts the local v2 file; no 24 GB re-download). Correctness: known-prompt
   outputs + perplexity match vs b37d0dd within noise.
3. **M2 — linear decode parity**: port `nvfp4/q4/q5/w8` Volta QPN+MMA GEMMs and the
   load-time prepack (the reference port's biggest win) into
   `src/ops/linear/**/v100/`. Gate: ≥ b37d0dd at 2K/8K.
4. **M3 — attention**: Volta flash attention + long-context scaling +
   `attn_input_proj`/`gdn_input_proj`/`linear_swiglu` fused kernels (the remaining
   105 touched files).
5. **M4 — speculative + extras**: MTP (width 1–7), context-lookup extension,
   `--lm-head-draft`, DFlash2 fusions, vision encoder.
6. **M5 — deploy swap**: full bake-off vs b37d0dd at 2K/8K/30K/125K; swap the binary
   in the llama-swap runtime dir via the existing `ninfer-swap-binary.sh` pattern,
   keep the old binary as revert.

## Risks / standing notes

- Upstream is active daily; rebase `volta` onto `master` at each milestone, not continuously.
- `NINFER_EXPERT_OFFLOAD=1` was chosen for v100 capability (62 GB host RAM); revisit
  if it drags in untested Blackwell-era offload paths at M1.
- The 16 GB V100 (GPU2, SXM2→PCIe adapter) is out of scope for single-engine scope;
  upstream has no multi-GPU Volta story either. One engine = one card (32 GB).
- Do not touch the production runtime dir (`LLAMACCP/ninfer/`) until M5.
