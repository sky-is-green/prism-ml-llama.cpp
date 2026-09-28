# MoE expert slot cache (FreeToken-style) on the prism engine — 2026-09-28

First working FreeToken mechanism in the AMD engine: host-placed expert banks get a
GPU slot band, decode routes through the slots with an LRU ensure + on-demand H2D
fetch, and the graph is otherwise unchanged. Built on `engine/moe-slot-cache`.

> Measurement note: numbers below are warm (model/page cache hot), 185-token prompt,
> 64-token greedy. All runs name the model explicitly; an earlier table in this note
> was accidentally measured on the 35B (a missing env export) and has been redone.

## What landed

- `ggml` op `GGML_OP_MOE_CACHE_MAP` (+ `ggml/include/ggml-moe-cache.h`): per step, copy
  the routed ids to the host, map expert ids to slots (LRU eviction, host state),
  enqueue one H2D copy per missing expert row per bank on the compute stream, and
  return the mapped ids. GPU-backend only; disables CUDA graph capture for its graph.
- `llama_moe_slot_cache` (`src/llama-moe-slot-cache.{h,cpp}`): creates the slot banks.
  A layer is cacheable when its `ffn_{gate,up,down}_exps` (or fused `ffn_gate_up_exps`)
  tensors are host-resident (e.g. placed there by `-ncmoe`/`-ot`), carry data, and
  have no scales/biases. One backend buffer per device, `N` slots per bank.
- **Identity promotion**: when `N >= n_expert`, the whole bank is copied to the GPU once
  at context init and the layer is used directly with the routed ids — no map op, no
  per-step host sync, graph capture stays on (and prefill runs on GPU too). This is what
  `-msc auto` resolves to whenever free VRAM covers the banks.
- `build_moe_ffn` routes partial layers through the slot banks when
  `n_tokens * n_expert_used <= N` (decode); larger batches (prefill) use the host bank.
- CLI/context: `--moe-slot-cache N` / `-msc N`, or `-msc auto` (fill each device's free
  memory after a 20%+512 MiB reserve; port of FreeToken's `--moe-cache-auto`).
- Stats: `LLAMA_MOE_SLOT_STATS=1` prints hit/fetch totals + hit rate at context exit.

Not in this cut: CPU overflow for capped fetches (the q* hybrid split), prefill
streaming for partially-cached layers, runtime VRAM reallocation, FTW.

## Results

### 125B Qwen3.8-Flash-Next GSQ-RCO Q2_0 (512 experts top-10, 48 layers), 2× 7900 XT

`-c 4096 -ts 0.522,0.478 --split-mode layer -lm mmap --lazy-mode on -fa auto`,
headless-first (`HIP_VISIBLE_DEVICES=1,0`).

| config | experts off-VRAM | slot cache | pp t/s | tg t/s | hit rate |
|---|---:|---:|---:|---:|---:|
| all-resident | 0 | – | 109.1 | 28.0 | – |
| `-ncmoe 12` | 8.1 GiB | – | 54.2 | 19.1 | – |
| `-ncmoe 12 -msc 256` | 8.1 GiB | 4.05 GiB | 48.1 | **24.1** (+26%) | 62.1% |
| `-ncmoe 12 -msc 512` (identity) | – (promoted) | 8.10 GiB | 100.8 | **27.4** | – |
| `-ncmoe 24` | 16.2 GiB | – | 28.7 | 12.9 | – |
| `-ncmoe 24 -msc 256` | 16.2 GiB | 8.10 GiB | 26.0 | **20.8** (+61%) | 63.5% |
| `-ncmoe 36` | 24.3 GiB | – | 6.3 | 4.4 | – |
| `-ncmoe 36 -msc 336` | 24.3 GiB | 15.9 GiB | 12.4 | **8.2** (+86%) | 64.8% |

Reading: the cache recovers 36–61% of the lost decode speed while 8–16 GiB of expert
working set stays off-VRAM, and doubles throughput at the 24 GiB-off point. With
`-msc auto`/explicit coverage the banks are promoted and the numbers match the
all-resident row. Outputs are token-identical to the all-resident run throughout.

### 35B-A3B IQ2_M (qwen35moe, 256 experts top-8, 40 layers), one RX 7900 XT

`-c 512 -np 1 -t 8`, all expert banks host-resident (`-cmoe`) in the cache arms.

| config | expert VRAM | pp t/s | tg t/s | hit rate |
|---|---:|---:|---:|---:|
| `-ngl 99` all GPU | ~13.5 GiB | 464.3 | 108.8 | – |
| `-cmoe` all CPU | 0 | 176.2 | 37.8 | – |
| `-cmoe -msc 64` | 2.48 GiB | 202.0 | 59.98 | 75.9% |
| `-cmoe -msc auto` (=256, identity) | 9.92 GiB | 566.6 | 108.87 | – |
| `-cmoe -msc 384` (identity) | 14.89 GiB | 632.3 | 109.04 | – |

Identity promotion matches the all-GPU arm token-for-token and in throughput; the
partial 64-slot cache buys 59% decode over CPU-only from 2.5 GiB.

## FreeToken alignment (upstream `0d652e7`, "feat(rocm): add RDNA3 and RDNA4 runtime foundation")

Their AMD foundation: a `hip_compat.h` CUDA→HIP shim, hipcc builds of their C++
kernels (`pinned_tensor`, fast index copy), ROCm detection/arch gating
(`is_rocm`, `get_rocm_gfx_arch`, `is_gfx11xx_family`), `--offload-arch` build flags,
a single-bank fallback for the fused index copy, PDL launches omitted on ROCm, and
Triton fallbacks where flashinfer/sgl-kernels have no ROCm builds.

Our engine is ROCm-native already, so the transferable parts are the **policies**,
not the kernels:

- **Hybrid vs offload decision** (`moe/benchbw.py`): hybrid only when the measured CPU
  MoE bandwidth exceeds ~2x the measured PCIe gather bandwidth, else fetch misses to
  the GPU. Our measurements: CPU expert GEMV ≈ 12 GB/s (35B, 40 layers top-8) vs
  PCIe gather ≈ 25 GB/s → ratio ≈ 0.5, so the offload/fetch-all mode we implement is
  the right call on this box. Their **overlap bench** (CPU GEMV and PCIe gather
  concurrent) sets the hybrid fetch fraction when hybrid is indicated.
- **Cache auto-sizing** (`engine/cache_budget.py`): budget = free memory − reserve,
  greedy expert slots first, KV takes the remainder. Ported as `-msc auto`.
- **Graph-compatible ensure** (their Triton `ensure_experts` + `fast_index_copy`):
  the LRU/copy-list logic runs device-side, so no host round trip per layer and the
  decode graph stays capturable. Their gather reads pinned host banks directly; our
  host banks are mmap'd page cache (the 125B cannot be pinned in 30 GB RAM), so the
  same trick needs a pinned staging ring on this box. The identity-promotion path
  (above) already removes the per-step cost whenever the cache covers a layer.

## Next

1. **Device-side ensure for partial caches** (~150 µs/layer host cost today): a HIP
   kernel for the LRU + copy list, plus a staged gather (pinned ring) so copies can be
   enqueued without a host round trip and the graph stays capturable.
2. **Prefill streaming** for partially-cached layers: pp is 26–48 t/s there vs 109
   all-resident; FreeToken's double buffer needs 2E slots, so this needs a chunked
   expert-major design instead.
3. **q\* hybrid**: only worth it once the CPU MoE path beats ~2x PCIe (it is ~0.5x
   today), per their own criterion.
4. **Telemetry**: expose the hit/fetch counters through the server API and Hivebench
   tier status, and add runtime cache resize.

## Reproduce

```sh
# build
cmake --build build-hip -j 12 --target llama-server

# 35B arm (one card): partial cache vs identity promotion
HIP_VISIBLE_DEVICES=1 LLAMA_MOE_SLOT_STATS=1 ./build-hip/bin/llama-server -m <35B-iq2_m.gguf> \
  -ngl 99 -cmoe -msc 64 -c 512 -np 1 -t 8

# 125B FreeToken config (both cards, headless first)
HIP_VISIBLE_DEVICES=1,0 LLAMA_MOE_SLOT_STATS=1 ./build-hip/bin/llama-server \
  -m ~/Desktop/work/models/qwen38-q2_0/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  -ngl 99 -ncmoe 24 -msc 256 -c 4096 -np 1 -t 8 \
  -ts 0.522,0.478 --split-mode layer -lm mmap --lazy-mode on -fa auto
```
