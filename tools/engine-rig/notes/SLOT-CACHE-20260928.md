# MoE expert slot cache (FreeToken-style) on the prism engine — 2026-09-28

First working FreeToken mechanism in the AMD engine: host-placed expert banks get a
GPU slot band, decode routes through the slots with an LRU ensure + on-demand H2D
fetch, and the graph is otherwise unchanged. Built on `engine/moe-slot-cache`.

> Measurement note: all numbers below are warm (model and page cache hot) on the same
> box, 185-token prompt, 64-token greedy (`ignore_eos`). Earlier session numbers taken
> right after a cold load run ~2x low. The engine rig's historical 125B decode figures
> (24–28 t/s) were measured with `LLAMA_GRAPH_DUMP=2` set by `run-125b.sh`, which
> costs roughly 2.3x; without it the same config does ~63 t/s.

## What landed

- `ggml` op `GGML_OP_MOE_CACHE_MAP` (+ `ggml/include/ggml-moe-cache.h`): per step, copy
  the routed ids to the host, map expert ids to slots (LRU eviction, host state),
  enqueue one H2D copy per missing expert row per bank on the compute stream, and
  return the mapped ids. GPU-backend only; disables CUDA graph capture for its graph.
- `llama_moe_slot_cache` (`src/llama-moe-slot-cache.{h,cpp}`): creates the slot banks.
  A layer is cacheable when its `ffn_{gate,up,down}_exps` (or fused `ffn_gate_up_exps`)
  tensors are host-resident (e.g. placed there by `-ncmoe`/`-ot`) and carry no
  scales/biases. One backend buffer per device, `N` slots per bank.
- `build_moe_ffn` routes cacheable layers through the slot banks when
  `n_tokens * n_expert_used <= N` (decode); larger batches (prefill) use the host
  bank exactly as before.
- CLI/context: `--moe-slot-cache N` / `-msc N`, or `-msc auto` to fill each device's
  free memory after a 20%+512 MiB reserve (port of FreeToken's `--moe-cache-auto`;
  clamp: one slot count for all cached layers, never above `n_expert`).
- Stats: `LLAMA_MOE_SLOT_STATS=1` prints hit/fetch totals + hit rate at context exit.

Not in this cut: CPU overflow for capped fetches (the q* hybrid split), prefill
double-buffered streaming, runtime VRAM reallocation, FTW. The decode path is
FreeToken's "gpu"/offload mode: every miss is fetched, nothing is computed on the CPU.

## Results

### 35B-A3B IQ2_M (qwen35moe, 256 experts top-8, 40 layers), one RX 7900 XT

`-c 512 -np 1 -t 8`, warm. Cache arm: all expert banks host-resident (`-cmoe`),
slots on GPU. Outputs are **token-identical to the all-GPU run**.

| config | expert VRAM | pp t/s | tg t/s | hit rate |
|---|---:|---:|---:|---:|
| `-ngl 99` all GPU | ~13.5 GiB | 756.6 | 108.9 | – |
| `-cmoe` all CPU | 0 | 232.5 | 40.7 | – |
| `-cmoe -msc 64` | 2.48 GiB | 224.0 | 59.9 | 75.9% |
| `-cmoe -msc auto` (=256 = full) | 9.92 GiB | 223.5 | 66.0 | 80.7% |

### 125B Qwen3.8-Flash-Next GSQ-RCO Q2_0 (512 experts top-10, 48 layers), 2× 7900 XT

`-c 4096 -ts 0.522,0.478 --split-mode layer -lm mmap --lazy-mode on -fa auto`,
headless-first (`HIP_VISIBLE_DEVICES=1,0`), warm.

| config | experts off-VRAM | slot cache | pp t/s | tg t/s | hit rate |
|---|---:|---:|---:|---:|---:|
| all-resident | 0 | – | 420.8 | 62.6 | – |
| `-ncmoe 12` | 12 layers (8.1 GiB) | – | 306.2 | 46.2 | – |
| `-ncmoe 12 -msc 256` | 12 layers | 4.05 GiB | 293.3 | **56.6** | 82.0% |
| `-ncmoe 24` | 24 layers (16.2 GiB) | – | 230.1 | 38.3 | – |
| `-ncmoe 24 -msc 256` | 24 layers | 8.10 GiB | 236.3 | **48.8** | 81.1% |
| `-ncmoe 36` | 36 layers (24.3 GiB) | – | 190.3 | 34.2 | – |
| `-ncmoe 36 -msc 336` | 36 layers | 15.9 GiB | 194.8 | **44.6** | 80.5% |
| `-ncmoe 36 -msc auto` | 36 layers | 8.94 GiB (auto) | 201.2 | **44.5** | 80.5% |

Reading: the cache recovers ~90% of the all-resident decode with a quarter of the
expert working set off-VRAM, and the gain grows with the offload (+22%, +28%, +30%).
Outputs are token-identical to the all-resident run. `msc auto` lands on the same
operating point as the tuned size.

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
  decode graph stays capturable. This is our next tok/s lever: the host-sync design
  currently costs ~150 µs/layer (≈6 ms/step at 36 cached layers; visible as 66 vs
  109 t/s on the 35B full-residency arm). Their gather reads pinned host banks
  directly; our host banks are mmap'd page cache (the 125B cannot be pinned in 30 GB
  RAM), so a device-side gather needs either a pinned staging ring or a
  hipMemcpyAsync-driven copy list with a single sync per step instead of per layer.

## Next

1. **Device-side ensure / graph-compatible decode**: kill the per-layer host sync
   (~150 µs/layer) and re-enable graph capture for cached graphs. Biggest measured
   tok/s lever for cached arms.
2. **Prefill streaming**: cached layers still fall back to CPU compute in prefill
   (pp 190–306 vs 421 all-resident). FreeToken's double buffer needs 2E slots; with
   S < E we need chunked/streamed expert-major prefill.
3. **q\* hybrid**: only worth it if the CPU MoE path is made much faster first
   (their threshold: CPU > 2x PCIe). llama.cpp's CPU mmid is ~12 GB/s here vs ~25 GB/s
   PCIe.
4. **Telemetry**: expose the hit/fetch counters through the server API and Hivebench
   tier status (engine → harness seam), and add runtime cache resize.

## Reproduce

```sh
# build
cmake --build build-hip -j 12 --target llama-server

# 35B arm (one card), cache-size sweep / auto
HIP_VISIBLE_DEVICES=1 LLAMA_MOE_SLOT_STATS=1 ./build-hip/bin/llama-server -m <35B-iq2_m.gguf> \
  -ngl 99 -cmoe -msc auto -c 512 -np 1 -t 8

# 125B FreeToken config (both cards, headless first)
HIP_VISIBLE_DEVICES=1,0 LLAMA_MOE_SLOT_STATS=1 ./build-hip/bin/llama-server \
  -m ~/Desktop/work/models/qwen38-q2_0/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  -ngl 99 -ncmoe 24 -msc 256 -c 4096 -np 1 -t 8 \
  -ts 0.522,0.478 --split-mode layer -lm mmap --lazy-mode on -fa auto
```
