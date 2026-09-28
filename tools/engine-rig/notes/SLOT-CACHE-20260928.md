# MoE expert slot cache (FreeToken-style) on the prism engine — 2026-09-28

First working FreeToken mechanism in the AMD engine: host-placed expert banks get a
GPU slot band, decode routes through the slots with an LRU ensure + on-demand H2D
fetch, and the graph is otherwise unchanged. Built on `engine/moe-slot-cache`.

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
- CLI/context: `--moe-slot-cache N` / `-msc N` (`llama_context_params.moe_slot_cache`).
- Stats: `LLAMA_MOE_SLOT_STATS=1` prints hit/fetch totals + hit rate at context exit.

Not in this cut: CPU overflow for capped fetches (the q* hybrid split), prefill
double-buffered streaming, runtime VRAM reallocation, FTW. The decode path is
FreeToken's "gpu" mode: every miss is fetched, nothing is computed on the CPU.

## Results

### 35B-A3B IQ2_M (qwen35moe, 256 experts top-8, 40 layers), one RX 7900 XT

185-token prompt, 64-token greedy, `-c 512 -np 1 -t 8`. Cache arm:
`-cmoe -msc N` (all expert banks host-resident, slots on GPU). Outputs are
**token-identical to the all-GPU run** (same kernels, copied bytes).

| config | expert VRAM | pp t/s | tg t/s | hit rate |
|---|---:|---:|---:|---:|
| `-ngl 99` all GPU | ~13.5 GiB | 756.6 | 108.9 | – |
| `-cmoe` all CPU | 0 | 224.9 | 37.8 | – |
| `-cmoe -msc 32` | 1.24 GiB | 219.8 | 42.9 | – |
| `-cmoe -msc 48` | 1.86 GiB | 222.6 | 53.6 | – |
| `-cmoe -msc 64` | 2.48 GiB | 222.0 | 60.3 | 75.9% |
| `-ncmoe 28` (12 layers GPU) | ~9.8 GiB | 285.8 | 55.3 | – |
| `-ncmoe 24` (16 layers GPU) | ~13.1 GiB | 307.9 | 46.8 | – |
| `-ncmoe 20` (20 layers GPU) | ~16.3 GiB | 346.1 | 56.5 | – |

Reading: **2.5 GiB of harnessed VRAM buys 60 t/s**, while whole-layer placement needs
~10–16 GiB for the same league and still trails at equal-ish budget. The hit rate at
25% residency (64/256 slots) is ~76% — the routing skew FreeToken bets on is real on
this model.

### 125B Qwen3.8-Flash-Next GSQ-RCO Q2_0 (512 experts top-10, 48 layers), 2× 7900 XT

Same prompt/bench, `-c 4096 -ts 0.522,0.478 --split-mode layer -lm mmap
--lazy-mode on -fa auto`, headless-first (`HIP_VISIBLE_DEVICES=1,0`).

| config | resident expert VRAM | slot cache | pp t/s | tg t/s | hit rate |
|---|---:|---:|---:|---:|---:|
| `-ncmoe 12` | 23.6 GiB | – | 54.3 | 19.64 | – |
| `-ncmoe 12 -msc 128` | 23.6 GiB | 2.02 GiB | 47.1 | **23.33** | 62.0% |
| `-ncmoe 24` | 15.8 GiB | – | 29.9 | 14.61 | – |
| `-ncmoe 24 -msc 256` | 15.8 GiB | 8.10 GiB | 29.8 | **18.95** | 63.5% |

Reference all-resident (no offload): ~26–28 t/s decode, ~460–497 t/s prefill.

Reading: with 12 layers' experts (8.1 GiB) off-VRAM, decode recovers to ~90% of the
all-resident rate (+19% over plain `-ncmoe 12`); doubling the offload to 24 layers
(+30% over `-ncmoe 24`) with 8 GiB of cache. Prefill still runs the cached layers on
CPU (decode-only cut) — that is the next lever.

## Next

1. **Prefill streaming**: when a cached layer's prefill ubatch routes more experts
   than slots, stream the layer through the slot band with double buffering instead
   of falling back to CPU compute (FreeToken's `prefill_overlap`).
2. **q\* hybrid**: cap fetches per step (`fetch_fraction = pcie_bw / cpu_bw`), compute
   the overflow experts on the CPU, and merge the partials; needs the masked second
   mmid pass and a `-1`-safe CPU path.
3. **Telemetry**: expose the hit/fetch counters through the server API (`/metrics`)
   and the Hivebench tier status instead of a debug log.
4. **Runtime cache resize** from free VRAM (FreeToken's `--moe-cache-auto` +
   `rebuild`).

## Reproduce

```sh
# build
cmake --build build-hip -j 12 --target llama-server

# 35B arm (one card)
HIP_VISIBLE_DEVICES=1 ./build-hip/bin/llama-server -m <35B-iq2_m.gguf> \
  -ngl 99 -cmoe -msc 64 -c 512 -np 1 -t 8

# 125B arm (both cards, headless first)
HIP_VISIBLE_DEVICES=1,0 LLAMA_MOE_SLOT_STATS=1 ./build-hip/bin/llama-server \
  -m ~/Desktop/work/models/qwen38-q2_0/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  -ngl 99 -ncmoe 12 -msc 128 -c 4096 -np 1 -t 8 \
  -ts 0.522,0.478 --split-mode layer -lm mmap --lazy-mode on -fa auto
```
