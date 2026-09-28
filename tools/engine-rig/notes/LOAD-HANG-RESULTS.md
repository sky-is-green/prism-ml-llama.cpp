# 125B smoke test → ROCm HSA upload hang: diagnosis + fix (2026-09-27 evening)

Executes step 1 of `HANDOFF-EXPERT-CACHE.md` ("125B single-GPU smoke test"),
then extends into the dual-GPU blocker the smoke test exposed.

## TL;DR

- Single-GPU tiering of the 125B works but is **RAM-bound** on this box:
  ~2–6 t/s decode / 4–28 t/s prefill sustained (page-cache/swap thrash).
- The config this box actually wants — **all 48 expert layers resident on both
  cards** — used to hang 100% of the time during load. Root cause found with a
  minimal repro: **ROCm/HSA stalls in `hipMemcpyAsync` (HostToDevice) after a
  few GiB of *distinct unpinned host source ranges***; llama.cpp's mmap path
  feeds exactly that pattern.
- Fix (verified in repro and in a real load): **bounce large `set_tensor`
  uploads through one reused 64 MiB pageable buffer**. Patch applied to
  `ggml/src/ggml-cuda/ggml-cuda.cu`, diff saved in
  `qwen125-smoke-20260927/hsa-repro/backend-set-tensor-bounce.patch`
  (uncommitted).
- Result: dual load succeeds in **57 s** and serves at **~25 t/s decode /
  ~385 t/s prefill**, stable — roughly **10× prefill and 5–10× decode** vs the
  best single-GPU numbers measured here.

## Setup

- Build: `build-hip` (HIP, ROCm 7.2.4, gfx1100). Server banner
  `0.3.0-dev (build 10785, commit 4f38fbd02)` (may lag branch head).
- Card mapping: **ROCm0 = card1 = display GPU** (desktop uses ~1.1–1.3 GiB),
  **ROCm1 = card0 = headless**. Verified via forced-alloc OOM + VRAM telemetry.
- Model anatomy (GGUF tensor table): 48 blocks, 512 experts top-10, expert FFN
  640, Q2_0 experts (ggml type 42, block 64) → **~675 MiB/layer, ~31.6 GiB**;
  `per_layer_token_embd.weight` **15.3 GiB** (lazy PLE); non-expert ~2.5 GiB.

## Part 1 — single-GPU tiering (RAM-bound)

Command: `-ngl 99 --n-cpu-moe 26 -c 4096 -np 1 -t 8 --split-mode none
--main-gpu 0 -lm mmap --lazy-mode on -fa auto` (ROCm0).

- loads in ~59–90 s; card1 ≈ 20.0 GiB total (model ≈ 18.9); coherent output
- warm short-context decode early in session: 17.5 t/s
- `bench2.sh` (167-token prompt, 128-token greedy, `cache_prompt:false`):
  prefill 4.35 → 10.78 → 28.22 t/s; decode 3.51 → 2.43 → 4.38 → 5.90 t/s
- evidence of thrash: `/proc/<pid>/io` 99 GB read for a ~50 GiB model;
  `vmstat` `bi` 300–980 MB/s, `si` up to 61 MB/s, `so` up to 74 MB/s
- no `-ncmoe` fixes it: VRAM caps GPU experts at ~22–23 layers, forcing
  ≥15.2 GiB of experts onto the CPU side of a 30 GB machine

Other single-GPU notes: `-ncmoe 25` on ROCm1 hung 3/3 loads (pre-fix);
`-ncmoe 26` ROCm1 1/2; `-ncmoe 26` ROCm0 2/3 (retries needed).

## Part 2 — the load hang: minimal repro and root cause

Repro: `hsa-repro/hsa_upload.cpp` (build with `hipcc -O2 -pthread`). It mirrors
`ggml_backend_cuda_buffer_set_tensor`: `hipSetDevice` → `hipMemcpyAsync(dst,
src, 675 MiB, HostToDevice, hipStreamPerThread)` → `hipStreamSynchronize`.

Findings (48×675 MiB, alternating two devices, unless noted):

| experiment | result |
|---|---|
| single device, up to 96 copies (64.8 GiB) | pass |
| dual, even split, ≥ ~19 GiB **read from source** | hang (chunk 28–29) |
| dual, 256 MiB chunks × 100 (25.6 GiB read) | hang at chunk 75 |
| dual, source cycling a 4 GiB file, 25.6 GiB VRAM touched | **pass** |
| dual, lopsided 15:1 / 1:15 (stays < ~19 GiB read) | pass |
| anon / pinned / `hipMemcpy` sync variants | hang identically |
| `HSA_ENABLE_SDMA=0` | worse: hang before allocation |
| dual, copies staged through one **reused** 675 MiB buffer | **pass** |
| dual, copies staged through one reused 64 MiB buffer | **pass** |

Interpretation: the stall is not about VRAM, buffer occupancy, pinning, DMA
mode, or page-cache pressure per se — it is triggered by **the cumulative
volume of distinct unpinned host source ranges passed to `hipMemcpyAsync`**.
Reusing one staging buffer (fixed host address) avoids it. The repro hangs in
the same place as the model load (main thread in
`libhsa-runtime64.so.1`, called from `libamdhip64.so.7` /
`ggml_backend_cuda_buffer_set_tensor`), confirmed with gdb
(`gdb-dual-ncmoe4.log` in the parent dir, `X-gdb-signature.log` for the repro).

Caution: repeated SIGKILLs of hung loads appear to degrade driver state
(threshold moved from ~19 GiB down to ~15.5 GiB over the session; single-device
runs that passed early later hung). Treat load hangs as expensive; prefer the
repro for experiments.

## The fix

`ggml/src/ggml-cuda/ggml-cuda.cu`, `ggml_backend_cuda_buffer_set_tensor`:
copies larger than 64 MiB are staged through one `thread_local` pageable
buffer (memcpy mmap→stage, then `hipMemcpyAsync` stage→device). Semantics
unchanged (copy is byte-identical); only the host source address pattern
changes. Diff: `hsa-repro/backend-set-tensor-bounce.patch` (36 lines).

## Part 3 — dual-GPU results with the fix

Command:
```
llama-server -m Qwen3.8-...00001-of-00002.gguf -ngl 99 -ncmoe 0 \
  -sm layer -ts 0.53,0.47 --device ROCm1,ROCm0 -c 4096 -np 1 -t 8 \
  -lm mmap --lazy-mode on -fa auto
```

- load 57 s; VRAM card0 19872 MiB / card1 18877 MiB (all 48 expert layers on
  GPU, CPU_Mapped 27.5 GiB for PLE/lazy)
- coherence: `"The capital of France is"` → `" Paris."`
- `bench2.sh`:

| pass | prefill | decode |
|---|---:|---:|
| warmup (8 tok) | 108.64 t/s | 25.05 t/s |
| r1 | 382.83 t/s | 23.94 t/s |
| r2 | 388.47 t/s | 25.38 t/s |
| cached decode | 29.88 t/s (4 tok) | 24.12 t/s |

- `vmstat` idle-heavy (CPU 86% idle), no swap churn.

## Recommendations

1. Keep the bounce patch (commit it or upstream); consider the same treatment
   for `set_tensor_2d` and the pinned-host-buffer upload paths, and check
   `ggml_start_upload`/async-upload ring behaviour for the same class of bug.
2. Prefer the dual-resident config for the 125B on this box; single-GPU
   tiering remains RAM-bound and is only for VRAM-constrained hosts.
3. Report the repro + stack to ROCm (gfx1100, ROCm 7.2.4) — it is a clean,
   self-contained driver/runtime bug with a deterministic trigger.
4. The expert cache remains orthogonal; nothing here changes its shelved status
   (`placement-sweep-20260927/THROUGHPUT.md`).

## Artifacts

- `hsa-repro/` — `hsa_upload.cpp` (+ built `hsa_upload`), `evictor.py`,
  `backend-set-tensor-bounce.patch`, all `*.log` matrices
- `bench.sh`, `bench2.sh` — server bench helpers; `gdb-run.sh` — hang-stack
  watchdog
- Logs: `server-*.log`, `bench2-*.log`, `vmstat-*.log`, `dual-*`, `gdb-*.log`
- The dual server used for the final numbers is left running on port 8090
  (pid in `dual-patched.pid.numeric`).
