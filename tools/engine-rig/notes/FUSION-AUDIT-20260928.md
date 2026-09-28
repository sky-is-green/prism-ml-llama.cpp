# phase1 uncommitted work: audit + disposition (2026-09-28 ~11:15)

The branch head `2b521be01` had a dirty tree with several pieces of in-flight
work. This note records what each piece was, whether it works, and where the
experimental parts are preserved.

Saved diffs:
- `uncommitted-20260928-1115.patch` — the whole dirty tree as found this morning
  (drop-in reference, NOT meant to be applied wholesale).
- `phase1-all-changes.patch` (2026-09-27 22:59) — earlier snapshot that contains
  the first HC-combine fusion (`hc_combine_fused_f32`) that was later removed
  from the tree.

## Kept and committed

1. **HSA bounce upload** (`ggml/src/ggml-cuda/ggml-cuda.cu`,
   `ggml_backend_cuda_buffer_set_tensor`). Stages `set_tensor` copies > 64 MiB
   through one reused pageable buffer. Verified in `hsa-repro/` and in the real
   dual-GPU 125B load (`RESULTS.md`): avoids the `hipMemcpyAsync` HSA stall.
   Committed as its own commit. Follow-up: consider `set_tensor_2d`, the
   pinned-host upload paths, and `ggml_start_upload` for the same treatment.
2. **`LLAMA_GRAPH_DUMP`** (`src/llama-context.cpp`, `dump_graph_ops`). Op
   histogram + optional per-node dump at `sched_reserve`. Used by the
   graphbench workflow (`LLAMA_GRAPH_DUMP=2`). Committed separately.

## Dropped from the tree (preserved in `uncommitted-20260928-1115.patch`)

3. **HC stream collapse fusion** (`hc_collapse_fused_f32` in `dsv4-hc.cu` +
   matcher in `ggml-cuda.cu`, pattern
   `{RESHAPE, VIEW, CONT, VIEW, ADD, VIEW, ADD, VIEW, ADD, SCALE}`).
   It **never fires**, for two independent reasons:
   - `ggml_can_fuse_subgraph_ext` requires every non-output node's `view_src`
     chain to be inside the fused window (or constant). Its first node is a
     RESHAPE, itself a view of the pre-reshape `gated` MUL which lies outside
     the window. Diagnosis: `canfuse4.log` shows `[canfuse] external view
     source at idx 10` (the collapse window at node 10).
   - Even if the matcher ran, its shape test is wrong (`in` is taken as
     `nodes[i]->src[0]` = the pre-reshape `[hc*n_embd, nt]` tensor, so
     `in->ne[0] == out->ne[0]` cannot hold), and the kernel receives
     `cont_node->src[0]` (a non-contiguous 2D VIEW) while asserting
     `ggml_is_contiguous(in)` and interpreting it as a contiguous
     `[n_embd, hc, nt]` tensor.
   - `hc_test`'s `hc_collapse PASS` is **vacuous**: the fusion never fires
     there either (same view_src failure), so the test compares unfused vs
     unfused. Any future subgraph test must assert that the fusion fired
     (e.g. via `GGML_CUDA_FUSION_LOG=1`).

4. **SCALE + SILU fusion** (`scale_silu_fused_f32` + matcher
   `{SCALE, UNARY(SILU)}`). Fires 300+ times per request but **aborts**
   (`hipErrorInvalidConfiguration`, "invalid configuration argument" from
   `ggml_cuda_kernel_launch`) on the first long prefill (nt ~ 167); the
   decode/coherence warmup (small nt) passes. Root cause not identified;
   `GGML_CUDA_PDL=0` does not help (PDL is compile-time off for HIP anyway).
   Logs: `bench7.log`, `bench9-fused.log`, `benchA-fused.log`, `bench10.log`,
   `bench11.log` (all abort in `ggml_cuda_op_scale_silu_fused`).

5. **Weakened `ggml_cuda_check_fusion_memory_ranges`** (skip internal
   intermediates). This is a shared safety check used by every backend fusion
   at HEAD; the change lets fused kernels read inputs that may alias the
   destination. Dropped for safety.

6. **`ggml_graph_find_tensor`** (ggml.h/ggml.c) and **`GGML_CANFUSE_DEBUG`**
   FAIL-reason logging. Only existed to support (3)/(5). Preserved in the
   patch; the reason logging is cheap to re-add if fusion work restarts.

7. **Loop anchor change** (call `ggml_cuda_try_fuse` for RESHAPE nodes inside
   `ggml_cuda_graph_evaluate_and_capture`). Only needed to reach the collapse
   matcher; dropped with it.

## Why the drop is low-cost

Per `graphbench/README.md`, Phase 1 was already revised: elementwise launch
count is not the bottleneck (the first HC-combine fusion was bit-exact and
performance-neutral). The collapse would save ~12 `[n_embd, nt]` tensor moves
per layer per token (≈ 6 MB/token at nt=1, ≈ 7 µs at 800 GB/s) — noise
against the ~40 ms/token budget. The real targets remain: heavy-op kernel
time (MUL_MAT/MUL_MAT_ID/attention) and the ~22 ms/token of non-GPU time.
