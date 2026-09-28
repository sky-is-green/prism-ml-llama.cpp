# Repo map (2026-09-28 consolidation)

The AMD engine work lives here. This note records which repo/branch is which so
a previous configuration cannot recur.

## Engine repo: `sky-is-green/prism-ml-llama.cpp` (this repository)

| branch | role |
|---|---|
| `master` | upstream llama.cpp + fork docs/CI; carries the model support used by the rig (`qwen4exp`, `--lazy-mode`/PLE) |
| `engine/amd-rig` | **the AMD test bed**: `master` merged current upstream + the three engine commits + `tools/engine-rig/` |
| `engine/moe-expert-sum` | the same three engine commits on the legacy engine line (for merging into `moe-corr-runtime`) |
| `moe-corr-runtime` | legacy engine line (dflash/dspark, hybrid-attention fusions, PQ2_0 port); based on an older upstream, forward-port onto `engine/amd-rig` pending |

The three engine commits: HSA bounce upload (`set_tensor` staging), MoE
expert-aggregation fusion, `LLAMA_GRAPH_DUMP`.

## Research/quantization fork: `CodeMasterCody3D/taardis-llama.cpp`

Ternary formats/recipes (Q1_0_g128, PQ2_0), TAARDIS releases, and the original
`expert-cache` branch. Its local clone is `~/Desktop/work/ternary-serve/taardis-llama.cpp`.
The shelved expert-cache sidecar commits exist only there (push from this account
is denied; they are preserved in the local clone and superseded by the shelved
verdict in `HANDOFF-EXPERT-CACHE.md`).

## Local layout (2026-09-28)

- Engine clone: `~/Desktop/work/prism-ml-llama.cpp`, branch `engine/amd-rig`,
  built in `build-hip` (llama-server / llama-perplexity / llama-cli).
- The taardis clone and the old rig `build-hip` were deleted; the taardis-only
  sidecar commits are preserved here as `archive/taardis-expert-cache`.
- `~/Desktop/work/ternary-serve/` is retired: only small measurement logs and
  the handoff docs remain (the rig moved here).

## Naming trap that caused the original mix-up

The taardis clone's `origin` URL is the pre-rename name of
`CodeMasterCody3D/taardis-llama.cpp` (`.../prism-ml-llama.cpp`), and GitHub
redirects git traffic, so `git remote -v` there looks like a prism-ml clone.
