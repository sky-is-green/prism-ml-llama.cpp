# Prism base results (2026-09-28)

The rig now runs on `engine/amd-rig` = prism `master` + current upstream +
HSA bounce upload + `LLAMA_GRAPH_DUMP` + `tools/engine-rig/`.

## 125B, dual-resident (2x 7900 XT, `-sm layer -ts 0.53,0.47`)

- load 44 s (vs 57-90 s on the taardis base)
- prefill 460-497 t/s, decode 24-28 t/s, GPU busy ~39%
- `bench2.sh` (server on :8090, 167-token prompt):
  pp 108.98 / 496.86 / 459.76 / 20.53 t/s; tg 27.62 / 27.10 / 24.13 / 26.76 t/s
- coherent; greedy reference regenerated for this base (the taardis-base
  reference differs after ~8 tokens, as expected from upstream kernel changes)

## Superseded: the custom MoE expert-sum fusion

Current upstream already folds the MoE expert-aggregation tail into one kernel:
`ggml_cuda_match_moe_weighted_reduction` + `ggml_cuda_op_moe_weighted_reduction`
(superset of the taardis-base kernel: it also handles the
`(experts * expert_scale) * router_weight` form). It fires first, so the
cherry-picked fusion became dead code on this branch and was reverted.

The custom kernel is kept on `engine/moe-expert-sum` (based on the legacy
`moe-corr-runtime` line, which predates the upstream fusion) for the forward
port of that branch.

## Harness notes

- `hc_test` compares the MoE subgraph GPU vs CPU and fails on real numerical
  differences (tolerance); GPU-vs-CPU rounding differs by ~1e-7 ulps on this
  base, as before.
- `GGML_CUDA_DISABLE_FUSION` (and the per-fusion kill switches) are read once
  per process, so fused-vs-unfused bit comparisons require two process runs.
