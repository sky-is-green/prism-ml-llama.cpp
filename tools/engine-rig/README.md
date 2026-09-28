# AMD engine rig

Development harness for the AMD/ROCm MoE serving engine work (FreeToken/Strata
style: hot-expert residency, kernel-count reduction, dispatch budget). Everything
here runs on the consumer box (2x RX 7900 XT, gfx1100, ROCm).

## Layout

| file | what it does |
|---|---|
| `graphbench.cpp` | HIP graph-replay microbenchmark: host-launch vs captured graph vs fused-per-layer, same node count/bytes as the real decode graph |
| `hc_test.cpp` | correctness harness for the MoE expert-aggregation fusion (CPU vs GPU fused vs GPU unfused, bit-exact) |
| `hsa_upload.cpp` | minimal repro for the ROCm/HSA `hipMemcpyAsync` stall on many distinct host ranges (the dual-GPU model-load hang) |
| `scripts/run-125b.sh` | start the dual-resident 125B server (`MODEL=... scripts/run-125b.sh <tag> [env...]`) |
| `scripts/bench2.sh` | compact prefill/decode/cached-decode bench against a running server |
| `scripts/decode-busy.py` | decode while sampling GPU busy% per card |
| `scripts/greedy-check.py` | greedy determinism check against `scripts/reference-greedy.jsonl` |
| `notes/BUDGET-20260928.md` | measured per-token budget and the MoE fusion A/B |

## Build (from the repo root)

```sh
cmake -S . -B build-hip -DGGML_HIP=ON -DGGML_HIP_GRAPHS=ON -DGGML_NATIVE=ON \
      -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DCMAKE_HIP_ARCHITECTURES=gfx1100
cmake --build build-hip -j 12 --target llama-server llama-perplexity llama-cli

hipcc -O2 -o graphbench tools/engine-rig/graphbench.cpp
hipcc -O2 -o hsa_upload tools/engine-rig/hsa_upload.cpp -pthread
hipcc -O2 -o hc_test tools/engine-rig/hc_test.cpp -I ggml/include -L build-hip/bin \
    -lggml -lggml-base -lggml-cpu -lggml-hip -Wl,-rpath,$PWD/build-hip/bin
```

## Correctness

```sh
GGML_LIB_DIR=$PWD/build-hip/bin HIP_VISIBLE_DEVICES=1 ./hc_test
```

`hc_test` runs the MoE expert-aggregation subgraph on CPU and GPU and fails on a
real numerical difference (tolerance). Note that on current upstream this
subgraph is folded into one kernel (`ggml_cuda_op_moe_weighted_reduction`), and
the kill switch for fusions (`GGML_CUDA_DISABLE_FUSION`) is read once per
process - so bit-exact fused-vs-unfused comparisons need two process runs, not
an in-process A/B.

## 125B smoke bench (dual-resident)

```sh
MODEL=/path/to/Qwen3.8...-00001-of-00002.gguf scripts/run-125b.sh baseline
WIKI=/path/to/wiki.test.raw bash scripts/bench2.sh 8090 baseline
python3 scripts/decode-busy.py 8090 256 baseline
python3 scripts/greedy-check.py 8090 /tmp/greedy.jsonl baseline
```

Reference numbers on this box: ~24.5 t/s decode / 300-380 t/s prefill, ~50% GPU
busy, ~40 ms/token. See `notes/BUDGET-20260928.md` for the budget and the
`GGML_CUDA_DISABLE_GRAPHS=1` control.

## Gotchas learned the hard way

- `hc_test` silently ran on CPU for a while: `ggml_backend_sched_reserve()`
  resets user backend assignments, so `ggml_backend_sched_set_tensor_backend()`
  must be called *after* reserve and *before* alloc (and leaves alone do not seed
  the assignment pass).
- `ggml_cuda_check_fusion_memory_ranges()` skips leaf inputs (`GGML_OP_NONE`),
  but the scheduler can reuse an input buffer for a fused output. Any fusion
  that reads an input while writing the output needs its own alias guard.
- HIP contracts `__fmul_rn` + `__fadd_rn` into FMA: kernels that must match an
  unfused MUL+ADD chain are compiled with `-ffp-contract=off` (see
  `mmid.cu` / `ggml-hip/CMakeLists.txt`).
- Graph capture vs host loop on this box costs only ~10% (`~4.3 ms/step`);
  per-kernel GPU-side serialization (~3.5 us/kernel) is the real budget. Fuse
  kernels; don't chase replay overhead.
