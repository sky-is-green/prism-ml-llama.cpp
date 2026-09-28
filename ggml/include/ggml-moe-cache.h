// MoE expert slot cache — shared state for the FreeToken-style expert residency path.
//
// The host bank (the layer's mmap'd/CPU expert tensors) stays where the loader put it;
// a GPU band of `n_slots` expert rows per bank is the residency target. Every decode
// step the routed expert ids are mapped to slot rows, misses are copied host->device
// on the compute stream, and the LRU bookkeeping is updated in place (see
// ggml_moe_cache_map). This header is internal: ggml-cuda and llama.cpp both include it,
// it is not part of the public ggml API.
#pragma once

#include "ggml.h"

#include <cstdint>
#include <vector>

// One MoE layer's cache-managed expert banks. `bank_*` are device tensors of the same
// type/shape as the host tensors but with ne[2] = n_slots; `src_*` are the host tensors
// (expert-major, stride nb[2]). The maps are host state: they are only touched by the
// op body, which executes once per graph step.
struct ggml_moe_cache_layer {
    int32_t il = -1;
    int32_t n_expert = 0;
    int32_t n_slots = 0;
    int32_t device = -1;

    // device slot banks — one per expert bank, shaped [ne0, ne1, n_slots].
    // Fused models set bank_gu; separate-gate models set bank_gate/bank_up.
    ggml_tensor * bank_gu = nullptr;    // ffn_gate_up_exps (fused gate+up)
    ggml_tensor * bank_gate = nullptr;  // ffn_gate_exps
    ggml_tensor * bank_up = nullptr;    // ffn_up_exps
    ggml_tensor * bank_down = nullptr;  // ffn_down_exps

    // host sources — shaped [ne0, ne1, n_expert]
    const ggml_tensor * src_gu = nullptr;
    const ggml_tensor * src_gate = nullptr;
    const ggml_tensor * src_up = nullptr;
    const ggml_tensor * src_down = nullptr;

    size_t bytes_gu = 0;    // bytes per expert row, gate_up bank
    size_t bytes_gate = 0;
    size_t bytes_up = 0;
    size_t bytes_down = 0;

    // LRU state, indexed by expert id / slot
    std::vector<int32_t> slot_for_id;  // [n_expert] slot index, -1 when not resident
    std::vector<int32_t> id_of_slot;   // [n_slots] expert id, -1 when free
    std::vector<int64_t> usage;        // [n_slots] last-use step
    int64_t step = 0;

    // true when n_slots >= n_expert: the bank is fully resident at identity slots, so
    // the graph uses it directly with the original ids (no per-step map op at all)
    bool identity = false;

    // cumulative counters (host side, for stats)
    int64_t n_hits = 0;
    int64_t n_fetches = 0;
};
