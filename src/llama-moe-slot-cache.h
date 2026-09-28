// MoE expert slot cache (FreeToken-style) — llama-side owner of the per-layer slot banks.
//
// Enabled with `--moe-slot-cache N` (context param `moe_slot_cache`). Every MoE layer
// whose expert banks were placed on the host (e.g. via `-ncmoe` / `--override-tensor`)
// gets `N` GPU slots per bank; decode graphs route through the slot banks and misses
// are fetched on demand (ggml_moe_cache_map). See the engine rig notes for the design.
#pragma once

#include "ggml-moe-cache.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

struct llama_model;
struct ggml_context;

typedef struct ggml_backend_buffer * ggml_backend_buffer_t;

class llama_moe_slot_cache {
public:
    llama_moe_slot_cache();
    ~llama_moe_slot_cache();

    // Creates a slot bank for every MoE layer whose expert banks are host-resident.
    // Returns the number of cached layers (0 => feature stays off).
    int init(const llama_model & model, int n_slots);

    bool enabled() const { return !layers.empty(); }

    ggml_moe_cache_layer * layer(int il) {
        const auto it = by_il.find(il);
        return it == by_il.end() ? nullptr : it->second;
    }

    int n_slots() const { return slots; }

    std::string stats() const;

private:
    int slots = 0;
    std::vector<std::unique_ptr<ggml_moe_cache_layer>> layers;
    std::map<int, ggml_moe_cache_layer *> by_il;
    std::vector<ggml_context *> ctxs;
    std::vector<ggml_backend_buffer_t> bufs;
};
