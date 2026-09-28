#include "llama-moe-slot-cache.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-backend.h"

#include <cinttypes>

llama_moe_slot_cache::llama_moe_slot_cache() = default;

llama_moe_slot_cache::~llama_moe_slot_cache() {
    // tensors live inside these contexts/buffers — free the buffers first
    for (ggml_backend_buffer_t buf : bufs) {
        ggml_backend_buffer_free(buf);
    }
    for (ggml_context * ctx : ctxs) {
        ggml_free(ctx);
    }
}

int llama_moe_slot_cache::init(const llama_model & model, int n_slots) {
    if (n_slots <= 0) {
        return 0;
    }

    struct candidate {
        int il;
        ggml_backend_dev_t dev;
        const ggml_tensor * gu;    // fused gate_up bank, or null
        const ggml_tensor * gate;  // separate gate bank, or null
        const ggml_tensor * up;
        const ggml_tensor * down;
    };

    std::vector<candidate> candidates;

    const int n_layer = (int) model.hparams.n_layer();
    for (int il = 0; il < n_layer; ++il) {
        const llama_layer & l = model.layers[il];

        const ggml_tensor * gu   = l.ffn_gate_up_exps;
        const ggml_tensor * gate = l.ffn_gate_exps;
        const ggml_tensor * up   = l.ffn_up_exps;
        const ggml_tensor * down = l.ffn_down_exps;

        if (down == nullptr || (gu == nullptr && (gate == nullptr || up == nullptr))) {
            continue;
        }
        if (l.ffn_gate_up_exps_b != nullptr || l.ffn_down_exps_b != nullptr ||
            l.ffn_up_exps_s != nullptr || l.ffn_gate_exps_s != nullptr || l.ffn_down_exps_s != nullptr) {
            LLAMA_LOG_WARN("%s: layer %d has expert scales/biases; the MoE slot cache does not support them - skipping\n", __func__, il);
            continue;
        }

        // all present expert banks must be host-resident (that is the point of the cache)
        const ggml_tensor * banks[4] = { gu, gate, up, down };
        bool host = true;
        int64_t n_expert = 0;
        for (const ggml_tensor * t : banks) {
            if (t == nullptr) {
                continue;
            }
            if (t->buffer == nullptr || !ggml_backend_buffer_is_host(t->buffer)) {
                host = false;
                break;
            }
            if (n_expert == 0) {
                n_expert = t->ne[2];
            } else if (n_expert != t->ne[2]) {
                LLAMA_LOG_WARN("%s: layer %d expert counts differ (%" PRId64 " vs %" PRId64 ") - skipping\n",
                               __func__, il, n_expert, t->ne[2]);
                host = false;
                break;
            }
        }
        if (!host) {
            continue;
        }

        candidates.push_back({ il, model.dev_layer(il), gu, gate, up, down });
    }

    if (candidates.empty()) {
        return 0;
    }

    // one context + one backend buffer per device, created up front so a single
    // allocation covers every bank that lands on that device
    std::map<ggml_backend_dev_t, ggml_context *> dev_ctx;
    std::map<ggml_backend_dev_t, std::vector<ggml_moe_cache_layer *>> dev_layers;

    auto ctx_for = [&](ggml_backend_dev_t dev) -> ggml_context * {
        const auto it = dev_ctx.find(dev);
        if (it != dev_ctx.end()) {
            return it->second;
        }
        ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead() * (size_t) 4 * (size_t) candidates.size() + 4096,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context * ctx = ggml_init(params);
        if (ctx == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to create the slot-cache context\n", __func__);
            return nullptr;
        }
        dev_ctx[dev] = ctx;
        ctxs.push_back(ctx);
        return ctx;
    };

    for (const candidate & c : candidates) {
        ggml_context * ctx = ctx_for(c.dev);
        if (ctx == nullptr) {
            continue;
        }

        auto lc = std::make_unique<ggml_moe_cache_layer>();
        lc->il       = c.il;
        lc->n_expert = (int32_t) c.down->ne[2];
        lc->n_slots  = n_slots;
        lc->device   = (int32_t) dev_layers[c.dev].size(); // informational only

        auto make_bank = [&](const ggml_tensor * src, const char * what) -> ggml_tensor * {
            if (src == nullptr) {
                return nullptr;
            }
            ggml_tensor * bank = ggml_new_tensor_3d(ctx, src->type, src->ne[0], src->ne[1], n_slots);
            ggml_format_name(bank, "moe_slot_bank_%s_%d", what, c.il);
            return bank;
        };

        lc->bank_gu   = make_bank(c.gu,   "gu");
        lc->bank_gate = make_bank(c.gate, "gate");
        lc->bank_up   = make_bank(c.up,   "up");
        lc->bank_down = make_bank(c.down, "down");

        lc->src_gu   = c.gu;
        lc->src_gate = c.gate;
        lc->src_up   = c.up;
        lc->src_down = c.down;

        lc->bytes_gu   = c.gu   ? c.gu->nb[2]   : 0;
        lc->bytes_gate = c.gate ? c.gate->nb[2] : 0;
        lc->bytes_up   = c.up   ? c.up->nb[2]   : 0;
        lc->bytes_down = c.down->nb[2];

        lc->slot_for_id.assign(lc->n_expert, -1);
        lc->id_of_slot.assign(n_slots, -1);
        lc->usage.assign(n_slots, 0);

        by_il[c.il] = lc.get();
        dev_layers[c.dev].push_back(lc.get());
        layers.push_back(std::move(lc));
    }

    // allocate every device's banks in one buffer, then validate the geometry
    for (auto & [dev, dev_ls] : dev_layers) {
        ggml_context * ctx = dev_ctx[dev];
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_dev_buffer_type(dev));
        if (buf == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to allocate %zu slot banks (%d slots each) on device '%s'\n",
                            __func__, dev_ls.size(), n_slots, ggml_backend_dev_name(dev));
            // leave the feature off rather than half-initialized
            by_il.clear();
            layers.clear();
            for (ggml_backend_buffer_t b : bufs) {
                ggml_backend_buffer_free(b);
            }
            bufs.clear();
            for (ggml_context * c : ctxs) {
                ggml_free(c);
            }
            ctxs.clear();
            return 0;
        }
        bufs.push_back(buf);

        for (ggml_moe_cache_layer * lc : dev_ls) {
            const std::pair<ggml_tensor *, size_t> pairs[4] = {
                { lc->bank_gu,   lc->bytes_gu   },
                { lc->bank_gate, lc->bytes_gate },
                { lc->bank_up,   lc->bytes_up   },
                { lc->bank_down, lc->bytes_down },
            };
            for (const auto & [bank, bytes] : pairs) {
                if (bank == nullptr) {
                    continue;
                }
                if (bank->data == nullptr) {
                    LLAMA_LOG_ERROR("%s: slot bank allocation did not set tensor data\n", __func__);
                    continue;
                }
                GGML_ASSERT(bank->nb[2] == bytes);
            }
        }
    }

    slots = n_slots;

    size_t total_bytes = 0;
    for (ggml_backend_buffer_t b : bufs) {
        total_bytes += ggml_backend_buffer_get_size(b);
    }

    fprintf(stderr, "%s: MoE slot cache: %zu layer(s), %d slots each, %.1f MiB on device(s)\n",
            __func__, layers.size(), n_slots, (double) total_bytes / (1024.0 * 1024.0));
    LLAMA_LOG_INFO("%s: MoE slot cache: %zu layer(s), %d slots each, %.1f MiB on device(s)\n",
                   __func__, layers.size(), n_slots, (double) total_bytes / (1024.0 * 1024.0));

    return (int) layers.size();
}

std::string llama_moe_slot_cache::stats() const {
    if (layers.empty()) {
        return "MoE slot cache: off";
    }

    int64_t hits = 0;
    int64_t fetches = 0;
    for (const auto & lc : layers) {
        hits    += lc->n_hits;
        fetches += lc->n_fetches;
    }
    const double total = (double) (hits + fetches);
    const double rate = total > 0.0 ? 100.0 * (double) hits / total : 0.0;

    char buf[256];
    snprintf(buf, sizeof(buf),
             "MoE slot cache: %zu layer(s) x %d slots, %" PRId64 " hits / %" PRId64 " fetches (%.1f%% hit rate)",
             layers.size(), slots, hits, fetches, rate);
    return std::string(buf);
}
