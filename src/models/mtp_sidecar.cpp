// The trained MTP head as a standalone draft model ("mtp-sidecar" GGUFs):
//   inputs : the target's head-input rows (the draft-mtp lane feeds them as the
//            batch's embd, n_embd wide) + the candidate token (embedded with the
//            target's own token_embd, shared via ctx_other)
//   head   : x = [rms(h), rms(e)] -> fc1 -> gelu -> t;
//            logits from rms(fc2 t) @ output.T (the target's output, also via
//            ctx_other), optional fc2-scale norm
//   state  : fc3 t predicts the target's NEXT head input row - the draft's own
//            nextn hidden, so the lane's chain stays in the target's h space
// Consumed by COMMON_SPECULATIVE_TYPE_DRAFT_MTP (n_mtp_layers = 1).
#include "models.h"

#include "llama-impl.h"

void llama_model_mtp_sidecar::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps, false);
    if (hparams.f_norm_rms_eps <= 0.0f) {
        hparams.f_norm_rms_eps = 1e-6f;
    }
    // everything else (n_embd, n_vocab, n_ctx_train, the dummy block count)
    // comes from the generic GGUF metadata; no trunk tensors are loaded
}

void llama_model_mtp_sidecar::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    // explicit shapes from the metas: fc1 [2*hidden, width], fc2 [hidden, width]
    auto * f1 = ml.get_tensor_meta("mtp.fc1.weight");
    if (f1 == nullptr) {
        throw std::runtime_error("mtp-sidecar requires mtp.fc1.weight [2*hidden, width]");
    }
    fc1 = create_tensor(tn(LLM_TENSOR_MTP_FC1, "weight"), { f1->ne[0], f1->ne[1] }, 0);
    fc2 = create_tensor(tn(LLM_TENSOR_MTP_FC2, "weight"), { n_embd, fc1->ne[1] }, 0);
    if (fc2 == nullptr) {
        throw std::runtime_error("mtp-sidecar requires mtp.fc2.weight [hidden, width]");
    }
    if (f1->ne[0] != 2 * n_embd || fc2->ne[1] != fc1->ne[1]) {
        throw std::runtime_error(format("mtp-sidecar shape mismatch: fc1 [%lld, %lld], fc2 [%lld, %lld], n_embd %lld",
                (long long) f1->ne[0], (long long) f1->ne[1], (long long) fc2->ne[0], (long long) fc2->ne[1],
                (long long) n_embd));
    }
    // the state head is optional (v1 sidecars are logits-only: usable for one
    // draft per round, or with a teacher-forced chain)
    fc3 = create_tensor(tn(LLM_TENSOR_MTP_FC3, "weight"), { n_embd, fc1->ne[1] }, TENSOR_NOT_REQUIRED);
    if (fc3 && fc3->ne[1] != fc1->ne[1]) {
        throw std::runtime_error("mtp-sidecar state-head width mismatch");
    }

    // the head-input width the NextN lane consumes: this model's n_embd
    hparams.n_embd_nextn_impl = hparams.n_embd;
    hparams.n_embd_out_impl   = hparams.n_embd;
}

std::unique_ptr<llm_graph_context> llama_model_mtp_sidecar::build_arch_graph(const llm_graph_params & params) const {
    switch (params.gtype) {
        case LLM_GRAPH_TYPE_DEFAULT:
        case LLM_GRAPH_TYPE_DECODER:
            return std::make_unique<graph>(*this, params);
        default:
            GGML_ABORT("invalid graph type for the mtp sidecar");
    }
}

llama_model_mtp_sidecar::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    const auto * m = static_cast<const llama_model_mtp_sidecar *>(&model);

    // both inputs matter and both are always present in the draft-mtp lane's
    // batches, so build them like the DFlash decoder does (build_inp_embd
    // selects one branch only)
    auto inp = std::make_unique<llm_graph_input_embd>(hparams.n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
    ggml_set_input(inp->tokens);
    res->t_inp_tokens = inp->tokens;

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, ubatch.n_tokens);
    ggml_set_input(inp->embd);

    res->add_input(std::move(inp));

    ggml_tensor * h = inp->embd;   // the target's head-input rows, n_embd wide
    GGML_ASSERT(ubatch.embd != nullptr && "the mtp-sidecar consumes embd rows (the draft-mtp lane's batches)");

    // the candidate token's embedding: the target's own token_embd (shared via ctx_other)
    auto * tok_embd = m->tok_embd;
    if (tok_embd == nullptr) {
        GGML_ASSERT(cparams.ctx_other != nullptr);
        const auto * model_other = llama_get_model(cparams.ctx_other);
        GGML_ASSERT(model_other->tok_embd != nullptr && "the mtp-sidecar requires the target model's token embeddings");
        tok_embd = model_other->tok_embd;
    }
    ggml_tensor * e = ggml_get_rows(ctx0, tok_embd, inp->tokens);

    ggml_tensor * out_ids = build_inp_out_ids();

    ggml_tensor * x = ggml_concat(ctx0,
            ggml_rms_norm(ctx0, h, hparams.f_norm_rms_eps),
            ggml_rms_norm(ctx0, e, hparams.f_norm_rms_eps), 0);
    cb(x, "mtp_x", -1);

    ggml_tensor * t = ggml_gelu(ctx0, ggml_mul_mat(ctx0, m->fc1, x));
    cb(t, "mtp_t", -1);

    ggml_tensor * z = ggml_mul_mat(ctx0, m->fc2, t);
    cb(z, "mtp_z", -1);

    if (out_ids) {
        z = ggml_get_rows(ctx0, z, out_ids);
    }
    res->t_h_nextn = z;   // without a state head the draft's own hidden is z

    // the target's output projection (shared via ctx_other)
    auto * output   = m->output;
    auto * output_s = m->output_s;
    if (output == nullptr) {
        GGML_ASSERT(cparams.ctx_other != nullptr);
        const auto * model_other = llama_get_model(cparams.ctx_other);
        GGML_ASSERT(model_other->output != nullptr && "the mtp-sidecar requires the target model's output projection");
        output   = model_other->output;
        output_s = model_other->output_s;
    }

    ggml_tensor * z_norm = ggml_rms_norm(ctx0, z, hparams.f_norm_rms_eps);
    if (m->norm) {
        z_norm = ggml_mul(ctx0, z_norm, m->norm);
    }
    ggml_tensor * logits = build_lora_mm(output, z_norm, output_s);
    cb(logits, "result_output", -1);
    res->t_logits = logits;

    // the state head: the draft's own nextn hidden row stays in the target's
    // head-input space, so the lane's chain can keep drafting with it
    if (m->fc3) {
        ggml_tensor * hnext = ggml_mul_mat(ctx0, m->fc3, t);
        cb(hnext, "mtp_h_next", -1);
        if (out_ids) {
            hnext = ggml_get_rows(ctx0, hnext, out_ids);
        }
        res->t_h_nextn = hnext;
    } else if (cparams.embeddings_nextn) {
        res->t_h_nextn = z;   // without a state head the draft's own hidden is z
    }

    ggml_build_forward_expand(gf, res->t_logits);
    if (res->t_h_nextn) {
        ggml_build_forward_expand(gf, res->t_h_nextn);
    }
}
