// Correctness harness for backend fusions on ROCm/CUDA.
//
// Covers the MoE expert-aggregation fusion: ffn_moe_out's
// MUL + per-expert VIEWs + serial ADDs chain, replaced by a single kernel
// (ggml_cuda_op_moe_expert_sum_fused, mmid.cu). The exact subgraph runs on the
// CPU (reference), on the GPU with the fusion disabled, and on the GPU with the
// fusion enabled; all three must be bit-identical.
//
// Build (from the repo root):
//   hipcc -O2 -o hc_test tools/engine-rig/hc_test.cpp \
//       -I ggml/include -L build-hip/bin \
//       -lggml -lggml-base -lggml-cpu -lggml-hip -Wl,-rpath,$PWD/build-hip/bin
// Run:
//   GGML_LIB_DIR=$PWD/build-hip/bin HIP_VISIBLE_DEVICES=1 ./hc_test
//
// Notes for future fixtures:
// - force the GPU with ggml_backend_sched_set_tensor_backend(sched, t, gpu)
//   *after* reserve and *before* alloc: reserve() resets user assignments, and
//   assigning only leaves does not seed the assignment pass;
// - read kill switches per call (no static env cache), so the same process can
//   run fused/unfused A/B passes.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static bool test_moe_sum(ggml_backend_t gpu, ggml_backend_t cpu, int64_t n_embd, int64_t n_used, int64_t nt) {
    auto run = [&](ggml_backend_t backend, ggml_backend_t cpu_b,
                   const std::vector<float> & experts, const std::vector<float> & weights) {
        ggml_init_params ip = { 16 * 1024 * 1024, nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * t_e = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_used, nt);
        ggml_tensor * t_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, n_used, nt);
        ggml_set_input(t_e);
        ggml_set_input(t_w);

        ggml_tensor * mul = ggml_mul(ctx, t_e, t_w);
        ggml_set_name(mul, "ffn_moe_weighted");

        std::vector<ggml_tensor *> views(n_used);
        for (int64_t k = 0; k < n_used; ++k) {
            views[k] = ggml_view_2d(ctx, mul, n_embd, nt, mul->nb[2], k * mul->nb[1]);
        }

        ggml_tensor * out = views[0];
        for (int64_t k = 1; k < n_used; ++k) {
            out = ggml_add(ctx, out, views[k]);
        }
        ggml_set_name(out, "ffn_moe_out");
        ggml_set_output(out);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        // mirror build_moe_ffn: order the views before the adds, then expand all
        for (int64_t k = 0; k < n_used; ++k) {
            ggml_build_forward_expand(gf, views[k]);
        }
        ggml_build_forward_expand(gf, out);

        ggml_backend_t backends[2] = { backend, cpu_b };
        ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
        ggml_backend_sched_reserve(sched, gf);
        ggml_backend_sched_set_tensor_backend(sched, t_e, backend);
        ggml_backend_sched_set_tensor_backend(sched, t_w, backend);
        ggml_backend_sched_set_tensor_backend(sched, mul, backend);
        ggml_backend_sched_set_tensor_backend(sched, out, backend);
        ggml_backend_sched_alloc_graph(sched, gf);

        ggml_backend_tensor_set(t_e, experts.data(), 0, experts.size() * sizeof(float));
        ggml_backend_tensor_set(t_w, weights.data(), 0, weights.size() * sizeof(float));
        ggml_backend_sched_graph_compute(sched, gf);

        std::vector<float> res(ggml_nelements(out));
        ggml_backend_tensor_get(out, res.data(), 0, res.size() * sizeof(float));
        ggml_backend_sched_free(sched);
        ggml_free(ctx);
        return res;
    };

    std::mt19937 g(2024 + (int) n_used);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<float> experts(n_embd * n_used * nt), weights(n_used * nt);
    for (auto & v : experts) v = d(g);
    for (auto & v : weights) v = d(g);

    const auto ref = run(cpu, cpu, experts, weights);

    setenv("GGML_CUDA_NO_MOE_SUM_FUSE", "1", 1);
    const auto unfused = run(gpu, cpu, experts, weights);
    unsetenv("GGML_CUDA_NO_MOE_SUM_FUSE");
    const auto fused = run(gpu, cpu, experts, weights);

    size_t bad_f = 0, bad_u = 0;
    double max_f = 0, max_u = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double df = std::fabs((double) ref[i] - (double) fused[i]);
        const double du = std::fabs((double) ref[i] - (double) unfused[i]);
        if (df != 0) ++bad_f;
        if (du != 0) ++bad_u;
        max_f = std::max(max_f, df);
        max_u = std::max(max_u, du);
    }
    printf("moe_sum        n_embd=%-5lld n_used=%-3lld nt=%-4lld: fused_diff=%zu (max %.3g) unfused_diff=%zu (max %.3g) %s\n",
           (long long) n_embd, (long long) n_used, (long long) nt, bad_f, max_f, bad_u, max_u,
           (bad_f || bad_u) ? "FAIL" : "PASS");
    return bad_f == 0 && bad_u == 0;
}

int main() {
    ggml_backend_load_all_from_path(getenv("GGML_LIB_DIR") ? getenv("GGML_LIB_DIR") : "build-hip/bin");

    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_t gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!cpu || !gpu) {
        fprintf(stderr, "backend init failed (cpu=%p gpu=%p)\n", (void *) cpu, (void *) gpu);
        return 1;
    }
    printf("cpu=%s gpu=%s\n", ggml_backend_name(cpu), ggml_backend_name(gpu));

    int failures = 0;
    failures += !test_moe_sum(gpu, cpu, 2560, 10, 8);
    failures += !test_moe_sum(gpu, cpu, 2560, 10, 1);
    failures += !test_moe_sum(gpu, cpu, 2048,  4, 17);
    failures += !test_moe_sum(gpu, cpu,  512,  2, 5);
    failures += !test_moe_sum(gpu, cpu, 1280, 15, 3);
    failures += !test_moe_sum(gpu, cpu, 2560, 10, 512); // ubatch prefill
    failures += !test_moe_sum(gpu, cpu, 2560, 10, 163); // smoke-test prompt

    ggml_backend_free(cpu);
    ggml_backend_free(gpu);
    printf(failures ? "FAIL (%d cases)\n" : "ALL PASS\n", failures);
    return failures ? 1 : 0;
}
