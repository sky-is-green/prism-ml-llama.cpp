// Correctness harness for backend fusions on ROCm/CUDA.
//
// Covers the MoE expert-aggregation tail: ffn_moe_out's MUL + per-expert VIEWs
// + serial ADDs chain. On current upstream this is folded into a single kernel
// by ggml_cuda_op_moe_weighted_reduction; on engine branches that carry a
// custom variant it is bit-exact by construction. This harness runs the exact
// subgraph on CPU and GPU and fails on a real numerical difference (tolerance);
// for bit-exact fused-vs-unfused runs, run the process twice with
// GGML_CUDA_DISABLE_FUSION=1 (the kill switch is read once per process).
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

        if (getenv("HC_TEST_BACKEND_DEBUG")) {
            ggml_backend_t bm = ggml_backend_sched_get_tensor_backend(sched, mul);
            ggml_backend_t bv = ggml_backend_sched_get_tensor_backend(sched, views[0]);
            ggml_backend_t bo = ggml_backend_sched_get_tensor_backend(sched, out);
            fprintf(stderr, "[hc] mul=%s view0=%s out=%s\n",
                    bm ? ggml_backend_name(bm) : "none", bv ? ggml_backend_name(bv) : "none",
                    bo ? ggml_backend_name(bo) : "none");
        }

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

    const auto ref     = run(cpu, cpu, experts, weights);
    const auto gpu_res = run(gpu, cpu, experts, weights);

    size_t exact_diff = 0;
    double max_delta = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = std::fabs((double) ref[i] - (double) gpu_res[i]);
        if (d != 0) ++exact_diff;
        max_delta = std::max(max_delta, d);
    }
    const bool ok = max_delta <= 1e-5;
    printf("moe_sum        n_embd=%-5lld n_used=%-3lld nt=%-4lld: exact_diff=%zu max_delta=%.3g %s\n",
           (long long) n_embd, (long long) n_used, (long long) nt, exact_diff, max_delta,
           ok ? "PASS" : "FAIL");
    return ok;
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
