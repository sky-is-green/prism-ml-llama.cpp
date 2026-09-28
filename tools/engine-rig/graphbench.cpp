// HIP graph-replay microbenchmark.
//
// Question: our 125B decode spends ~0.70 ms/layer of ~0.92 ms in host-side
// dispatch (~7,954 nodes/token at ~4.2 us/node). How much of that disappears
// if the node sequence is captured once into a HIP graph and replayed?
//
// This emits the same node count with configurable per-node work, and measures:
//   A) host launch loop (what llama.cpp's backend graph_compute does), sync per token
//   B) captured hipGraph replay, sync per token
//   C) captured hipGraph replay, sync once after many tokens (pure replay ceiling)
//
// build: hipcc -O2 -o graphbench graphbench.cpp
//
// usage: ./graphbench [--device N] [--layers 48] [--nodes-per-layer 166]
//                     [--tokens 64] [--kb 16] [--heavy 4] [--heavy-mult 64]
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define HIPCHECK(x) do { hipError_t _e = (x); if (_e != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(_e), __FILE__, __LINE__); exit(2); } } while (0)

static double now_s() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

__global__ void k_work(float * buf, int n, float a) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = buf[i];
        #pragma unroll 4
        for (int r = 0; r < 4; r++) v = fmaf(v, a, 0.5f);
        buf[i] = v;
    }
}

// fused: one launch processes a whole layer's worth of buffers
__global__ void k_fused(float * buf, long n, float a) {
    const long stride = (long)gridDim.x * blockDim.x;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        float v = buf[i];
        #pragma unroll 4
        for (int r = 0; r < 4; r++) v = fmaf(v, a, 0.5f);
        buf[i] = v;
    }
}

struct Node {
    float * ptr;
    int     n;
};

static void launch_nodes(hipStream_t st, const std::vector<Node> & nodes) {
    for (const auto & nd : nodes) {
        const int block = 256;
        const int grid  = (nd.n + block - 1) / block;
        hipLaunchKernelGGL(k_work, dim3(grid), dim3(block), 0, st, nd.ptr, nd.n, 1.000001f);
    }
}

int main(int argc, char ** argv) {
    int device = 1, layers = 48, nodes_per_layer = 166, tokens = 64, kb = 16, heavy = 4, heavy_mult = 64;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return std::atoi(argv[++i]); };
        if      (a == "--device")          device          = next();
        else if (a == "--layers")          layers          = next();
        else if (a == "--nodes-per-layer") nodes_per_layer = next();
        else if (a == "--tokens")          tokens          = next();
        else if (a == "--kb")              kb              = next();
        else if (a == "--heavy")           heavy           = next();
        else if (a == "--heavy-mult")      heavy_mult      = next();
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }

    hipDeviceProp_t prop;
    HIPCHECK(hipGetDeviceProperties(&prop, device));
    printf("device %d: %s\n", device, prop.name);
    HIPCHECK(hipSetDevice(device));

    const int total_nodes = layers * nodes_per_layer;
    const int light_n = kb * 1024 / (int)sizeof(float);
    const int heavy_n = light_n * heavy_mult;

    // one buffer per node, stable addresses (required for capture)
    std::vector<Node> nodes;
    nodes.reserve(total_nodes);
    size_t total_bytes = 0;
    for (int l = 0; l < layers; l++) {
        for (int i = 0; i < nodes_per_layer; i++) {
            const bool is_heavy = heavy > 0 && i < heavy;   // heavy nodes first in the layer
            nodes.push_back({ nullptr, is_heavy ? heavy_n : light_n });
            total_bytes += nodes.back().n * sizeof(float);
        }
    }
    printf("nodes/token=%d (heavy=%d/layer x%d work), bytes=%.1f MiB, tokens=%d\n",
           total_nodes, heavy, heavy_mult, total_bytes / 1048576.0, tokens);

    float * base = nullptr;
    HIPCHECK(hipMalloc(&base, total_bytes));
    size_t off = 0;
    for (auto & nd : nodes) { nd.ptr = base + off; off += nd.n; }
    HIPCHECK(hipMemset(base, 0, total_bytes));

    hipStream_t st;
    HIPCHECK(hipStreamCreate(&st));

    // ---- A: host launch loop ------------------------------------------------
    launch_nodes(st, nodes);
    HIPCHECK(hipStreamSynchronize(st));

    double t0 = now_s();
    for (int t = 0; t < tokens; t++) {
        launch_nodes(st, nodes);
        HIPCHECK(hipStreamSynchronize(st));
    }
    double t1 = now_s();
    const double a_ms = (t1 - t0) / tokens * 1e3;
    printf("A host launches : %8.2f ms/token  %8.1f tok/s  %6.1f ns/node\n",
           a_ms, 1e3 / a_ms, (t1 - t0) / (double)tokens / total_nodes * 1e9);

    // ---- B: captured graph replay -------------------------------------------
    hipGraph_t graph;
    HIPCHECK(hipStreamBeginCapture(st, hipStreamCaptureModeThreadLocal));
    launch_nodes(st, nodes);
    HIPCHECK(hipStreamEndCapture(st, &graph));

    hipGraphExec_t exec;
    HIPCHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    size_t n_graph_nodes = 0;
    HIPCHECK(hipGraphGetNodes(graph, nullptr, &n_graph_nodes));

    HIPCHECK(hipGraphLaunch(exec, st));
    HIPCHECK(hipStreamSynchronize(st));

    t0 = now_s();
    for (int t = 0; t < tokens; t++) {
        HIPCHECK(hipGraphLaunch(exec, st));
        HIPCHECK(hipStreamSynchronize(st));
    }
    t1 = now_s();
    const double b_ms = (t1 - t0) / tokens * 1e3;
    printf("B graph replay  : %8.2f ms/token  %8.1f tok/s  (graph nodes=%zu)\n",
           b_ms, 1e3 / b_ms, n_graph_nodes);
    printf("  speedup A/B   : %.2fx\n", a_ms / b_ms);

    // ---- C: replay without per-token host sync ------------------------------
    const int burst = tokens;
    t0 = now_s();
    for (int t = 0; t < burst; t++) {
        HIPCHECK(hipGraphLaunch(exec, st));
    }
    HIPCHECK(hipStreamSynchronize(st));
    t1 = now_s();
    const double c_ms = (t1 - t0) / burst * 1e3;
    printf("C replay burst  : %8.2f ms/token  %8.1f tok/s  (no per-token sync)\n",
           c_ms, 1e3 / c_ms);

    // ---- D/E: fused per-layer variant (same total work, 48x fewer launches) --
    std::vector<float *> layer_bufs(layers, nullptr);
    std::vector<size_t>  layer_elems(layers, 0);
    for (int l = 0; l < layers; l++) {
        size_t elems = (size_t)heavy * heavy_n + (size_t)(nodes_per_layer - heavy) * light_n;
        layer_elems[l] = elems;
        HIPCHECK(hipMalloc(&layer_bufs[l], elems * sizeof(float)));
        HIPCHECK(hipMemset(layer_bufs[l], 0, elems * sizeof(float)));
    }
    auto launch_fused = [&](hipStream_t s) {
        for (int l = 0; l < layers; l++) {
            const int block = 256;
            const int grid  = 256;
            hipLaunchKernelGGL(k_fused, dim3(grid), dim3(block), 0, s, layer_bufs[l], (long)layer_elems[l], 1.000001f);
        }
    };

    launch_fused(st);
    HIPCHECK(hipStreamSynchronize(st));
    t0 = now_s();
    for (int t = 0; t < tokens; t++) {
        launch_fused(st);
        HIPCHECK(hipStreamSynchronize(st));
    }
    t1 = now_s();
    const double d_ms = (t1 - t0) / tokens * 1e3;
    const double gb = (double)total_bytes / 1e9;
    printf("D fused launches: %8.2f ms/token  %8.1f tok/s  (%.1f GB/s effective)\n",
           d_ms, 1e3 / d_ms, gb * 1e3 / d_ms);

    hipGraph_t fgraph;
    HIPCHECK(hipStreamBeginCapture(st, hipStreamCaptureModeThreadLocal));
    launch_fused(st);
    HIPCHECK(hipStreamEndCapture(st, &fgraph));
    hipGraphExec_t fexec;
    HIPCHECK(hipGraphInstantiate(&fexec, fgraph, nullptr, nullptr, 0));
    HIPCHECK(hipGraphLaunch(fexec, st));
    HIPCHECK(hipStreamSynchronize(st));
    t0 = now_s();
    for (int t = 0; t < tokens; t++) {
        HIPCHECK(hipGraphLaunch(fexec, st));
        HIPCHECK(hipStreamSynchronize(st));
    }
    t1 = now_s();
    const double e_ms = (t1 - t0) / tokens * 1e3;
    printf("E fused graph   : %8.2f ms/token  %8.1f tok/s  (%.1f GB/s effective)\n",
           e_ms, 1e3 / e_ms, gb * 1e3 / e_ms);
    printf("  fused speedup vs A: %.2fx (D) %.2fx (E)\n", a_ms / d_ms, a_ms / e_ms);

    for (int l = 0; l < layers; l++) HIPCHECK(hipFree(layer_bufs[l]));
    HIPCHECK(hipGraphExecDestroy(fexec));
    HIPCHECK(hipGraphDestroy(fgraph));

    HIPCHECK(hipGraphExecDestroy(exec));
    HIPCHECK(hipGraphDestroy(graph));
    HIPCHECK(hipStreamDestroy(st));
    HIPCHECK(hipFree(base));
    return 0;
}
