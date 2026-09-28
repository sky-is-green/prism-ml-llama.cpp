// Minimal repro for the 125B load hang: large HostToDevice copies from
// file-backed mmap to one or two AMD GPUs, in the same call pattern llama.cpp
// uses in ggml_backend_cuda_buffer_set_tensor:
//   hipSetDevice(dev);
//   hipMemcpyAsync(dst+off, src, size, hipMemcpyHostToDevice, hipStreamPerThread);
//   hipStreamSynchronize(hipStreamPerThread);
// A monitor thread reports the last chunk so a hang identifies the step.
//
// build: hipcc -O2 -pthread -o hsa_upload hsa_upload.cpp
//
// usage: ./hsa_upload [--file PATH] [--devs 1,0] [--chunk-mb 675] [--chunks 48]
//                     [--alloc-mb 17000,19400] [--mode async|sync]
//                     [--src mmap|anon|pinned] [--rounds 1]
#include <hip/hip_runtime.h>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <pthread.h>
#include <string>
#include <vector>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

static double t_elapsed() {
    static struct timespec t0;
    static int init = 0;
    if (!init) { clock_gettime(CLOCK_MONOTONIC, &t0); init = 1; }
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec - t0.tv_sec) + (t.tv_nsec - t0.tv_nsec) / 1e9;
}

#define HIPCHECK(x) do { hipError_t _e = (x); if (_e != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(_e), __FILE__, __LINE__); exit(2); } } while (0)

static std::atomic<int>  g_chunk{-1};
static std::atomic<long> g_bytes{0};
static std::atomic<int>  g_done{0};

static void * monitor(void *) {
    int last = -1;
    while (!g_done.load()) {
        sleep(5);
        int c = g_chunk.load();
        FILE * mi = fopen("/proc/meminfo", "r");
        long avail = -1, cached = -1;
        if (mi) {
            char line[256];
            while (fgets(line, sizeof line, mi)) {
                if (sscanf(line, "MemAvailable: %ld kB", &avail) == 1) continue;
                if (sscanf(line, "Cached: %ld kB", &cached) == 1) continue;
            }
            fclose(mi);
        }
        if (c != last) {
            printf("[monitor] chunk=%d bytes=%.1f GiB avail=%.1f GiB cached=%.1f GiB\n",
                   c, g_bytes.load() / 1073741824.0, avail / 1048576.0, cached / 1048576.0);
            last = c;
        } else if (c >= 0) {
            printf("[monitor] STALL: chunk %d in progress; avail=%.1f GiB cached=%.1f GiB\n",
                   c, avail / 1048576.0, cached / 1048576.0);
        }
        fflush(stdout);
    }
    return nullptr;
}

int main(int argc, char ** argv) {
    std::string file, src_mode = "mmap", mode = "async", devs_s = "1,0", alloc_s = "17000,19400", pattern_s = "";
    size_t chunk_mb = 675, chunks = 48, rounds = 1;
    bool evict = false;
    size_t stage_mb = 0;
    bool stage_pinned = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if      (a == "--file")    file     = next();
        else if (a == "--devs")    devs_s   = next();
        else if (a == "--chunk-mb")chunk_mb = std::stoul(next());
        else if (a == "--chunks")  chunks   = std::stoul(next());
        else if (a == "--alloc-mb")alloc_s  = next();
        else if (a == "--mode")    mode     = next();
        else if (a == "--src")     src_mode = next();
        else if (a == "--rounds")  rounds   = std::stoul(next());
        else if (a == "--pattern") pattern_s = next();
        else if (a == "--evict")   evict    = true;
        else if (a == "--stage-mb")stage_mb = std::stoul(next());
        else if (a == "--stage-pinned") stage_pinned = true;
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }

    int n_dev = 0;
    HIPCHECK(hipGetDeviceCount(&n_dev));
    printf("devices: %d\n", n_dev);
    for (int d = 0; d < n_dev; d++) {
        hipDeviceProp_t p; HIPCHECK(hipGetDeviceProperties(&p, d));
        printf("  dev%d: %s\n", d, p.name);
    }

    // parse device order
    std::vector<int> devs;
    for (size_t pos = 0; pos < devs_s.size(); ) {
        size_t comma = devs_s.find(',', pos);
        devs.push_back(std::stoi(devs_s.substr(pos, comma - pos)));
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }

    // parse per-device allocation
    std::vector<size_t> alloc_mb;
    for (size_t pos = 0; pos < alloc_s.size(); ) {
        size_t comma = alloc_s.find(',', pos);
        alloc_mb.push_back(std::stoull(alloc_s.substr(pos, comma - pos)));
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    while (alloc_mb.size() < devs.size()) alloc_mb.push_back(alloc_mb.back());

    const size_t chunk = chunk_mb * 1024 * 1024;
    const size_t total = chunk * chunks;
    size_t src_len = total;   // set below for mmap sources

    // source buffer
    void * src = nullptr;
    if (src_mode == "mmap" || src_mode == "pinned") {
        if (file.empty()) { fprintf(stderr, "--src %s needs --file\n", src_mode.c_str()); return 1; }
        int fd = open(file.c_str(), O_RDONLY);
        if (fd < 0) { perror("open"); return 1; }
        struct stat st; fstat(fd, &st);
        size_t mlen = (size_t)st.st_size < total ? (size_t)st.st_size : total;
        void * m = mmap(nullptr, mlen, PROT_READ, MAP_PRIVATE, fd, 0);
        if (m == MAP_FAILED) { perror("mmap"); return 1; }
        if (src_mode == "pinned") {
            if (hipHostRegister(m, mlen, hipHostRegisterReadOnly) != hipSuccess) {
                printf("hipHostRegister failed; using mmap unpinned\n");
            } else {
                printf("source: mmap + hipHostRegister(ReadOnly), %zu MiB\n", mlen >> 20);
            }
        } else {
            printf("source: mmap (file-backed, unpinned), %zu MiB\n", mlen >> 20);
        }
        src = m;
        src_len = mlen;
    } else {
        src = malloc(total);
        if (!src) { fprintf(stderr, "malloc failed\n"); return 1; }
        memset(src, 0x5a, total);
        printf("source: anonymous malloc, %zu MiB\n", total >> 20);
    }

    // optional reusable staging buffer: all device copies read from this fixed
    // host address range instead of walking the mmap
    void * stage = nullptr;
    const size_t stage_bytes = stage_mb * 1024 * 1024;
    if (stage_bytes > 0) {
        if (stage_pinned) {
            if (hipHostMalloc(&stage, stage_bytes, 0) != hipSuccess) {
                printf("hipHostMalloc failed; using anon stage\n");
                stage = malloc(stage_bytes);
            } else {
                printf("staging: pinned %zu MiB\n", stage_mb);
            }
        } else {
            stage = malloc(stage_bytes);
            printf("staging: anon %zu MiB\n", stage_mb);
        }
        if (!stage) { fprintf(stderr, "stage alloc failed\n"); return 1; }
    }

    // destination allocations (one per device, like llama.cpp)
    std::vector<void *> dst(devs.size(), nullptr);
    for (size_t d = 0; d < devs.size(); d++) {
        hipSetDevice(devs[d]);
        size_t bytes = alloc_mb[d] * 1024 * 1024;
        HIPCHECK(hipMalloc(&dst[d], bytes));
        printf("allocated dev%d (hip:%d): %zu MiB at %p\n", devs[d], devs[d], bytes >> 20, dst[d]);
        fflush(stdout);
    }

    pthread_t mon; pthread_create(&mon, nullptr, monitor, nullptr);

    printf("copy: %zu chunks x %zu MiB, mode=%s, devices=", chunks, chunk_mb, mode.c_str());
    for (int d : devs) printf("%d ", d);
    printf("\n"); fflush(stdout);

    // pattern sequence: per cycle, which device index (into devs) gets the chunk
    std::vector<size_t> seq;
    if (pattern_s.empty()) {
        for (size_t d = 0; d < devs.size(); d++) seq.push_back(d);   // round robin
    } else {
        for (size_t pos = 0; pos < pattern_s.size(); ) {
            size_t comma = pattern_s.find(',', pos);
            size_t v = std::stoul(pattern_s.substr(pos, comma - pos));
            if (v >= devs.size()) { fprintf(stderr, "pattern entry %zu out of range\n", v); return 1; }
            seq.push_back(v);
            if (comma == std::string::npos) break;
            pos = comma + 1;
        }
    }

    for (size_t r = 0; r < rounds; r++) {
        std::vector<size_t> used(devs.size(), 0);
        for (size_t i = 0; i < chunks; i++) {
            size_t di = seq[i % seq.size()];
            int dev = devs[di];
            size_t dev_bytes = alloc_mb[di] * 1024 * 1024;
            size_t off_in_dev = used[di] * chunk;
            if (off_in_dev + chunk > dev_bytes) {
                fprintf(stderr, "chunk %zu does not fit dev%d (off=%zu + %zu > %zu)\n",
                        i, dev, off_in_dev, chunk, dev_bytes);
                g_done.store(1); pthread_join(mon, nullptr); return 1;
            }
            used[di]++;
            g_chunk.store((int)i);
            printf("chunk %zu/%zu -> dev%d slot=%zu off=%zu MiB t=%.1fs\n", i, chunks, dev, used[di] - 1, off_in_dev >> 20, t_elapsed());
            fflush(stdout);
            hipSetDevice(dev);
            const char * s = (const char *)src + ((i * chunk) % src_len);
            char * dptr = (char *)dst[di] + off_in_dev;
            if (stage && mode == "async") {
                for (size_t o = 0; o < chunk; o += stage_bytes) {
                    size_t piece = stage_bytes < chunk - o ? stage_bytes : chunk - o;
                    memcpy(stage, s + o, piece);
                    HIPCHECK(hipMemcpyAsync(dptr + o, stage, piece, hipMemcpyHostToDevice, hipStreamPerThread));
                    HIPCHECK(hipStreamSynchronize(hipStreamPerThread));
                    g_bytes.fetch_add((long)piece);
                }
            } else if (mode == "async") {
                HIPCHECK(hipMemcpyAsync(dptr, s, chunk, hipMemcpyHostToDevice, hipStreamPerThread));
                HIPCHECK(hipStreamSynchronize(hipStreamPerThread));
                g_bytes.fetch_add((long)chunk);
            } else {
                HIPCHECK(hipMemcpy(dptr, s, chunk, hipMemcpyHostToDevice));
                g_bytes.fetch_add((long)chunk);
            }
            if (evict && src_mode == "mmap") {
                madvise((void *)s, chunk, MADV_DONTNEED);
            }
        }
    }

    g_done.store(1);
    pthread_join(mon, nullptr);
    long mbps = 0;
    printf("ALL DONE: %ld MiB uploaded\n", g_bytes.load() >> 20);
    return 0;
}
