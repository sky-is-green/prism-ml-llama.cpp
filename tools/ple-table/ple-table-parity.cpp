// Scion P5: bit-exact parity for the 32-element PLE table containers (Q3X32,
// Q2X32) and the Q4_0 table they replace.
//
// Probe rows are decoded through the real ggml get_rows op on the chosen
// backend (CPU by default, a GPU device with --device) and compared
// bit-for-bit with an independent scalar decoder over the raw GGUF bytes.
//
// usage:
//   ple-table-parity <gguf> [--device NAME] [--rows N] [--table NAME]
//
// exit 0 when every probe row is bit-exact.

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

static const char * DEFAULT_TABLE = "per_layer_token_embd.weight";

// independent decoder of the packed bytes; mirrors the PLE-TABLE-TYPES spec
static void ref_decode_row(enum ggml_type type, const uint8_t * row, float * out, int64_t n) {
    switch (type) {
        case GGML_TYPE_Q4_0:
            for (int64_t b = 0; b < n/32; b++) {
                const uint8_t * blk = row + b*18;
                const float d = ggml_fp16_to_fp32(*(const ggml_fp16_t *) blk);
                for (int j = 0; j < 16; j++) {
                    out[b*32 + j]      = ((blk[2 + j] & 0x0F) - 8) * d;
                    out[b*32 + j + 16] = ((blk[2 + j] >>   4) - 8) * d;
                }
            }
            break;
        case GGML_TYPE_Q3X32:
            for (int64_t b = 0; b < n/32; b++) {
                const uint8_t * blk = row + b*14;
                const float d = ggml_fp16_to_fp32(*(const ggml_fp16_t *) blk);
                const uint8_t * qs = blk + 2;
                const uint8_t * qh = blk + 10;
                for (int j = 0; j < 32; j++) {
                    const int c = ((qs[j/4] >> ((j % 4)*2)) & 3) | (((qh[j/8] >> (j % 8)) & 1) << 2);
                    out[b*32 + j] = (2*c - 7) * d;
                }
            }
            break;
        case GGML_TYPE_Q2X32:
            for (int64_t b = 0; b < n/32; b++) {
                const uint8_t * blk = row + b*10;
                const float d = ggml_fp16_to_fp32(*(const ggml_fp16_t *) blk);
                const uint8_t * qs = blk + 2;
                for (int j = 0; j < 32; j++) {
                    const int c = (qs[j/4] >> ((j % 4)*2)) & 3;
                    out[b*32 + j] = (2*c - 3) * d;
                }
            }
            break;
        default:
            std::fprintf(stderr, "no reference decoder for type %d\n", (int) type);
            std::abort();
    }
}

static size_t expected_row_bytes(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0: return 90;
        case GGML_TYPE_Q3X32: return 70;
        case GGML_TYPE_Q2X32: return 50;
        default: return 0;
    }
}

static uint64_t lcg_next(uint64_t & s) {
    s = s*6364136223846793005ull + 1442695040888963407ull;
    return s >> 33;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <gguf> [--device NAME] [--rows N] [--table NAME]\n", argv[0]);
        return 2;
    }

    std::string path = argv[1];
    std::string device_name;
    std::string table_name = DEFAULT_TABLE;
    std::string dump;
    int n_random = 32;

    for (int i = 2; i < argc; ++i) {
        const std::string f = argv[i];
        if (f == "--device" && i + 1 < argc) {
            device_name = argv[++i];
        } else if (f == "--rows" && i + 1 < argc) {
            n_random = std::max(0, std::atoi(argv[++i]));
        } else if (f == "--table" && i + 1 < argc) {
            table_name = argv[++i];
        } else if (f == "--dump" && i + 1 < argc) {
            dump = argv[++i];
        } else {
            std::fprintf(stderr, "unknown argument %s\n", f.c_str());
            return 2;
        }
    }

#if defined(_WIN32)
    HANDLE hfile = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hfile == INVALID_HANDLE_VALUE) { std::fprintf(stderr, "open failed: %s\n", path.c_str()); return 1; }
    LARGE_INTEGER fsize_li;
    GetFileSizeEx(hfile, &fsize_li);
    const size_t fsize = (size_t) fsize_li.QuadPart;
    HANDLE hmap = CreateFileMappingA(hfile, NULL, PAGE_READONLY, 0, 0, NULL);
    const uint8_t * base = (const uint8_t *) MapViewOfFile(hmap, FILE_MAP_READ, 0, 0, 0);
    if (!base) { std::fprintf(stderr, "mmap failed\n"); return 1; }
#else
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) { std::fprintf(stderr, "open failed: %s\n", path.c_str()); return 1; }
    struct stat st;
    if (fstat(fd, &st) != 0) { std::fprintf(stderr, "stat failed\n"); return 1; }
    const size_t fsize = (size_t) st.st_size;
    const uint8_t * base = (const uint8_t *) mmap(NULL, fsize, PROT_READ, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { std::fprintf(stderr, "mmap failed\n"); return 1; }
    // row probes only: never fault in the whole table
    posix_madvise((void *) base, fsize, POSIX_MADV_RANDOM);
#endif

    struct gguf_init_params params = { /*.no_alloc =*/ true, /*.ctx =*/ NULL };
    struct gguf_context * gctx = gguf_init_from_file(path.c_str(), params);
    if (!gctx) { std::fprintf(stderr, "not a gguf: %s\n", path.c_str()); return 1; }

    const int64_t tid = gguf_find_tensor(gctx, table_name.c_str());
    if (tid < 0) { std::fprintf(stderr, "tensor %s not found\n", table_name.c_str()); return 1; }
    const enum ggml_type type = gguf_get_tensor_type(gctx, tid);
    const int64_t n_rows  = gguf_get_tensor_size(gctx, tid) / ggml_row_size(type, 160);
    const size_t row_bytes = ggml_row_size(type, 160);
    if (row_bytes != expected_row_bytes(type) || gguf_get_tensor_size(gctx, tid) % row_bytes != 0) {
        std::fprintf(stderr, "unexpected geometry for %s: type %d row %zu\n", table_name.c_str(), (int) type, row_bytes);
        return 1;
    }
    const uint8_t * data = base + gguf_get_data_offset(gctx) + gguf_get_tensor_offset(gctx, tid);
    const size_t data_size = (size_t) n_rows * row_bytes;
    if (gguf_get_data_offset(gctx) + gguf_get_tensor_offset(gctx, tid) + data_size > fsize) {
        std::fprintf(stderr, "tensor data out of file bounds\n");
        return 1;
    }

    // probe rows: fixed landmarks + deterministic random spread
    std::vector<int32_t> rows;
    for (const int64_t r : { (int64_t) 0, (int64_t) 1, (int64_t) 12345, (int64_t) 20000003, n_rows - 1 }) {
        const int32_t v = (int32_t) std::min<int64_t>(r, n_rows - 1);
        if (std::find(rows.begin(), rows.end(), v) == rows.end()) rows.push_back(v);
    }
    uint64_t seed = 1234;
    for (int i = 0; i < n_random; ++i) {
        rows.push_back((int32_t) (lcg_next(seed) % (uint64_t) n_rows));
    }

    ggml_backend_load_all();
    ggml_backend_dev_t dev = device_name.empty()
        ? ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)
        : ggml_backend_dev_by_name(device_name.c_str());
    if (!dev) {
        std::fprintf(stderr, "device not found: %s\n", device_name.empty() ? "CPU" : device_name.c_str());
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            std::fprintf(stderr, "  %s (%s)\n", ggml_backend_dev_name(d), ggml_backend_dev_description(d));
        }
        return 1;
    }
    struct ggml_backend_dev_props props;
    ggml_backend_dev_get_props(dev, &props);
    const bool zero_copy = props.caps.buffer_from_host_ptr;
    std::printf("device %s (%s)%s, type %s, %" PRId64 " rows x 160\n",
                ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
                zero_copy ? " host-ptr" : " copy", ggml_type_name(type), n_rows);

    struct ggml_init_params iparams = {
        /*.mem_size   =*/ (size_t) 4*1024*1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(iparams);

    struct ggml_tensor * table = ggml_new_tensor_2d(ctx, type, 160, n_rows);
    struct ggml_tensor * idx   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) rows.size());
    struct ggml_tensor * dst   = ggml_get_rows(ctx, table, idx);

    ggml_backend_t backend = ggml_backend_dev_init(dev, NULL);
    if (!backend) { std::fprintf(stderr, "backend init failed\n"); return 1; }

    ggml_backend_buffer_t table_buf = NULL;
    if (zero_copy) {
        table_buf = ggml_backend_dev_buffer_from_host_ptr(dev, (void *) data, data_size, data_size);
        if (!table_buf) { std::fprintf(stderr, "host-ptr buffer failed\n"); return 1; }
        ggml_backend_tensor_alloc(table_buf, table, (void *) data);
    }
    ggml_backend_buffer_t ctx_buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!zero_copy) {
        if (!ctx_buf) { std::fprintf(stderr, "backend buffer alloc failed (%.1f GB?)\n", data_size/1e9); return 1; }
        ggml_backend_tensor_set(table, data, 0, data_size);
    }
    ggml_backend_tensor_set(idx, rows.data(), 0, rows.size()*sizeof(int32_t));

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, dst);
    const enum ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) { std::fprintf(stderr, "compute failed: %d\n", (int) status); return 1; }

    std::vector<float> got(rows.size()*160);
    ggml_backend_tensor_get(dst, got.data(), 0, got.size()*sizeof(float));

    int64_t ndiff = 0;
    double max_abs = 0.0;
    for (size_t p = 0; p < rows.size(); ++p) {
        float want[160];
        ref_decode_row(type, data + (size_t) rows[p]*row_bytes, want, 160);
        for (int j = 0; j < 160; ++j) {
            const float g = got[p*160 + j];
            if (memcmp(&g, &want[j], sizeof(float)) != 0) {
                ndiff++;
                max_abs = std::max(max_abs, (double) std::fabs(g - want[j]));
            }
        }
    }

    std::printf("%s PLE: %zu probes, %" PRId64 " of %zu values differ, max_abs %.3e %s\n",
                ggml_type_name(type), rows.size(), ndiff, rows.size()*160, max_abs,
                ndiff == 0 ? "PASS" : "FAIL");

    if (!dump.empty()) {
        // fixture for the Strata reader test: magic, version, type, n, dim,
        // i32 rows, f32 values (all little-endian)
        FILE * df = std::fopen(dump.c_str(), "wb");
        if (!df) { std::fprintf(stderr, "cannot write %s\n", dump.c_str()); return 1; }
        const uint32_t version = 1;
        const uint32_t t32 = (uint32_t) type;
        const uint32_t np  = (uint32_t) rows.size();
        const uint32_t dim = 160;
        std::fwrite("PLEF", 1, 4, df);
        std::fwrite(&version, 4, 1, df);
        std::fwrite(&t32, 4, 1, df);
        std::fwrite(&np, 4, 1, df);
        std::fwrite(&dim, 4, 1, df);
        std::fwrite(rows.data(), 4, rows.size(), df);
        std::fwrite(got.data(), 4, got.size(), df);
        std::fclose(df);
        std::printf("wrote fixture %s (%zu rows)\n", dump.c_str(), rows.size());
    }

    if (ctx_buf)   ggml_backend_buffer_free(ctx_buf);
    if (table_buf) ggml_backend_buffer_free(table_buf);
    ggml_backend_free(backend);
    ggml_free(ctx);
    gguf_free(gctx);
#if defined(_WIN32)
    UnmapViewOfFile(base);
    CloseHandle(hmap);
    CloseHandle(hfile);
#else
    munmap((void *) base, fsize);
    close(fd);
#endif
    return ndiff == 0 ? 0 : 1;
}
