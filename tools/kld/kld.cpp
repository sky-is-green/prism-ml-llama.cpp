// Scion P5 shrink: full-vocabulary KLD of a served GGUF against a parked
// teacher log-probability park (the half the PyTorch 48-layer gate produced).
//
// The teacher park is one fp32 file per window, row-major
// [seq-1, n_vocab] log-softmax rows (the same rows qwen4exp_eval teachers
// park).  Student rows are the logits of the same token positions; the tool
// reports the same tail statistics as moe/kld_eval.py so a shrink step can be
// read against the release's 0.5800.
//
// usage:
//   llama-kld -m model.gguf --tokens tokens.bin --teacher-dir DIR
//             [--windows 8] [--seq 512] [-ngl 99] [-t 8] [--out kld.json]
//   llama-kld -m model.gguf --tokens tokens.bin --dump-teacher DIR  (park it)

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

struct kld_args {
    std::string model;
    std::string teacher_dir;
    std::string dump_teacher;
    std::string tokens;
    std::string out = "kld.json";
    int ngl = 99;
    int n_threads = 8;
    int windows = 8;
    int seq = 512;
    bool flash_attn = true;
};

static void usage(const char * prog) {
    fprintf(stderr,
        "usage: %s -m model.gguf --tokens tokens.bin (--teacher-dir DIR | --dump-teacher DIR)\n"
        "       [--windows 8] [--seq 512] [-ngl 99] [-t 8] [--out kld.json] [--no-fa]\n",
        prog);
}

static std::string need_value(int & i, int argc, char ** argv, const char * flag) {
    if (i + 1 >= argc) {
        throw std::runtime_error(std::string(flag) + " needs a value");
    }
    return argv[++i];
}

static kld_args parse_args(int argc, char ** argv) {
    kld_args a;
    for (int i = 1; i < argc; ++i) {
        const std::string f = argv[i];
        if (f == "-m" || f == "--model") {
            a.model = need_value(i, argc, argv, f.c_str());
        } else if (f == "--teacher-dir") {
            a.teacher_dir = need_value(i, argc, argv, f.c_str());
        } else if (f == "--dump-teacher") {
            a.dump_teacher = need_value(i, argc, argv, f.c_str());
        } else if (f == "--tokens") {
            a.tokens = need_value(i, argc, argv, f.c_str());
        } else if (f == "--out") {
            a.out = need_value(i, argc, argv, f.c_str());
        } else if (f == "--windows") {
            a.windows = std::stoi(need_value(i, argc, argv, f.c_str()));
        } else if (f == "--seq") {
            a.seq = std::stoi(need_value(i, argc, argv, f.c_str()));
        } else if (f == "-ngl" || f == "--gpu-layers") {
            a.ngl = std::stoi(need_value(i, argc, argv, f.c_str()));
        } else if (f == "-t" || f == "--threads") {
            a.n_threads = std::stoi(need_value(i, argc, argv, f.c_str()));
        } else if (f == "--no-fa") {
            a.flash_attn = false;
        } else if (f == "-h" || f == "--help") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + f);
        }
    }
    if (a.model.empty())          { throw std::runtime_error("-m is required"); }
    if (a.tokens.empty())         { throw std::runtime_error("--tokens is required"); }
    if (a.teacher_dir.empty() && a.dump_teacher.empty()) {
        throw std::runtime_error("pass --teacher-dir and/or --dump-teacher");
    }
    if (a.seq < 2 || a.windows < 1) { throw std::runtime_error("bad --seq/--windows"); }
    return a;
}

static std::vector<uint8_t> read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        throw std::runtime_error("cannot open " + path);
    }
    const std::streamsize size = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> buf(size);
    if (size > 0 && !f.read((char *) buf.data(), size)) {
        throw std::runtime_error("short read on " + path);
    }
    return buf;
}

static std::string window_path(const std::string & dir, int w) {
    char name[64];
    snprintf(name, sizeof(name), "/w%d.bin", w);
    return dir + name;
}

static int run(int argc, char ** argv) {
    const kld_args a = parse_args(argc, argv);

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = a.ngl;
    llama_model * model = llama_model_load_from_file(a.model.c_str(), mparams);
    if (model == nullptr) {
        throw std::runtime_error("failed to load model " + a.model);
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int n_rows  = a.seq - 1;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx            = a.seq;
    cparams.n_batch          = a.seq;
    cparams.n_ubatch         = a.seq;
    cparams.n_threads        = a.n_threads;
    cparams.n_threads_batch  = a.n_threads;
    cparams.flash_attn_type  = a.flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                                            : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        throw std::runtime_error("failed to create context");
    }

    const auto tok_bytes = read_file(a.tokens);
    if (tok_bytes.size() != (size_t) a.windows * a.seq * sizeof(int32_t)) {
        throw std::runtime_error("tokens file size mismatch (expected windows*seq*4)");
    }
    const int32_t * tokens = (const int32_t *) tok_bytes.data();

    if (!a.dump_teacher.empty()) {
        std::filesystem::create_directories(a.dump_teacher);
    }
    if (!a.teacher_dir.empty()) {
        std::error_code ec;
        if (!std::filesystem::exists(a.teacher_dir, ec)) {
            throw std::runtime_error("teacher dir does not exist: " + a.teacher_dir);
        }
    }

    std::vector<float> trow(n_vocab);
    std::vector<float> srow(n_vocab);

    std::vector<double> klds;
    klds.reserve((size_t) a.windows * n_rows);
    double sum_t_ent = 0.0, sum_s_ent = 0.0, sum_t_pk = 0.0, sum_s_pk = 0.0;
    int64_t n_same = 0;

    for (int w = 0; w < a.windows; ++w) {
        fprintf(stderr, "window %d/%d\n", w + 1, a.windows);
        llama_memory_clear(llama_get_memory(ctx), true);

        llama_batch batch = llama_batch_init(a.seq, 0, 1);
        for (int i = 0; i < a.seq; ++i) {
            batch.token[i]     = tokens[(size_t) w * a.seq + i];
            batch.pos[i]       = i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = true;
        }
        batch.n_tokens = a.seq;
        const int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            throw std::runtime_error("llama_decode failed, rc=" + std::to_string(rc));
        }

        std::ofstream dump;
        if (!a.dump_teacher.empty()) {
            dump.open(window_path(a.dump_teacher, w), std::ios::binary);
            if (!dump) {
                throw std::runtime_error("cannot write " + window_path(a.dump_teacher, w));
            }
        }
        std::ifstream teacher;
        if (!a.teacher_dir.empty()) {
            teacher.open(window_path(a.teacher_dir, w), std::ios::binary);
            if (!teacher) {
                throw std::runtime_error("cannot open " + window_path(a.teacher_dir, w));
            }
        }

        for (int i = 0; i < n_rows; ++i) {
            const float * slogits = llama_get_logits_ith(ctx, i);
            if (slogits == nullptr) {
                throw std::runtime_error("no logits for row " + std::to_string(i));
            }

            float mx = -std::numeric_limits<float>::infinity();
            for (int v = 0; v < n_vocab; ++v) {
                mx = std::max(mx, slogits[v]);
            }
            double sum = 0.0;
            for (int v = 0; v < n_vocab; ++v) {
                sum += std::exp((double) (slogits[v] - mx));
            }
            const double lse = mx + std::log(sum);
            for (int v = 0; v < n_vocab; ++v) {
                srow[v] = (float) ((double) slogits[v] - lse);
            }

            if (dump) {
                dump.write((const char *) srow.data(), (size_t) n_vocab * sizeof(float));
                if (a.teacher_dir.empty()) {
                    continue;
                }
            }

            if (!teacher.read((char *) trow.data(), (size_t) n_vocab * sizeof(float))) {
                throw std::runtime_error("short read on " + window_path(a.teacher_dir, w));
            }

            double kld = 0.0, t_ent = 0.0, s_ent = 0.0;
            float t_best = -std::numeric_limits<float>::infinity();
            float s_best = -std::numeric_limits<float>::infinity();
            int t_arg = 0, s_arg = 0;
            for (int v = 0; v < n_vocab; ++v) {
                const double tp = std::exp((double) trow[v]);
                const double st = (double) srow[v];
                kld   += tp * ((double) trow[v] - st);
                t_ent -= tp * (double) trow[v];
                s_ent -= std::exp(st) * st;
                if (trow[v] > t_best) { t_best = trow[v]; t_arg = v; }
                if (srow[v] > s_best) { s_best = srow[v]; s_arg = v; }
            }
            klds.push_back(kld);
            sum_t_ent += t_ent;
            sum_s_ent += s_ent;
            sum_t_pk  += std::exp((double) t_best);
            sum_s_pk  += std::exp((double) s_best);
            n_same    += (t_arg == s_arg);
        }
    }

    if (!a.dump_teacher.empty() && a.teacher_dir.empty()) {
        fprintf(stderr, "dumped %d windows to %s\n", a.windows, a.dump_teacher.c_str());
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    }

    const int64_t n_tokens = (int64_t) klds.size();
    double mean = 0.0, maxv = klds[0];
    int argmax = 0;
    for (int64_t i = 0; i < n_tokens; ++i) {
        mean += klds[i];
        if (klds[i] > maxv) { maxv = klds[i]; argmax = (int) i; }
    }
    mean /= n_tokens;

    std::vector<double> sorted(klds);
    std::sort(sorted.begin(), sorted.end());
    auto pct = [&](double p) {
        const int64_t idx = std::min<int64_t>((int64_t) std::nearbyint(p / 100.0 * (n_tokens - 1)), n_tokens - 1);
        return sorted[idx];
    };
    const int64_t p999_idx = std::min<int64_t>((int64_t) std::nearbyint(0.999 * (n_tokens - 1)), n_tokens - 1);
    const int64_t p999_rank = n_tokens - p999_idx;

    std::vector<int> order(n_tokens);
    for (int64_t i = 0; i < n_tokens; ++i) { order[i] = (int) i; }
    std::partial_sort(order.begin(), order.begin() + std::min<int64_t>(16, n_tokens), order.end(),
                      [&](int x, int y) { return klds[x] > klds[y]; });

    std::ofstream out(a.out);
    if (!out) {
        throw std::runtime_error("cannot write " + a.out);
    }
    char num[64];
    auto fnum = [&](double v) {
        snprintf(num, sizeof(num), "%.9g", v);
        return std::string(num);
    };
    out << "{\n";
    out << "  \"format\": \"scion-kld-gguf\",\n";
    out << "  \"n_tokens\": " << n_tokens << ",\n";
    out << "  \"mean\": " << fnum(mean) << ",\n";
    out << "  \"max\": " << fnum(maxv) << ",\n";
    out << "  \"argmax_token\": " << argmax << ",\n";
    out << "  \"p50\": " << fnum(pct(50.0)) << ",\n";
    out << "  \"p90\": " << fnum(pct(90.0)) << ",\n";
    out << "  \"p99\": " << fnum(pct(99.0)) << ",\n";
    out << "  \"p99.9\": " << fnum(pct(99.9)) << ",\n";
    out << "  \"p999_rank\": " << p999_rank << ",\n";
    out << "  \"p999_resolved\": " << (p999_rank >= 100 ? "true" : "false") << ",\n";
    out << "  \"worst_tokens\": [\n";
    for (int64_t k = 0; k < std::min<int64_t>(16, n_tokens); ++k) {
        const int t = order[k];
        out << "    {\"token\": " << t << ", \"window\": " << t / n_rows
            << ", \"pos\": " << t % n_rows << ", \"kld\": " << fnum(klds[t]) << "}"
            << (k + 1 < std::min<int64_t>(16, n_tokens) ? "," : "") << "\n";
    }
    out << "  ],\n";
    out << "  \"teacher_entropy_nats\": " << fnum(sum_t_ent / n_tokens) << ",\n";
    out << "  \"teacher_top1_prob\": " << fnum(sum_t_pk / n_tokens) << ",\n";
    out << "  \"student_entropy_nats\": " << fnum(sum_s_ent / n_tokens) << ",\n";
    out << "  \"student_top1_prob\": " << fnum(sum_s_pk / n_tokens) << ",\n";
    out << "  \"sharper_than_teacher\": " << ((sum_s_ent / n_tokens) < (sum_t_ent / n_tokens) - 0.5 ? "true" : "false") << ",\n";
    out << "  \"top1_agreement\": " << fnum((double) n_same / n_tokens) << "\n";
    out << "}\n";
    out.close();

    printf("KLD: %lld tokens mean %.6f p50 %.4f p90 %.4f p99 %.4f p99.9 %.4f max %.4f top1 %.4f\n",
           (long long) n_tokens, mean, pct(50.0), pct(90.0), pct(99.0), pct(99.9), maxv,
           (double) n_same / n_tokens);
    printf("wrote %s\n", a.out.c_str());

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

int main(int argc, char ** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        usage(argv[0]);
        return 1;
    }
}
