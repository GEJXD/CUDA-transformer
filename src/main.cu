#include "transformer.h"

#include "cuda_utils.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <stdexcept>
#include <vector>

namespace {

struct CliOptions {
    std::string precision = "fp32";
    int batch = 1;
    int src_seq_len = 128;
    int tgt_seq_len = 128;
    int hidden = 768;
    int heads = 12;
    int ffn = 3072;
    int encoder_layers = 6;
    int decoder_layers = 6;
    int iters = 100;
};

double estimate_seq2seq_flops(
    int batch,
    int src_seq_len,
    int tgt_seq_len,
    int hidden,
    int ffn,
    int enc_layers,
    int dec_layers) {
    const double b = static_cast<double>(batch);
    const double s = static_cast<double>(src_seq_len);
    const double t = static_cast<double>(tgt_seq_len);
    const double d = static_cast<double>(hidden);
    const double f = static_cast<double>(ffn);

    const double enc_qkv = 6.0 * b * s * d * d;
    const double enc_attn = 4.0 * b * s * s * d;
    const double enc_proj = 2.0 * b * s * d * d;
    const double enc_ffn = 4.0 * b * s * d * f;
    const double enc_per_layer = enc_qkv + enc_attn + enc_proj + enc_ffn;

    const double dec_self_qkv = 6.0 * b * t * d * d;
    const double dec_self_attn = 4.0 * b * t * t * d;
    const double dec_self_proj = 2.0 * b * t * d * d;
    const double dec_cross_q = 2.0 * b * t * d * d;
    const double dec_cross_kv = 4.0 * b * s * d * d;
    const double dec_cross_attn = 4.0 * b * t * s * d;
    const double dec_cross_proj = 2.0 * b * t * d * d;
    const double dec_ffn = 4.0 * b * t * d * f;
    const double dec_per_layer =
        dec_self_qkv + dec_self_attn + dec_self_proj + dec_cross_q + dec_cross_kv + dec_cross_attn + dec_cross_proj + dec_ffn;

    return enc_per_layer * static_cast<double>(enc_layers) + dec_per_layer * static_cast<double>(dec_layers);
}

void print_usage(const char* prog) {
    printf("Usage:\n");
    printf("  %s --help\n", prog);
    printf("  %s [precision] [batch] [src_seq_len] [tgt_seq_len] [hidden] [heads] [ffn] [encoder_layers] [decoder_layers] [iters]\n", prog);
    printf("  %s --precision [fp32|fp16] --batch [N] --src-seq-len [N] --tgt-seq-len [N] --hidden [N] --heads [N] --ffn [N] --encoder-layers [N] --decoder-layers [N] --iters [N]\n", prog);
    printf("\n");
    printf("precision:\n");
    printf("  fp32    FP32 baseline path\n");
    printf("  fp16    Mixed precision path (FP16 GEMM via Tensor Cores, FP32 accum)\n");
    printf("\n");
    printf("named options:\n");
    printf("  --encoder-layers  Number of encoder layers\n");
    printf("  --decoder-layers  Number of decoder layers\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s fp32 4 128 128 768 12 3072 6 6 20\n", prog);
    printf("  %s fp16 4 256 128 768 12 3072 12 12 20\n", prog);
    printf("  %s --precision fp16 --batch 4 --src-seq-len 256 --tgt-seq-len 128 --hidden 768 --heads 12 --ffn 3072 --encoder-layers 12 --decoder-layers 12 --iters 20\n", prog);
}

int parse_positive_int(const char* name, const char* text) {
    char* end = nullptr;
    const long v = std::strtol(text, &end, 10);
    if (end == text || *end != '\0') {
        throw std::invalid_argument(std::string(name) + " must be an integer, got '" + text + "'");
    }
    if (v <= 0) {
        throw std::invalid_argument(std::string(name) + " must be > 0, got '" + text + "'");
    }
    if (v > 2147483647L) {
        throw std::invalid_argument(std::string(name) + " is too large, got '" + text + "'");
    }
    return static_cast<int>(v);
}

bool is_option_like(const char* text) {
    return text != nullptr && text[0] == '-';
}

void apply_named_option_or_throw(CliOptions* opt, const char* key, const char* value) {
    if (std::strcmp(key, "--precision") == 0) {
        opt->precision = value;
        return;
    }
    if (std::strcmp(key, "--batch") == 0) {
        opt->batch = parse_positive_int("batch", value);
        return;
    }
    if (std::strcmp(key, "--src-seq-len") == 0) {
        opt->src_seq_len = parse_positive_int("src_seq_len", value);
        return;
    }
    if (std::strcmp(key, "--tgt-seq-len") == 0) {
        opt->tgt_seq_len = parse_positive_int("tgt_seq_len", value);
        return;
    }
    if (std::strcmp(key, "--hidden") == 0) {
        opt->hidden = parse_positive_int("hidden", value);
        return;
    }
    if (std::strcmp(key, "--heads") == 0) {
        opt->heads = parse_positive_int("heads", value);
        return;
    }
    if (std::strcmp(key, "--ffn") == 0) {
        opt->ffn = parse_positive_int("ffn", value);
        return;
    }
    if (std::strcmp(key, "--encoder-layers") == 0) {
        opt->encoder_layers = parse_positive_int("encoder_layers", value);
        return;
    }
    if (std::strcmp(key, "--decoder-layers") == 0) {
        opt->decoder_layers = parse_positive_int("decoder_layers", value);
        return;
    }
    if (std::strcmp(key, "--iters") == 0) {
        opt->iters = parse_positive_int("iters", value);
        return;
    }

    throw std::invalid_argument(std::string("unknown option: ") + key);
}

CliOptions parse_cli_or_throw(int argc, char** argv) {
    CliOptions opt;

    if (argc == 1) {
        return opt;
    }

    if (argc == 2 && (std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0)) {
        print_usage(argv[0]);
        std::exit(0);
    }

    if (is_option_like(argv[1])) {
        for (int i = 1; i < argc; ++i) {
            const char* key = argv[i];
            if (std::strcmp(key, "--help") == 0 || std::strcmp(key, "-h") == 0) {
                print_usage(argv[0]);
                std::exit(0);
            }
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string("missing value for option: ") + key);
            }
            const char* value = argv[i + 1];
            if (is_option_like(value)) {
                throw std::invalid_argument(std::string("missing value for option: ") + key);
            }
            apply_named_option_or_throw(&opt, key, value);
            ++i;
        }

        if (opt.precision != "fp32" && opt.precision != "fp16") {
            throw std::invalid_argument("precision must be one of: fp32, fp16");
        }
        if (opt.hidden % opt.heads != 0) {
            throw std::invalid_argument("hidden must be divisible by heads");
        }

        return opt;
    }

    if (argc != 11) {
        throw std::invalid_argument(
            "invalid argument count: expected 10 positional args after program name, or --help");
    }

    opt.precision = argv[1];
    opt.batch = parse_positive_int("batch", argv[2]);
    opt.src_seq_len = parse_positive_int("src_seq_len", argv[3]);
    opt.tgt_seq_len = parse_positive_int("tgt_seq_len", argv[4]);
    opt.hidden = parse_positive_int("hidden", argv[5]);
    opt.heads = parse_positive_int("heads", argv[6]);
    opt.ffn = parse_positive_int("ffn", argv[7]);
    opt.encoder_layers = parse_positive_int("encoder_layers", argv[8]);
    opt.decoder_layers = parse_positive_int("decoder_layers", argv[9]);
    opt.iters = parse_positive_int("iters", argv[10]);

    if (opt.precision != "fp32" && opt.precision != "fp16") {
        throw std::invalid_argument("precision must be one of: fp32, fp16");
    }
    if (opt.hidden % opt.heads != 0) {
        throw std::invalid_argument("hidden must be divisible by heads");
    }

    return opt;
}

}  // namespace

int main(int argc, char** argv) {
    float* d_encoder_input = nullptr;
    float* d_decoder_input = nullptr;
    float* d_output = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    try {
        const CliOptions opt = parse_cli_or_throw(argc, argv);
        const Precision precision = (opt.precision == "fp16") ? Precision::FP16 : Precision::FP32;

        const size_t enc_elem_count = static_cast<size_t>(opt.batch) * opt.src_seq_len * opt.hidden;
        const size_t dec_elem_count = static_cast<size_t>(opt.batch) * opt.tgt_seq_len * opt.hidden;
        const size_t enc_bytes = enc_elem_count * sizeof(float);
        const size_t dec_bytes = dec_elem_count * sizeof(float);

        std::vector<float> h_encoder_input(enc_elem_count, 0.01f);
        std::vector<float> h_decoder_input(dec_elem_count, 0.01f);
        std::vector<float> h_output(dec_elem_count, 0.0f);

        CUDA_CHECK(cudaMalloc(&d_encoder_input, enc_bytes));
        CUDA_CHECK(cudaMalloc(&d_decoder_input, dec_bytes));
        CUDA_CHECK(cudaMalloc(&d_output, dec_bytes));
        CUDA_CHECK(cudaMemcpy(d_encoder_input, h_encoder_input.data(), enc_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_decoder_input, h_decoder_input.data(), dec_bytes, cudaMemcpyHostToDevice));

        const int warmup = 10;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        TransformerConfig cfg{
            precision,
            opt.batch,
            opt.src_seq_len,
            opt.tgt_seq_len,
            opt.hidden,
            opt.heads,
            opt.ffn,
            opt.encoder_layers,
            opt.decoder_layers,
            1e-5f,
        };
        Transformer model(cfg);

        for (int i = 0; i < warmup; ++i) {
            model.forward(d_encoder_input, d_decoder_input, d_output);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < opt.iters; ++i) {
            model.forward(d_encoder_input, d_decoder_input, d_output);
        }
        CUDA_CHECK(cudaEventRecord(stop));

        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, dec_bytes, cudaMemcpyDeviceToHost));

        printf("Full encoder-decoder Transformer forward benchmark\n");
        printf("precision=%s batch=%d src_seq=%d tgt_seq=%d hidden=%d heads=%d ffn=%d encoder_layers=%d decoder_layers=%d iters=%d\n",
               opt.precision.c_str(),
               opt.batch,
               opt.src_seq_len,
               opt.tgt_seq_len,
               opt.hidden,
               opt.heads,
               opt.ffn,
               opt.encoder_layers,
               opt.decoder_layers,
               opt.iters);
        printf("avg latency: %.3f ms\n", ms / static_cast<float>(opt.iters));
        printf("throughput: %.2f tokens/s\n", (opt.batch * opt.tgt_seq_len * 1000.0f * opt.iters) / ms);
        const int layer_count = opt.encoder_layers + opt.decoder_layers;
        const double flops_per_iter = estimate_seq2seq_flops(
            opt.batch,
            opt.src_seq_len,
            opt.tgt_seq_len,
            opt.hidden,
            opt.ffn,
            opt.encoder_layers,
            opt.decoder_layers);
        const double avg_ms = static_cast<double>(ms) / static_cast<double>(opt.iters);
        const double avg_sec = avg_ms * 1e-3;
        const double achieved_tflops = flops_per_iter / avg_sec / 1e12;
        const double layer_avg_ms = avg_ms / static_cast<double>(layer_count);
        printf("theoretical FLOPs/iter: %.3e\n", flops_per_iter);
        printf("achieved throughput: %.3f TFLOPS\n", achieved_tflops);
        printf("per-layer avg latency: %.3f ms\n", layer_avg_ms);
        printf("output checksum sample: %.6f %.6f %.6f\n", h_output[0], h_output[1], h_output[2]);

        if (start != nullptr) {
            CUDA_CHECK(cudaEventDestroy(start));
            start = nullptr;
        }
        if (stop != nullptr) {
            CUDA_CHECK(cudaEventDestroy(stop));
            stop = nullptr;
        }
        if (d_encoder_input != nullptr) {
            CUDA_CHECK(cudaFree(d_encoder_input));
            d_encoder_input = nullptr;
        }
        if (d_decoder_input != nullptr) {
            CUDA_CHECK(cudaFree(d_decoder_input));
            d_decoder_input = nullptr;
        }
        if (d_output != nullptr) {
            CUDA_CHECK(cudaFree(d_output));
            d_output = nullptr;
        }

        return 0;
    } catch (const std::exception& e) {
        if (start != nullptr) {
            cudaEventDestroy(start);
        }
        if (stop != nullptr) {
            cudaEventDestroy(stop);
        }
        if (d_encoder_input != nullptr) {
            cudaFree(d_encoder_input);
        }
        if (d_decoder_input != nullptr) {
            cudaFree(d_decoder_input);
        }
        if (d_output != nullptr) {
            cudaFree(d_output);
        }

        fprintf(stderr, "error: %s\n\n", e.what());
        fprintf(stderr, "Use --help to see valid arguments.\n");
        print_usage(argv[0]);
        return 1;
    }
}
