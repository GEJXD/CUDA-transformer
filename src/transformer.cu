#include "transformer.h"

#include "cuda_ops.h"
#include "gemm.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

#define CUDA_CHECK(expr)                                                                 \
    do {                                                                                 \
        cudaError_t err__ = (expr);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            throw std::runtime_error(cudaGetErrorString(err__));                         \
        }                                                                                \
    } while (0)

void alloc_device(float** ptr, size_t bytes) {
    CUDA_CHECK(cudaMalloc(ptr, bytes));
}

void free_device(float* ptr) {
    if (ptr != nullptr) {
        cudaFree(ptr);
    }
}

void init_param(float* d_ptr, size_t count, float low, float high) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(low, high);

    std::vector<float> host(count);
    for (size_t i = 0; i < count; ++i) {
        host[i] = dist(rng);
    }
    CUDA_CHECK(cudaMemcpy(d_ptr, host.data(), count * sizeof(float), cudaMemcpyHostToDevice));
}

void init_constant(float* d_ptr, size_t count, float value) {
    std::vector<float> host(count, value);
    CUDA_CHECK(cudaMemcpy(d_ptr, host.data(), count * sizeof(float), cudaMemcpyHostToDevice));
}

}  // namespace

TransformerBlock::TransformerBlock(const TransformerConfig& cfg) : cfg_(cfg) {
    if (cfg_.hidden_size % cfg_.num_heads != 0) {
        throw std::invalid_argument("hidden_size must be divisible by num_heads");
    }
    head_dim_ = cfg_.hidden_size / cfg_.num_heads;

    init_weights();
    allocate_workspace();
}

TransformerBlock::~TransformerBlock() {
    free_workspace();
    free_weights();
}

void TransformerBlock::init_weights() {
    const int d = cfg_.hidden_size;
    const int ff = cfg_.ffn_size;

    alloc_device(&w_q_, static_cast<size_t>(d) * d * sizeof(float));
    alloc_device(&w_k_, static_cast<size_t>(d) * d * sizeof(float));
    alloc_device(&w_v_, static_cast<size_t>(d) * d * sizeof(float));
    alloc_device(&w_o_, static_cast<size_t>(d) * d * sizeof(float));

    alloc_device(&b_q_, static_cast<size_t>(d) * sizeof(float));
    alloc_device(&b_k_, static_cast<size_t>(d) * sizeof(float));
    alloc_device(&b_v_, static_cast<size_t>(d) * sizeof(float));
    alloc_device(&b_o_, static_cast<size_t>(d) * sizeof(float));

    alloc_device(&ln1_gamma_, static_cast<size_t>(d) * sizeof(float));
    alloc_device(&ln1_beta_, static_cast<size_t>(d) * sizeof(float));
    alloc_device(&ln2_gamma_, static_cast<size_t>(d) * sizeof(float));
    alloc_device(&ln2_beta_, static_cast<size_t>(d) * sizeof(float));

    alloc_device(&w1_, static_cast<size_t>(d) * ff * sizeof(float));
    alloc_device(&b1_, static_cast<size_t>(ff) * sizeof(float));
    alloc_device(&w2_, static_cast<size_t>(ff) * d * sizeof(float));
    alloc_device(&b2_, static_cast<size_t>(d) * sizeof(float));

    init_param(w_q_, static_cast<size_t>(d) * d, -0.02f, 0.02f);
    init_param(w_k_, static_cast<size_t>(d) * d, -0.02f, 0.02f);
    init_param(w_v_, static_cast<size_t>(d) * d, -0.02f, 0.02f);
    init_param(w_o_, static_cast<size_t>(d) * d, -0.02f, 0.02f);

    init_constant(b_q_, d, 0.0f);
    init_constant(b_k_, d, 0.0f);
    init_constant(b_v_, d, 0.0f);
    init_constant(b_o_, d, 0.0f);

    init_constant(ln1_gamma_, d, 1.0f);
    init_constant(ln1_beta_, d, 0.0f);
    init_constant(ln2_gamma_, d, 1.0f);
    init_constant(ln2_beta_, d, 0.0f);

    init_param(w1_, static_cast<size_t>(d) * ff, -0.02f, 0.02f);
    init_constant(b1_, ff, 0.0f);
    init_param(w2_, static_cast<size_t>(ff) * d, -0.02f, 0.02f);
    init_constant(b2_, d, 0.0f);
}

void TransformerBlock::allocate_workspace() {
    const int s = cfg_.seq_len;
    const int d = cfg_.hidden_size;
    const int h = cfg_.num_heads;
    const int ff = cfg_.ffn_size;

    alloc_device(&x_norm_, static_cast<size_t>(s) * d * sizeof(float));
    alloc_device(&q_, static_cast<size_t>(s) * d * sizeof(float));
    alloc_device(&k_, static_cast<size_t>(s) * d * sizeof(float));
    alloc_device(&v_, static_cast<size_t>(s) * d * sizeof(float));

    alloc_device(&q_heads_, static_cast<size_t>(h) * s * head_dim_ * sizeof(float));
    alloc_device(&k_heads_, static_cast<size_t>(h) * s * head_dim_ * sizeof(float));
    alloc_device(&v_heads_, static_cast<size_t>(h) * s * head_dim_ * sizeof(float));
    alloc_device(&k_heads_t_, static_cast<size_t>(h) * head_dim_ * s * sizeof(float));
    alloc_device(&scores_, static_cast<size_t>(h) * s * s * sizeof(float));
    alloc_device(&ctx_heads_, static_cast<size_t>(h) * s * head_dim_ * sizeof(float));

    alloc_device(&ctx_, static_cast<size_t>(s) * d * sizeof(float));
    alloc_device(&attn_out_, static_cast<size_t>(s) * d * sizeof(float));
    alloc_device(&residual_, static_cast<size_t>(s) * d * sizeof(float));
    alloc_device(&ffn_hidden_, static_cast<size_t>(s) * ff * sizeof(float));
    alloc_device(&ffn_out_, static_cast<size_t>(s) * d * sizeof(float));
}

void TransformerBlock::free_weights() {
    free_device(w_q_);
    free_device(w_k_);
    free_device(w_v_);
    free_device(w_o_);
    free_device(b_q_);
    free_device(b_k_);
    free_device(b_v_);
    free_device(b_o_);
    free_device(ln1_gamma_);
    free_device(ln1_beta_);
    free_device(ln2_gamma_);
    free_device(ln2_beta_);
    free_device(w1_);
    free_device(b1_);
    free_device(w2_);
    free_device(b2_);
}

void TransformerBlock::free_workspace() {
    free_device(x_norm_);
    free_device(q_);
    free_device(k_);
    free_device(v_);
    free_device(q_heads_);
    free_device(k_heads_);
    free_device(v_heads_);
    free_device(k_heads_t_);
    free_device(scores_);
    free_device(ctx_heads_);
    free_device(ctx_);
    free_device(attn_out_);
    free_device(residual_);
    free_device(ffn_hidden_);
    free_device(ffn_out_);
}

void TransformerBlock::forward(const float* d_input, float* d_output, cudaStream_t stream) {
    const int s = cfg_.seq_len;
    const int d = cfg_.hidden_size;
    const int h = cfg_.num_heads;
    const int ff = cfg_.ffn_size;

    launch_layernorm(d_input, ln1_gamma_, ln1_beta_, x_norm_, s, d, cfg_.layernorm_eps, stream);

    launch_sgemm_double_buffering(s, d, d, 1.0f, x_norm_, w_q_, 0.0f, q_, stream);
    launch_sgemm_double_buffering(s, d, d, 1.0f, x_norm_, w_k_, 0.0f, k_, stream);
    launch_sgemm_double_buffering(s, d, d, 1.0f, x_norm_, w_v_, 0.0f, v_, stream);
    launch_add_bias(q_, b_q_, s, d, stream);
    launch_add_bias(k_, b_k_, s, d, stream);
    launch_add_bias(v_, b_v_, s, d, stream);

    launch_pack_heads(q_, q_heads_, s, d, h, stream);
    launch_pack_heads(k_, k_heads_, s, d, h, stream);
    launch_pack_heads(v_, v_heads_, s, d, h, stream);

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    #pragma unroll
    for (int head = 0; head < h; ++head) {
        float* q_h = q_heads_ + static_cast<size_t>(head) * s * head_dim_;
        float* k_h = k_heads_ + static_cast<size_t>(head) * s * head_dim_;
        float* v_h = v_heads_ + static_cast<size_t>(head) * s * head_dim_;
        float* kt_h = k_heads_t_ + static_cast<size_t>(head) * head_dim_ * s;
        float* score_h = scores_ + static_cast<size_t>(head) * s * s;
        float* ctx_h = ctx_heads_ + static_cast<size_t>(head) * s * head_dim_;

        launch_transpose(k_h, kt_h, s, head_dim_, stream);
        launch_sgemm_double_buffering(s, s, head_dim_, 1.0f, q_h, kt_h, 0.0f, score_h, stream);
        launch_scale(score_h, scale, s * s, stream);
        launch_softmax_rows(score_h, s, s, stream);
        launch_sgemm_double_buffering(s, head_dim_, s, 1.0f, score_h, v_h, 0.0f, ctx_h, stream);
    }

    launch_unpack_heads(ctx_heads_, ctx_, s, d, h, stream);
    launch_sgemm_double_buffering(s, d, d, 1.0f, ctx_, w_o_, 0.0f, attn_out_, stream);
    launch_add_bias(attn_out_, b_o_, s, d, stream);
    launch_residual_add(d_input, attn_out_, residual_, s * d, stream);

    launch_layernorm(residual_, ln2_gamma_, ln2_beta_, x_norm_, s, d, cfg_.layernorm_eps, stream);
    launch_sgemm_double_buffering(s, ff, d, 1.0f, x_norm_, w1_, 0.0f, ffn_hidden_, stream);
    launch_add_bias(ffn_hidden_, b1_, s, ff, stream);
    launch_gelu(ffn_hidden_, s * ff, stream);

    launch_sgemm_double_buffering(s, d, ff, 1.0f, ffn_hidden_, w2_, 0.0f, ffn_out_, stream);
    launch_add_bias(ffn_out_, b2_, s, d, stream);
    launch_residual_add(residual_, ffn_out_, d_output, s * d, stream);
}
