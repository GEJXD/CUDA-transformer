#pragma once

#include <cuda_runtime.h>

struct TransformerConfig {
    int seq_len;
    int hidden_size;
    int num_heads;
    int ffn_size;
    float layernorm_eps;
};

class TransformerBlock {
public:
    explicit TransformerBlock(const TransformerConfig& cfg);
    ~TransformerBlock();

    TransformerBlock(const TransformerBlock&) = delete;
    TransformerBlock& operator=(const TransformerBlock&) = delete;

    void forward(const float* d_input, float* d_output, cudaStream_t stream = nullptr);

private:
    void init_weights();
    void allocate_workspace();
    void free_weights();
    void free_workspace();

    TransformerConfig cfg_{};
    int head_dim_ = 0;

    float* w_q_ = nullptr;
    float* w_k_ = nullptr;
    float* w_v_ = nullptr;
    float* w_o_ = nullptr;
    float* b_q_ = nullptr;
    float* b_k_ = nullptr;
    float* b_v_ = nullptr;
    float* b_o_ = nullptr;

    float* ln1_gamma_ = nullptr;
    float* ln1_beta_ = nullptr;
    float* ln2_gamma_ = nullptr;
    float* ln2_beta_ = nullptr;

    float* w1_ = nullptr;
    float* b1_ = nullptr;
    float* w2_ = nullptr;
    float* b2_ = nullptr;

    float* x_norm_ = nullptr;
    float* q_ = nullptr;
    float* k_ = nullptr;
    float* v_ = nullptr;

    float* q_heads_ = nullptr;
    float* k_heads_ = nullptr;
    float* v_heads_ = nullptr;
    float* k_heads_t_ = nullptr;
    float* scores_ = nullptr;
    float* ctx_heads_ = nullptr;

    float* ctx_ = nullptr;
    float* attn_out_ = nullptr;
    float* residual_ = nullptr;
    float* ffn_hidden_ = nullptr;
    float* ffn_out_ = nullptr;
};
