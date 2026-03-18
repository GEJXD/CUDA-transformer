#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

enum class Precision {
    FP32,
    FP16,
};

struct EncoderBlockConfig {
    Precision precision;
    int batch_size;
    int seq_len;
    int hidden_size;
    int num_heads;
    int ffn_size;
    float layernorm_eps;
};

class EncoderBlock {
public:
    explicit EncoderBlock(const EncoderBlockConfig& cfg);
    ~EncoderBlock();

    EncoderBlock(const EncoderBlock&) = delete;
    EncoderBlock& operator=(const EncoderBlock&) = delete;

    void forward(const float* d_input, float* d_output, cudaStream_t stream = nullptr);

private:
    void init_weights();
    void allocate_workspace();
    void free_weights();
    void free_workspace();

    EncoderBlockConfig cfg_{};
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

    __half* w_q_fp16_ = nullptr;
    __half* w_k_fp16_ = nullptr;
    __half* w_v_fp16_ = nullptr;
    __half* w_o_fp16_ = nullptr;
    __half* w1_fp16_ = nullptr;
    __half* w2_fp16_ = nullptr;

    float* x_norm_ = nullptr;
    float* q_ = nullptr;
    float* k_ = nullptr;
    float* v_ = nullptr;

    float* q_heads_ = nullptr;
    float* k_heads_ = nullptr;
    float* k_heads_t_ = nullptr;
    float* v_heads_ = nullptr;
    float* scores_ = nullptr;
    float* ctx_heads_ = nullptr;

    float* ctx_ = nullptr;
    float* attn_out_ = nullptr;
    float* residual_ = nullptr;
    float* ffn_hidden_ = nullptr;
    float* ffn_out_ = nullptr;

    __half* act_fp16_ = nullptr;
    void* cublas_handle_ = nullptr;
};
