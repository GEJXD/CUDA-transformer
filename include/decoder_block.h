#pragma once

#include "encoder_block.h"

#include <cuda_runtime.h>

struct DecoderBlockConfig {
    Precision precision;
    int batch_size;
    int src_seq_len;
    int tgt_seq_len;
    int hidden_size;
    int num_heads;
    int ffn_size;
    float layernorm_eps;
};

class DecoderBlock {
public:
    explicit DecoderBlock(const DecoderBlockConfig& cfg);
    ~DecoderBlock();

    DecoderBlock(const DecoderBlock&) = delete;
    DecoderBlock& operator=(const DecoderBlock&) = delete;

    void forward(
        const float* d_input,
        const float* d_enc_out,
        float* d_output,
        cudaStream_t stream = nullptr);

private:
    void init_weights();
    void allocate_workspace();
    void free_weights();
    void free_workspace();

    DecoderBlockConfig cfg_{};
    int head_dim_ = 0;

    float* w_q_self_ = nullptr;
    float* w_k_self_ = nullptr;
    float* w_v_self_ = nullptr;
    float* w_o_self_ = nullptr;
    float* b_q_self_ = nullptr;
    float* b_k_self_ = nullptr;
    float* b_v_self_ = nullptr;
    float* b_o_self_ = nullptr;

    float* w_q_cross_ = nullptr;
    float* w_k_cross_ = nullptr;
    float* w_v_cross_ = nullptr;
    float* w_o_cross_ = nullptr;
    float* b_q_cross_ = nullptr;
    float* b_k_cross_ = nullptr;
    float* b_v_cross_ = nullptr;
    float* b_o_cross_ = nullptr;

    float* ln1_gamma_ = nullptr;
    float* ln1_beta_ = nullptr;
    float* ln2_gamma_ = nullptr;
    float* ln2_beta_ = nullptr;
    float* ln3_gamma_ = nullptr;
    float* ln3_beta_ = nullptr;

    float* w1_ = nullptr;
    float* b1_ = nullptr;
    float* w2_ = nullptr;
    float* b2_ = nullptr;

    __half* w_q_self_fp16_ = nullptr;
    __half* w_k_self_fp16_ = nullptr;
    __half* w_v_self_fp16_ = nullptr;
    __half* w_o_self_fp16_ = nullptr;
    __half* w_q_cross_fp16_ = nullptr;
    __half* w_k_cross_fp16_ = nullptr;
    __half* w_v_cross_fp16_ = nullptr;
    __half* w_o_cross_fp16_ = nullptr;
    __half* w1_fp16_ = nullptr;
    __half* w2_fp16_ = nullptr;

    float* x_norm_ = nullptr;

    float* q_self_ = nullptr;
    float* k_self_ = nullptr;
    float* v_self_ = nullptr;
    float* qh_self_ = nullptr;
    float* kh_self_ = nullptr;
    float* kh_self_t_ = nullptr;
    float* vh_self_ = nullptr;
    float* scores_self_ = nullptr;
    float* ctxh_self_ = nullptr;
    float* ctx_self_ = nullptr;
    float* self_out_ = nullptr;
    float* residual1_ = nullptr;

    float* q_cross_ = nullptr;
    float* k_cross_ = nullptr;
    float* v_cross_ = nullptr;
    float* qh_cross_ = nullptr;
    float* kh_cross_ = nullptr;
    float* kh_cross_t_ = nullptr;
    float* vh_cross_ = nullptr;
    float* scores_cross_ = nullptr;
    float* ctxh_cross_ = nullptr;
    float* ctx_cross_ = nullptr;
    float* cross_out_ = nullptr;
    float* residual2_ = nullptr;

    float* ffn_hidden_ = nullptr;
    float* ffn_out_ = nullptr;

    __half* act_fp16_ = nullptr;
    void* cublas_handle_ = nullptr;
};
