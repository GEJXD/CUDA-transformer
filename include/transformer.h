#pragma once

#include "decoder_block.h"
#include "encoder_block.h"

#include <cuda_runtime.h>

struct TransformerConfig {
    Precision precision;
    int batch_size;
    int src_seq_len;
    int tgt_seq_len;
    int hidden_size;
    int num_heads;
    int ffn_size;
    int num_encoder_layers;
    int num_decoder_layers;
    float layernorm_eps;
};

class Transformer {
public:
    explicit Transformer(const TransformerConfig& cfg);
    ~Transformer();

    Transformer(const Transformer&) = delete;
    Transformer& operator=(const Transformer&) = delete;

    void forward(
        const float* d_encoder_input,
        const float* d_decoder_input,
        float* d_output,
        cudaStream_t stream = nullptr);

private:
    void allocate_workspace();
    void free_workspace();

    TransformerConfig cfg_{};
    EncoderBlockConfig encoder_block_cfg_{};
    DecoderBlockConfig decoder_block_cfg_{};

    EncoderBlock** encoder_layers_ = nullptr;
    DecoderBlock** decoder_layers_ = nullptr;

    float* enc_buffer_a_ = nullptr;
    float* enc_buffer_b_ = nullptr;
    float* dec_buffer_a_ = nullptr;
    float* dec_buffer_b_ = nullptr;
};
