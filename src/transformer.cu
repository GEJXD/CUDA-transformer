#include "transformer.h"

#include "cuda_utils.h"

#include <stdexcept>

namespace {

using cuda_utils::alloc_device;
using cuda_utils::free_device;

}  // namespace

Transformer::Transformer(const TransformerConfig& cfg) : cfg_(cfg) {
    if (cfg_.batch_size <= 0) {
        throw std::invalid_argument("batch_size must be > 0");
    }
    if (cfg_.src_seq_len <= 0 || cfg_.tgt_seq_len <= 0) {
        throw std::invalid_argument("src_seq_len and tgt_seq_len must be > 0");
    }
    if (cfg_.hidden_size % cfg_.num_heads != 0) {
        throw std::invalid_argument("hidden_size must be divisible by num_heads");
    }
    if (cfg_.num_encoder_layers <= 0 || cfg_.num_decoder_layers <= 0) {
        throw std::invalid_argument("num_encoder_layers and num_decoder_layers must be > 0");
    }

    encoder_block_cfg_ = EncoderBlockConfig{
        cfg_.precision,
        cfg_.batch_size,
        cfg_.src_seq_len,
        cfg_.hidden_size,
        cfg_.num_heads,
        cfg_.ffn_size,
        cfg_.layernorm_eps,
    };
    decoder_block_cfg_ = DecoderBlockConfig{
        cfg_.precision,
        cfg_.batch_size,
        cfg_.src_seq_len,
        cfg_.tgt_seq_len,
        cfg_.hidden_size,
        cfg_.num_heads,
        cfg_.ffn_size,
        cfg_.layernorm_eps,
    };

    encoder_layers_ = new EncoderBlock*[static_cast<size_t>(cfg_.num_encoder_layers)];
    for (int i = 0; i < cfg_.num_encoder_layers; ++i) {
        encoder_layers_[i] = new EncoderBlock(encoder_block_cfg_);
    }

    decoder_layers_ = new DecoderBlock*[static_cast<size_t>(cfg_.num_decoder_layers)];
    for (int i = 0; i < cfg_.num_decoder_layers; ++i) {
        decoder_layers_[i] = new DecoderBlock(decoder_block_cfg_);
    }

    allocate_workspace();
}

Transformer::~Transformer() {
    free_workspace();

    if (encoder_layers_ != nullptr) {
        for (int i = 0; i < cfg_.num_encoder_layers; ++i) {
            delete encoder_layers_[i];
        }
        delete[] encoder_layers_;
    }

    if (decoder_layers_ != nullptr) {
        for (int i = 0; i < cfg_.num_decoder_layers; ++i) {
            delete decoder_layers_[i];
        }
        delete[] decoder_layers_;
    }
}

void Transformer::allocate_workspace() {
    const size_t enc_bytes = static_cast<size_t>(cfg_.batch_size) * cfg_.src_seq_len * cfg_.hidden_size * sizeof(float);
    const size_t dec_bytes = static_cast<size_t>(cfg_.batch_size) * cfg_.tgt_seq_len * cfg_.hidden_size * sizeof(float);

    alloc_device(&enc_buffer_a_, enc_bytes);
    if (cfg_.num_encoder_layers > 1) {
        alloc_device(&enc_buffer_b_, enc_bytes);
    }
    if (cfg_.num_decoder_layers > 1) {
        alloc_device(&dec_buffer_a_, dec_bytes);
        alloc_device(&dec_buffer_b_, dec_bytes);
    }
}

void Transformer::free_workspace() {
    free_device(enc_buffer_a_);
    free_device(enc_buffer_b_);
    free_device(dec_buffer_a_);
    free_device(dec_buffer_b_);
}

void Transformer::forward(
    const float* d_encoder_input,
    const float* d_decoder_input,
    float* d_output,
    cudaStream_t stream) {
    const float* enc_current = d_encoder_input;
    float* enc_next = enc_buffer_a_;
    float* enc_alt = enc_buffer_b_;

    float* encoder_memory = enc_buffer_a_;
    for (int i = 0; i < cfg_.num_encoder_layers; ++i) {
        if (i == cfg_.num_encoder_layers - 1) {
            enc_next = encoder_memory;
        }

        encoder_layers_[i]->forward(enc_current, enc_next, stream);
        enc_current = enc_next;
        float* tmp = enc_next;
        enc_next = enc_alt;
        enc_alt = tmp;
    }

    const float* dec_current = d_decoder_input;
    float* dec_next = dec_buffer_a_;
    float* dec_alt = dec_buffer_b_;

    for (int i = 0; i < cfg_.num_decoder_layers; ++i) {
        if (i == cfg_.num_decoder_layers - 1) {
            dec_next = d_output;
        }
        decoder_layers_[i]->forward(dec_current, encoder_memory, dec_next, stream);
        dec_current = dec_next;
        float* tmp = dec_next;
        dec_next = dec_alt;
        dec_alt = tmp;
    }
}
