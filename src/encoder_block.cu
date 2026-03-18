#include "encoder_block.h"

#include "block_common.h"
#include "cuda_utils.h"
#include "cuda_ops.h"
#include "gemm.h"

#include <cublas_v2.h>
#include <cmath>
#include <stdexcept>

namespace {

using cuda_utils::alloc_device;
using cuda_utils::alloc_device_half;
using cuda_utils::free_device;
using cuda_utils::free_device_half;

}  // namespace

EncoderBlock::EncoderBlock(const EncoderBlockConfig& cfg) : cfg_(cfg) {
    if (cfg_.precision != Precision::FP32 && cfg_.precision != Precision::FP16) {
        throw std::invalid_argument("precision must be FP32 or FP16");
    }
    if (cfg_.batch_size <= 0) {
        throw std::invalid_argument("batch_size must be > 0");
    }
    if (cfg_.hidden_size % cfg_.num_heads != 0) {
        throw std::invalid_argument("hidden_size must be divisible by num_heads");
    }
    head_dim_ = cfg_.hidden_size / cfg_.num_heads;

    if (cfg_.precision == Precision::FP16) {
        cublasHandle_t handle = nullptr;
        CUBLAS_CHECK(cublasCreate(&handle));
        cublas_handle_ = handle;
    }

    init_weights();
    allocate_workspace();
}

EncoderBlock::~EncoderBlock() {
    free_workspace();
    free_weights();
    if (cublas_handle_ != nullptr) {
        cublasDestroy(static_cast<cublasHandle_t>(cublas_handle_));
        cublas_handle_ = nullptr;
    }
}

void EncoderBlock::init_weights() {
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

    if (cfg_.precision == Precision::FP16) {
        alloc_device_half(&w_q_fp16_, static_cast<size_t>(d) * d);
        alloc_device_half(&w_k_fp16_, static_cast<size_t>(d) * d);
        alloc_device_half(&w_v_fp16_, static_cast<size_t>(d) * d);
        alloc_device_half(&w_o_fp16_, static_cast<size_t>(d) * d);
        alloc_device_half(&w1_fp16_, static_cast<size_t>(d) * ff);
        alloc_device_half(&w2_fp16_, static_cast<size_t>(ff) * d);

        float_to_half_kernel(w_q_, w_q_fp16_, d * d, nullptr);
        float_to_half_kernel(w_k_, w_k_fp16_, d * d, nullptr);
        float_to_half_kernel(w_v_, w_v_fp16_, d * d, nullptr);
        float_to_half_kernel(w_o_, w_o_fp16_, d * d, nullptr);
        float_to_half_kernel(w1_, w1_fp16_, d * ff, nullptr);
        float_to_half_kernel(w2_, w2_fp16_, ff * d, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

void EncoderBlock::allocate_workspace() {
    const int b = cfg_.batch_size;
    const int s = cfg_.seq_len;
    const int d = cfg_.hidden_size;
    const int h = cfg_.num_heads;
    const int ff = cfg_.ffn_size;
    const int m = b * s;

    alloc_device(&x_norm_, static_cast<size_t>(m) * d * sizeof(float));
    alloc_device(&q_, static_cast<size_t>(m) * d * sizeof(float));
    alloc_device(&k_, static_cast<size_t>(m) * d * sizeof(float));
    alloc_device(&v_, static_cast<size_t>(m) * d * sizeof(float));

    alloc_device(&q_heads_, static_cast<size_t>(b) * h * s * head_dim_ * sizeof(float));
    alloc_device(&k_heads_, static_cast<size_t>(b) * h * s * head_dim_ * sizeof(float));
    alloc_device(&k_heads_t_, static_cast<size_t>(b) * h * head_dim_ * s * sizeof(float));
    alloc_device(&v_heads_, static_cast<size_t>(b) * h * s * head_dim_ * sizeof(float));
    alloc_device(&scores_, static_cast<size_t>(b) * h * s * s * sizeof(float));
    alloc_device(&ctx_heads_, static_cast<size_t>(b) * h * s * head_dim_ * sizeof(float));

    alloc_device(&ctx_, static_cast<size_t>(m) * d * sizeof(float));
    alloc_device(&attn_out_, static_cast<size_t>(m) * d * sizeof(float));
    alloc_device(&residual_, static_cast<size_t>(m) * d * sizeof(float));
    alloc_device(&ffn_hidden_, static_cast<size_t>(m) * ff * sizeof(float));
    alloc_device(&ffn_out_, static_cast<size_t>(m) * d * sizeof(float));

    if (cfg_.precision == Precision::FP16) {
        const size_t max_act = static_cast<size_t>(m) * ((ff > d) ? ff : d);
        alloc_device_half(&act_fp16_, max_act);
    }
}

void EncoderBlock::free_weights() {
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

    free_device_half(w_q_fp16_);
    free_device_half(w_k_fp16_);
    free_device_half(w_v_fp16_);
    free_device_half(w_o_fp16_);
    free_device_half(w1_fp16_);
    free_device_half(w2_fp16_);
}

void EncoderBlock::free_workspace() {
    free_device(x_norm_);
    free_device(q_);
    free_device(k_);
    free_device(v_);
    free_device(q_heads_);
    free_device(k_heads_);
    free_device(k_heads_t_);
    free_device(v_heads_);
    free_device(scores_);
    free_device(ctx_heads_);
    free_device(ctx_);
    free_device(attn_out_);
    free_device(residual_);
    free_device(ffn_hidden_);
    free_device(ffn_out_);
    free_device_half(act_fp16_);
}

void EncoderBlock::forward(const float* d_input, float* d_output, cudaStream_t stream) {
    const int b = cfg_.batch_size;
    const int s = cfg_.seq_len;
    const int d = cfg_.hidden_size;
    const int h = cfg_.num_heads;
    const int ff = cfg_.ffn_size;
    const int m = b * s;
    const bool use_fp16 = (cfg_.precision == Precision::FP16);
    cublasHandle_t handle = static_cast<cublasHandle_t>(cublas_handle_);

    launch_layernorm(d_input, ln1_gamma_, ln1_beta_, x_norm_, m, d, cfg_.layernorm_eps, stream);

    if (use_fp16) {
        float_to_half_kernel(x_norm_, act_fp16_, m * d, stream);
        gemm_fp16_tensor_core(handle, m, d, d, act_fp16_, w_q_fp16_, q_, stream);
        gemm_fp16_tensor_core(handle, m, d, d, act_fp16_, w_k_fp16_, k_, stream);
        gemm_fp16_tensor_core(handle, m, d, d, act_fp16_, w_v_fp16_, v_, stream);
    } else {
        launch_sgemm_double_buffering(m, d, d, 1.0f, x_norm_, w_q_, 0.0f, q_, stream);
        launch_sgemm_double_buffering(m, d, d, 1.0f, x_norm_, w_k_, 0.0f, k_, stream);
        launch_sgemm_double_buffering(m, d, d, 1.0f, x_norm_, w_v_, 0.0f, v_, stream);
    }
    launch_add_bias(q_, b_q_, m, d, stream);
    launch_add_bias(k_, b_k_, m, d, stream);
    launch_add_bias(v_, b_v_, m, d, stream);

    pack_heads_batched_kernel(q_, q_heads_, b, s, d, h, head_dim_, stream);
    pack_heads_batched_kernel(k_, k_heads_, b, s, d, h, head_dim_, stream);
    pack_heads_batched_kernel(v_, v_heads_, b, s, d, h, head_dim_, stream);

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    launch_attention_scores_batched_gemm(
        q_heads_,
        k_heads_,
        k_heads_t_,
        scores_,
        b,
        h,
        s,
        s,
        head_dim_,
        scale,
        false,
        stream);
    launch_softmax_rows(scores_, b * h * s, s, stream);

    attention_context_batched_kernel(
        scores_,
        v_heads_,
        ctx_heads_,
        b,
        h,
        s,
        s,
        head_dim_,
        stream);

    unpack_heads_batched_kernel(ctx_heads_, ctx_, b, s, d, h, head_dim_, stream);
    if (use_fp16) {
        float_to_half_kernel(ctx_, act_fp16_, m * d, stream);
        gemm_fp16_tensor_core(handle, m, d, d, act_fp16_, w_o_fp16_, attn_out_, stream);
    } else {
        launch_sgemm_double_buffering(m, d, d, 1.0f, ctx_, w_o_, 0.0f, attn_out_, stream);
    }
    launch_add_bias(attn_out_, b_o_, m, d, stream);
    launch_residual_add(d_input, attn_out_, residual_, m * d, stream);

    launch_layernorm(residual_, ln2_gamma_, ln2_beta_, x_norm_, m, d, cfg_.layernorm_eps, stream);
    if (use_fp16) {
        float_to_half_kernel(x_norm_, act_fp16_, m * d, stream);
        gemm_fp16_tensor_core(handle, m, ff, d, act_fp16_, w1_fp16_, ffn_hidden_, stream);
    } else {
        launch_sgemm_double_buffering(m, ff, d, 1.0f, x_norm_, w1_, 0.0f, ffn_hidden_, stream);
    }
    launch_add_bias(ffn_hidden_, b1_, m, ff, stream);
    launch_gelu(ffn_hidden_, m * ff, stream);

    if (use_fp16) {
        float_to_half_kernel(ffn_hidden_, act_fp16_, m * ff, stream);
        gemm_fp16_tensor_core(handle, m, d, ff, act_fp16_, w2_fp16_, ffn_out_, stream);
    } else {
        launch_sgemm_double_buffering(m, d, ff, 1.0f, ffn_hidden_, w2_, 0.0f, ffn_out_, stream);
    }
    launch_add_bias(ffn_out_, b2_, m, d, stream);
    launch_residual_add(residual_, ffn_out_, d_output, m * d, stream);
}
