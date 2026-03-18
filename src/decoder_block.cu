#include "decoder_block.h"

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

DecoderBlock::DecoderBlock(const DecoderBlockConfig& cfg) : cfg_(cfg) {
    if (cfg_.precision != Precision::FP32 && cfg_.precision != Precision::FP16) {
        throw std::invalid_argument("precision must be FP32 or FP16");
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

DecoderBlock::~DecoderBlock() {
    free_workspace();
    free_weights();
    if (cublas_handle_ != nullptr) {
        cublasDestroy(static_cast<cublasHandle_t>(cublas_handle_));
        cublas_handle_ = nullptr;
    }
}

void DecoderBlock::init_weights() {
    const int d = cfg_.hidden_size;
    const int ff = cfg_.ffn_size;

    alloc_device(&w_q_self_, static_cast<size_t>(d) * d * sizeof(float));
    alloc_device(&w_k_self_, static_cast<size_t>(d) * d * sizeof(float));
    alloc_device(&w_v_self_, static_cast<size_t>(d) * d * sizeof(float));
    alloc_device(&w_o_self_, static_cast<size_t>(d) * d * sizeof(float));
    alloc_device(&b_q_self_, static_cast<size_t>(d) * sizeof(float));
    alloc_device(&b_k_self_, static_cast<size_t>(d) * sizeof(float));
    alloc_device(&b_v_self_, static_cast<size_t>(d) * sizeof(float));
    alloc_device(&b_o_self_, static_cast<size_t>(d) * sizeof(float));

    alloc_device(&w_q_cross_, static_cast<size_t>(d) * d * sizeof(float));
    alloc_device(&w_k_cross_, static_cast<size_t>(d) * d * sizeof(float));
    alloc_device(&w_v_cross_, static_cast<size_t>(d) * d * sizeof(float));
    alloc_device(&w_o_cross_, static_cast<size_t>(d) * d * sizeof(float));
    alloc_device(&b_q_cross_, static_cast<size_t>(d) * sizeof(float));
    alloc_device(&b_k_cross_, static_cast<size_t>(d) * sizeof(float));
    alloc_device(&b_v_cross_, static_cast<size_t>(d) * sizeof(float));
    alloc_device(&b_o_cross_, static_cast<size_t>(d) * sizeof(float));

    alloc_device(&ln1_gamma_, static_cast<size_t>(d) * sizeof(float));
    alloc_device(&ln1_beta_, static_cast<size_t>(d) * sizeof(float));
    alloc_device(&ln2_gamma_, static_cast<size_t>(d) * sizeof(float));
    alloc_device(&ln2_beta_, static_cast<size_t>(d) * sizeof(float));
    alloc_device(&ln3_gamma_, static_cast<size_t>(d) * sizeof(float));
    alloc_device(&ln3_beta_, static_cast<size_t>(d) * sizeof(float));

    alloc_device(&w1_, static_cast<size_t>(d) * ff * sizeof(float));
    alloc_device(&b1_, static_cast<size_t>(ff) * sizeof(float));
    alloc_device(&w2_, static_cast<size_t>(ff) * d * sizeof(float));
    alloc_device(&b2_, static_cast<size_t>(d) * sizeof(float));

    init_param(w_q_self_, static_cast<size_t>(d) * d, -0.02f, 0.02f);
    init_param(w_k_self_, static_cast<size_t>(d) * d, -0.02f, 0.02f);
    init_param(w_v_self_, static_cast<size_t>(d) * d, -0.02f, 0.02f);
    init_param(w_o_self_, static_cast<size_t>(d) * d, -0.02f, 0.02f);
    init_constant(b_q_self_, d, 0.0f);
    init_constant(b_k_self_, d, 0.0f);
    init_constant(b_v_self_, d, 0.0f);
    init_constant(b_o_self_, d, 0.0f);

    init_param(w_q_cross_, static_cast<size_t>(d) * d, -0.02f, 0.02f);
    init_param(w_k_cross_, static_cast<size_t>(d) * d, -0.02f, 0.02f);
    init_param(w_v_cross_, static_cast<size_t>(d) * d, -0.02f, 0.02f);
    init_param(w_o_cross_, static_cast<size_t>(d) * d, -0.02f, 0.02f);
    init_constant(b_q_cross_, d, 0.0f);
    init_constant(b_k_cross_, d, 0.0f);
    init_constant(b_v_cross_, d, 0.0f);
    init_constant(b_o_cross_, d, 0.0f);

    init_constant(ln1_gamma_, d, 1.0f);
    init_constant(ln1_beta_, d, 0.0f);
    init_constant(ln2_gamma_, d, 1.0f);
    init_constant(ln2_beta_, d, 0.0f);
    init_constant(ln3_gamma_, d, 1.0f);
    init_constant(ln3_beta_, d, 0.0f);

    init_param(w1_, static_cast<size_t>(d) * ff, -0.02f, 0.02f);
    init_constant(b1_, ff, 0.0f);
    init_param(w2_, static_cast<size_t>(ff) * d, -0.02f, 0.02f);
    init_constant(b2_, d, 0.0f);

    if (cfg_.precision == Precision::FP16) {
        alloc_device_half(&w_q_self_fp16_, static_cast<size_t>(d) * d);
        alloc_device_half(&w_k_self_fp16_, static_cast<size_t>(d) * d);
        alloc_device_half(&w_v_self_fp16_, static_cast<size_t>(d) * d);
        alloc_device_half(&w_o_self_fp16_, static_cast<size_t>(d) * d);

        alloc_device_half(&w_q_cross_fp16_, static_cast<size_t>(d) * d);
        alloc_device_half(&w_k_cross_fp16_, static_cast<size_t>(d) * d);
        alloc_device_half(&w_v_cross_fp16_, static_cast<size_t>(d) * d);
        alloc_device_half(&w_o_cross_fp16_, static_cast<size_t>(d) * d);

        alloc_device_half(&w1_fp16_, static_cast<size_t>(d) * ff);
        alloc_device_half(&w2_fp16_, static_cast<size_t>(ff) * d);

        float_to_half_kernel(w_q_self_, w_q_self_fp16_, d * d, nullptr);
        float_to_half_kernel(w_k_self_, w_k_self_fp16_, d * d, nullptr);
        float_to_half_kernel(w_v_self_, w_v_self_fp16_, d * d, nullptr);
        float_to_half_kernel(w_o_self_, w_o_self_fp16_, d * d, nullptr);

        float_to_half_kernel(w_q_cross_, w_q_cross_fp16_, d * d, nullptr);
        float_to_half_kernel(w_k_cross_, w_k_cross_fp16_, d * d, nullptr);
        float_to_half_kernel(w_v_cross_, w_v_cross_fp16_, d * d, nullptr);
        float_to_half_kernel(w_o_cross_, w_o_cross_fp16_, d * d, nullptr);

        float_to_half_kernel(w1_, w1_fp16_, d * ff, nullptr);
        float_to_half_kernel(w2_, w2_fp16_, ff * d, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

void DecoderBlock::allocate_workspace() {
    const int b = cfg_.batch_size;
    const int s = cfg_.src_seq_len;
    const int t = cfg_.tgt_seq_len;
    const int d = cfg_.hidden_size;
    const int h = cfg_.num_heads;
    const int ff = cfg_.ffn_size;
    const int mt = b * t;
    const int ms = b * s;

    alloc_device(&x_norm_, static_cast<size_t>(mt) * d * sizeof(float));

    alloc_device(&q_self_, static_cast<size_t>(mt) * d * sizeof(float));
    alloc_device(&k_self_, static_cast<size_t>(mt) * d * sizeof(float));
    alloc_device(&v_self_, static_cast<size_t>(mt) * d * sizeof(float));
    alloc_device(&qh_self_, static_cast<size_t>(b) * h * t * head_dim_ * sizeof(float));
    alloc_device(&kh_self_, static_cast<size_t>(b) * h * t * head_dim_ * sizeof(float));
    alloc_device(&kh_self_t_, static_cast<size_t>(b) * h * head_dim_ * t * sizeof(float));
    alloc_device(&vh_self_, static_cast<size_t>(b) * h * t * head_dim_ * sizeof(float));
    alloc_device(&scores_self_, static_cast<size_t>(b) * h * t * t * sizeof(float));
    alloc_device(&ctxh_self_, static_cast<size_t>(b) * h * t * head_dim_ * sizeof(float));
    alloc_device(&ctx_self_, static_cast<size_t>(mt) * d * sizeof(float));
    alloc_device(&self_out_, static_cast<size_t>(mt) * d * sizeof(float));
    alloc_device(&residual1_, static_cast<size_t>(mt) * d * sizeof(float));

    alloc_device(&q_cross_, static_cast<size_t>(mt) * d * sizeof(float));
    alloc_device(&k_cross_, static_cast<size_t>(ms) * d * sizeof(float));
    alloc_device(&v_cross_, static_cast<size_t>(ms) * d * sizeof(float));
    alloc_device(&qh_cross_, static_cast<size_t>(b) * h * t * head_dim_ * sizeof(float));
    alloc_device(&kh_cross_, static_cast<size_t>(b) * h * s * head_dim_ * sizeof(float));
    alloc_device(&kh_cross_t_, static_cast<size_t>(b) * h * head_dim_ * s * sizeof(float));
    alloc_device(&vh_cross_, static_cast<size_t>(b) * h * s * head_dim_ * sizeof(float));
    alloc_device(&scores_cross_, static_cast<size_t>(b) * h * t * s * sizeof(float));
    alloc_device(&ctxh_cross_, static_cast<size_t>(b) * h * t * head_dim_ * sizeof(float));
    alloc_device(&ctx_cross_, static_cast<size_t>(mt) * d * sizeof(float));
    alloc_device(&cross_out_, static_cast<size_t>(mt) * d * sizeof(float));
    alloc_device(&residual2_, static_cast<size_t>(mt) * d * sizeof(float));

    alloc_device(&ffn_hidden_, static_cast<size_t>(mt) * ff * sizeof(float));
    alloc_device(&ffn_out_, static_cast<size_t>(mt) * d * sizeof(float));

    if (cfg_.precision == Precision::FP16) {
        const size_t max_act = static_cast<size_t>((ms > mt) ? ms : mt) * ((ff > d) ? ff : d);
        alloc_device_half(&act_fp16_, max_act);
    }
}

void DecoderBlock::free_weights() {
    free_device(w_q_self_);
    free_device(w_k_self_);
    free_device(w_v_self_);
    free_device(w_o_self_);
    free_device(b_q_self_);
    free_device(b_k_self_);
    free_device(b_v_self_);
    free_device(b_o_self_);

    free_device(w_q_cross_);
    free_device(w_k_cross_);
    free_device(w_v_cross_);
    free_device(w_o_cross_);
    free_device(b_q_cross_);
    free_device(b_k_cross_);
    free_device(b_v_cross_);
    free_device(b_o_cross_);

    free_device(ln1_gamma_);
    free_device(ln1_beta_);
    free_device(ln2_gamma_);
    free_device(ln2_beta_);
    free_device(ln3_gamma_);
    free_device(ln3_beta_);

    free_device(w1_);
    free_device(b1_);
    free_device(w2_);
    free_device(b2_);

    free_device_half(w_q_self_fp16_);
    free_device_half(w_k_self_fp16_);
    free_device_half(w_v_self_fp16_);
    free_device_half(w_o_self_fp16_);

    free_device_half(w_q_cross_fp16_);
    free_device_half(w_k_cross_fp16_);
    free_device_half(w_v_cross_fp16_);
    free_device_half(w_o_cross_fp16_);

    free_device_half(w1_fp16_);
    free_device_half(w2_fp16_);
}

void DecoderBlock::free_workspace() {
    free_device(x_norm_);

    free_device(q_self_);
    free_device(k_self_);
    free_device(v_self_);
    free_device(qh_self_);
    free_device(kh_self_);
    free_device(kh_self_t_);
    free_device(vh_self_);
    free_device(scores_self_);
    free_device(ctxh_self_);
    free_device(ctx_self_);
    free_device(self_out_);
    free_device(residual1_);

    free_device(q_cross_);
    free_device(k_cross_);
    free_device(v_cross_);
    free_device(qh_cross_);
    free_device(kh_cross_);
    free_device(kh_cross_t_);
    free_device(vh_cross_);
    free_device(scores_cross_);
    free_device(ctxh_cross_);
    free_device(ctx_cross_);
    free_device(cross_out_);
    free_device(residual2_);

    free_device(ffn_hidden_);
    free_device(ffn_out_);

    free_device_half(act_fp16_);
}

void DecoderBlock::forward(const float* d_input, const float* d_enc_out, float* d_output, cudaStream_t stream) {
    const int b = cfg_.batch_size;
    const int s = cfg_.src_seq_len;
    const int t = cfg_.tgt_seq_len;
    const int d = cfg_.hidden_size;
    const int h = cfg_.num_heads;
    const int ff = cfg_.ffn_size;
    const int mt = b * t;
    const int ms = b * s;
    const bool use_fp16 = (cfg_.precision == Precision::FP16);
    cublasHandle_t handle = static_cast<cublasHandle_t>(cublas_handle_);

    launch_layernorm(d_input, ln1_gamma_, ln1_beta_, x_norm_, mt, d, cfg_.layernorm_eps, stream);
    if (use_fp16) {
        float_to_half_kernel(x_norm_, act_fp16_, mt * d, stream);
        gemm_fp16_tensor_core(handle, mt, d, d, act_fp16_, w_q_self_fp16_, q_self_, stream);
        gemm_fp16_tensor_core(handle, mt, d, d, act_fp16_, w_k_self_fp16_, k_self_, stream);
        gemm_fp16_tensor_core(handle, mt, d, d, act_fp16_, w_v_self_fp16_, v_self_, stream);
    } else {
        launch_sgemm_double_buffering(mt, d, d, 1.0f, x_norm_, w_q_self_, 0.0f, q_self_, stream);
        launch_sgemm_double_buffering(mt, d, d, 1.0f, x_norm_, w_k_self_, 0.0f, k_self_, stream);
        launch_sgemm_double_buffering(mt, d, d, 1.0f, x_norm_, w_v_self_, 0.0f, v_self_, stream);
    }
    launch_add_bias(q_self_, b_q_self_, mt, d, stream);
    launch_add_bias(k_self_, b_k_self_, mt, d, stream);
    launch_add_bias(v_self_, b_v_self_, mt, d, stream);

    pack_heads_batched_kernel(q_self_, qh_self_, b, t, d, h, head_dim_, stream);
    pack_heads_batched_kernel(k_self_, kh_self_, b, t, d, h, head_dim_, stream);
    pack_heads_batched_kernel(v_self_, vh_self_, b, t, d, h, head_dim_, stream);

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    launch_attention_scores_batched_gemm(
        qh_self_, kh_self_, kh_self_t_, scores_self_, b, h, t, t, head_dim_, scale, true, stream);
    launch_softmax_rows(scores_self_, b * h * t, t, stream);

    attention_context_batched_kernel(
        scores_self_, vh_self_, ctxh_self_, b, h, t, t, head_dim_, stream);

    unpack_heads_batched_kernel(ctxh_self_, ctx_self_, b, t, d, h, head_dim_, stream);
    if (use_fp16) {
        float_to_half_kernel(ctx_self_, act_fp16_, mt * d, stream);
        gemm_fp16_tensor_core(handle, mt, d, d, act_fp16_, w_o_self_fp16_, self_out_, stream);
    } else {
        launch_sgemm_double_buffering(mt, d, d, 1.0f, ctx_self_, w_o_self_, 0.0f, self_out_, stream);
    }
    launch_add_bias(self_out_, b_o_self_, mt, d, stream);
    launch_residual_add(d_input, self_out_, residual1_, mt * d, stream);

    launch_layernorm(residual1_, ln2_gamma_, ln2_beta_, x_norm_, mt, d, cfg_.layernorm_eps, stream);
    if (use_fp16) {
        float_to_half_kernel(x_norm_, act_fp16_, mt * d, stream);
        gemm_fp16_tensor_core(handle, mt, d, d, act_fp16_, w_q_cross_fp16_, q_cross_, stream);

        float_to_half_kernel(d_enc_out, act_fp16_, ms * d, stream);
        gemm_fp16_tensor_core(handle, ms, d, d, act_fp16_, w_k_cross_fp16_, k_cross_, stream);
        gemm_fp16_tensor_core(handle, ms, d, d, act_fp16_, w_v_cross_fp16_, v_cross_, stream);
    } else {
        launch_sgemm_double_buffering(mt, d, d, 1.0f, x_norm_, w_q_cross_, 0.0f, q_cross_, stream);
        launch_sgemm_double_buffering(ms, d, d, 1.0f, d_enc_out, w_k_cross_, 0.0f, k_cross_, stream);
        launch_sgemm_double_buffering(ms, d, d, 1.0f, d_enc_out, w_v_cross_, 0.0f, v_cross_, stream);
    }
    launch_add_bias(q_cross_, b_q_cross_, mt, d, stream);
    launch_add_bias(k_cross_, b_k_cross_, ms, d, stream);
    launch_add_bias(v_cross_, b_v_cross_, ms, d, stream);

    pack_heads_batched_kernel(q_cross_, qh_cross_, b, t, d, h, head_dim_, stream);
    pack_heads_batched_kernel(k_cross_, kh_cross_, b, s, d, h, head_dim_, stream);
    pack_heads_batched_kernel(v_cross_, vh_cross_, b, s, d, h, head_dim_, stream);

    launch_attention_scores_batched_gemm(
        qh_cross_, kh_cross_, kh_cross_t_, scores_cross_, b, h, t, s, head_dim_, scale, false, stream);
    launch_softmax_rows(scores_cross_, b * h * t, s, stream);

    attention_context_batched_kernel(
        scores_cross_, vh_cross_, ctxh_cross_, b, h, t, s, head_dim_, stream);

    unpack_heads_batched_kernel(ctxh_cross_, ctx_cross_, b, t, d, h, head_dim_, stream);
    if (use_fp16) {
        float_to_half_kernel(ctx_cross_, act_fp16_, mt * d, stream);
        gemm_fp16_tensor_core(handle, mt, d, d, act_fp16_, w_o_cross_fp16_, cross_out_, stream);
    } else {
        launch_sgemm_double_buffering(mt, d, d, 1.0f, ctx_cross_, w_o_cross_, 0.0f, cross_out_, stream);
    }
    launch_add_bias(cross_out_, b_o_cross_, mt, d, stream);
    launch_residual_add(residual1_, cross_out_, residual2_, mt * d, stream);

    launch_layernorm(residual2_, ln3_gamma_, ln3_beta_, x_norm_, mt, d, cfg_.layernorm_eps, stream);
    if (use_fp16) {
        float_to_half_kernel(x_norm_, act_fp16_, mt * d, stream);
        gemm_fp16_tensor_core(handle, mt, ff, d, act_fp16_, w1_fp16_, ffn_hidden_, stream);
    } else {
        launch_sgemm_double_buffering(mt, ff, d, 1.0f, x_norm_, w1_, 0.0f, ffn_hidden_, stream);
    }
    launch_add_bias(ffn_hidden_, b1_, mt, ff, stream);
    launch_gelu(ffn_hidden_, mt * ff, stream);

    if (use_fp16) {
        float_to_half_kernel(ffn_hidden_, act_fp16_, mt * ff, stream);
        gemm_fp16_tensor_core(handle, mt, d, ff, act_fp16_, w2_fp16_, ffn_out_, stream);
    } else {
        launch_sgemm_double_buffering(mt, d, ff, 1.0f, ffn_hidden_, w2_, 0.0f, ffn_out_, stream);
    }
    launch_add_bias(ffn_out_, b2_, mt, d, stream);
    launch_residual_add(residual2_, ffn_out_, d_output, mt * d, stream);
}
