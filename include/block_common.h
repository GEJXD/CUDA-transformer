#pragma once

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

void init_param(float* d_ptr, size_t count, float low, float high);
void init_constant(float* d_ptr, size_t count, float value);

void float_to_half_kernel(const float* in, __half* out, int size, cudaStream_t stream);

void gemm_fp16_tensor_core(
    cublasHandle_t handle,
    int m,
    int n,
    int k,
    const __half* a,
    const __half* b,
    float* c,
    cudaStream_t stream);

void pack_heads_batched_kernel(
    const float* in,
    float* out,
    int batch,
    int seq_len,
    int hidden,
    int num_heads,
    int head_dim,
    cudaStream_t stream);

void unpack_heads_batched_kernel(
    const float* in,
    float* out,
    int batch,
    int seq_len,
    int hidden,
    int num_heads,
    int head_dim,
    cudaStream_t stream);

void attention_context_batched_kernel(
    const float* probs,
    const float* v_heads,
    float* ctx_heads,
    int batch,
    int num_heads,
    int q_len,
    int kv_len,
    int head_dim,
    cudaStream_t stream);

void launch_attention_scores_batched_gemm(
    const float* q_heads,
    const float* k_heads,
    float* k_heads_t,
    float* scores,
    int batch,
    int num_heads,
    int q_len,
    int kv_len,
    int head_dim,
    float scale,
    bool causal_mask = false,
    cudaStream_t stream = nullptr);
