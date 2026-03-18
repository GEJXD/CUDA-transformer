#pragma once

#include <cuda_runtime.h>

void launch_add_bias(float* x, const float* bias, int rows, int cols, cudaStream_t stream = nullptr);
void launch_gelu(float* x, int size, cudaStream_t stream = nullptr);
void launch_residual_add(const float* a, const float* b, float* out, int size, cudaStream_t stream = nullptr);
void launch_scale(float* x, float scale, int size, cudaStream_t stream = nullptr);
void launch_transpose(const float* in, float* out, int rows, int cols, cudaStream_t stream = nullptr);
void launch_softmax_rows(float* x, int rows, int cols, cudaStream_t stream = nullptr);
void launch_layernorm(
    const float* x,
    const float* gamma,
    const float* beta,
    float* y,
    int rows,
    int cols,
    float eps,
    cudaStream_t stream = nullptr);
// void launch_pack_heads(
//     const float* in,
//     float* out,
//     int seq_len,
//     int hidden,
//     int num_heads,
//     cudaStream_t stream = nullptr);
// void launch_unpack_heads(
//     const float* in,
//     float* out,
//     int seq_len,
//     int hidden,
//     int num_heads,
//     cudaStream_t stream = nullptr);
