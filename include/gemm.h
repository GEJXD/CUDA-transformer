#pragma once

#include <cuda_runtime.h>

void launch_sgemm_double_buffering(
    int m,
    int n,
    int k,
    float alpha,
    const float* a,
    const float* b,
    float beta,
    float* c,
    cudaStream_t stream = nullptr);
