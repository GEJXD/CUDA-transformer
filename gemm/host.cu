#include <stdlib.h>
#include "gemm.cu"
#include "cuda_check.h"

void init_matrix(float* mat, int rows, int cols, float val) {
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = val;
    }
}

int main() {
    int m = 8192, k = 8192, n = 8192;
    size_t size_A = m * k * sizeof(float);
    size_t size_B = k * n * sizeof(float);
    size_t size_C = m * n * sizeof(float);

    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);
    
    init_matrix(h_A, m, k, 1.0f);
    init_matrix(h_B, k, n, 2.0f);
    init_matrix(h_C, m, n, 0.0f);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, size_A));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size_B));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice));

    // constexpr size_t block_size = 16;
    // dim3 blockDim(block_size, block_size);
    // dim3 gridDim((n + blockDim.x - 1) / blockDim.x, 
    //              (m + blockDim.y - 1) / blockDim.y);
    
    // for sgemm block tiling 1D version
    // constexpr size_t block_size = 16;
    // constexpr size_t thread_m = 4;
    // dim3 blockDim(block_size, block_size);
    // dim3 gridDim((n + block_size - 1) / block_size,
    //          (m + block_size * thread_m - 1) / (block_size * thread_m));

    // for sgemm block tiling 2D version
    constexpr size_t block_size = 16;
    constexpr size_t thread_m = 12;
    constexpr size_t thread_n = 8;
    dim3 blockDim(block_size, block_size);
    dim3 gridDim(
        (n + block_size * thread_n - 1) / (block_size * thread_n),
        (m + block_size * thread_m - 1) / (block_size * thread_m)
    );

    float alpha = 1.0f, beta = 0.0f;
    // sgemm_naive<<<gridDim, blockDim>>>(m, n, k, alpha, d_A, d_B, beta, d_C);
    // sgemm_smem_tiling<float, block_size><<<gridDim, blockDim>>>(m, n, k, alpha, d_A, d_B, beta, d_C);
    // sgemm_block_tiling<float, block_size, thread_m>
    //     <<<gridDim, blockDim>>>(m, n, k, alpha, d_A, d_B, beta, d_C);
    // sgemm_block_tiling2<float, block_size, thread_m, thread_n>
    //     <<<gridDim, blockDim>>>(m, n, k, alpha, d_A, d_B, beta, d_C);
    // sgemm_block_tiling2_flatten<float, block_size, thread_m, thread_n>
    //     <<<gridDim, blockDim>>>(m, n, k, alpha, d_A, d_B, beta, d_C);
    sgemm_double_buffering< block_size, thread_m, thread_n>
        <<<gridDim, blockDim>>>(m, n, k, alpha, d_A, d_B, beta, d_C);
    // sgemm_block_tiling2_swizzled<float, block_size, thread_m, thread_n>
    //     <<<gridDim, blockDim>>>(m, n, k, alpha, d_A, d_B, beta, d_C);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

    printf("C[0][0] = %f (Expected: %f)\n", h_C[0], alpha * m * 1.0f * 2.0f);


    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}
