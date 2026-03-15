#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "gemm.cu"
#include "cuda_check.h"

void init_matrix(float* mat, int rows, int cols, float val) {
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = val;
    }
}

void init_matrix_random(float* mat, int rows, int cols, float scale = 1.0f) {
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX * scale;
    }
}

#define CUDA_CHECK_CUBLAS(status_) do {                             \
    if ((status_) != CUBLAS_STATUS_SUCCESS) {                       \
        fprintf(stderr, "cuBLAS error at %s:%d, code=%d\n",         \
                __FILE__, __LINE__, static_cast<int>(status_));    \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
} while(0)

bool verify_result(const float* ref, const float* test, int rows, int cols, 
                   float tol_abs = 1e-3f, float tol_rel = 1e-2f) {
    float max_err = 0.0f, max_rel_err = 0.0f;
    int err_idx = -1;
    
    for (int i = 0; i < rows * cols; ++i) {
        float err = fabsf(test[i] - ref[i]);
        float rel_err = (fabsf(ref[i]) > 1e-8f) ? err / fabsf(ref[i]) : err;
        if (err > max_err) { max_err = err; err_idx = i; }
        if (rel_err > max_rel_err) max_rel_err = rel_err;
    }
    
    bool pass = (max_err <= tol_abs) && (max_rel_err <= tol_rel);
    if (!pass) {
        int row = err_idx / cols, col = err_idx % cols;
        fprintf(stderr, "[VERIFY FAIL] max_abs_err=%.3e @ [%d,%d] (ref=%.3f, test=%.3f)\n",
                max_err, row, col, ref[err_idx], test[err_idx]);
        fprintf(stderr, "[VERIFY INFO] max_rel_err=%.3e, tol_abs=%.3e, tol_rel=%.3e\n",
                max_rel_err, tol_abs, tol_rel);
    }
    return pass;
}

class CudaTimer {
    cudaEvent_t start_, stop_;
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    
    void start() { CUDA_CHECK(cudaEventRecord(start_, 0)); }
    
    float stop() {
        CUDA_CHECK(cudaEventRecord(stop_, 0));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
};

inline double compute_gflops(int m, int n, int k, float elapsed_ms) {
    return (2.0 * m * n * k) / (elapsed_ms * 1e-3) / 1e9;
}

/**
 * @brief 执行 cuBLAS SGEMM: C = alpha * A * B + beta * C
 * @note 输入输出均为行优先（row-major）布局
 *       内部通过转置参数转换为列优先调用: C^T = B^T * A^T
 */
void cublas_sgemm_rowmajor(cublasHandle_t handle,
                           int m, int n, int k,
                           float alpha, const float* A, const float* B,
                           float beta, float* C) {
    // cuBLAS 使用列优先: C_col = A_col * B_col
    // 行优先 C_row = A_row * B_row  ⟺  C_col^T = A_col^T * B_col^T
    // ⟹ 调用: C^T = B^T * A^T, 即: cublasSgemm(..., B^T, A^T)
    const cublasOperation_t transa = CUBLAS_OP_T;  // B^T
    const cublasOperation_t transb = CUBLAS_OP_T;  // A^T
    
    // 列优先下矩阵维度: B^T: [n×k], A^T: [k×m], C^T: [n×m]
    const int lda = k;  // B^T 的 leading dimension
    const int ldb = m;  // A^T 的 leading dimension  
    const int ldc = n;  // C^T 的 leading dimension
    
    cublasStatus_t status = cublasSgemm(handle,
                                        transa, transb,
                                        n, m, k,        // 注意: m↔n 交换
                                        &alpha,
                                        B, lda,         // B^T
                                        A, ldb,         // A^T
                                        &beta,
                                        C, ldc);        // C^T
    CUDA_CHECK_CUBLAS(status);
}

#define CUDA_CHECK_CUBLAS(status_) do {                             \
    if ((status_) != CUBLAS_STATUS_SUCCESS) {                       \
        fprintf(stderr, "cuBLAS error at %s:%d, code=%d\n",         \
                __FILE__, __LINE__, static_cast<int>(status_));    \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
} while(0)

int main(int argc, char** argv) {
    int m = 8192, k = 8192, n = 8192;
    if (argc >= 4) {
        m = atoi(argv[1]); k = atoi(argv[2]); n = atoi(argv[3]);
    }
    
    printf("=== GEMM Benchmark: M=%d, K=%d, N=%d ===\n", m, k, n);
    
    size_t size_A = m * k * sizeof(float);
    size_t size_B = k * n * sizeof(float);
    size_t size_C = m * n * sizeof(float);
    
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C_custom = (float*)malloc(size_C);
    float *h_C_opt    = (float*)malloc(size_C);
    float *h_C_cublas = (float*)malloc(size_C);
    
    init_matrix(h_A, m, k, 1.0f);
    init_matrix(h_B, k, n, 2.0f);
    init_matrix(h_C_custom, m, n, 0.0f);
    init_matrix(h_C_opt,    m, n, 0.0f);
    init_matrix(h_C_cublas, m, n, 0.0f);
    
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, size_A));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size_B));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size_C));
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    
    constexpr size_t block_size = 16;
    constexpr size_t thread_m = 12;
    constexpr size_t thread_n = 8;
    
    dim3 blockDim(block_size, block_size);
    dim3 gridDim(
        (n + block_size * thread_n - 1) / (block_size * thread_n),
        (m + block_size * thread_m - 1) / (block_size * thread_m)
    );
    
    float alpha = 1.0f, beta = 0.0f;
    CudaTimer timer;
    
    printf("\n[TEST 1] Custom Kernel: sgemm_block_tiling2\n");
    CUDA_CHECK(cudaMemcpy(d_C, h_C_custom, size_C, cudaMemcpyHostToDevice));
    
    // ================ Custom Kernel ==========================
    
    // Warm-up run
    sgemm_double_buffering< block_size, thread_m, thread_n>
        <<<gridDim, blockDim>>>(m, n, k, alpha, d_A, d_B, beta, d_C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Timed run
    CUDA_CHECK(cudaMemcpy(d_C, h_C_custom, size_C, cudaMemcpyHostToDevice));
    timer.start();
    sgemm_double_buffering<block_size, thread_m, thread_n>
        <<<gridDim, blockDim>>>(m, n, k, alpha, d_A, d_B, beta, d_C);
    CUDA_CHECK(cudaGetLastError());
    float custom_ms = timer.stop();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_C_custom, d_C, size_C, cudaMemcpyDeviceToHost));
    
    double custom_gflops = compute_gflops(m, n, k, custom_ms);
    printf("  Time: %.3f ms | Performance: %.2f GFLOPS\n", 
           custom_ms, custom_gflops);
    
    printf("\n[TEST 2] cuBLAS SGEMM (row-major wrapper)\n");
    CUDA_CHECK(cudaMemcpy(d_C, h_C_cublas, size_C, cudaMemcpyHostToDevice));

    // ================ cuBLAS Kernel ==========================
    
    // Warm-up
    cublas_sgemm_rowmajor(cublas_handle, m, n, k, 
                          alpha, d_A, d_B, beta, d_C);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Timed run
    CUDA_CHECK(cudaMemcpy(d_C, h_C_cublas, size_C, cudaMemcpyHostToDevice));
    timer.start();
    cublas_sgemm_rowmajor(cublas_handle, m, n, k,
                          alpha, d_A, d_B, beta, d_C);
    float cublas_ms = timer.stop();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_C_cublas, d_C, size_C, cudaMemcpyDeviceToHost));
    
    double cublas_gflops = compute_gflops(m, n, k, cublas_ms);
    printf("  Time: %.3f ms | Performance: %.2f GFLOPS\n",
           cublas_ms, cublas_gflops);
    
    printf("\n[VERIFICATION] Custom vs cuBLAS\n");
    bool correct = verify_result(h_C_cublas, h_C_custom, m, n);

    // ================ SUMMARY ==========================

    printf("\n=== PERFORMANCE SUMMARY ===\n");
    
    printf("  Custom Kernel : %.2f GFLOPS (%.3f ms)\n", 
           custom_gflops, custom_ms);
    printf("  cuBLAS        : %.2f GFLOPS (%.3f ms)\n", 
           cublas_gflops, cublas_ms);
    double speedup = custom_ms / cublas_ms;
    printf("  Speedup       : %.2fx (cuBLAS / Custom)\n", speedup);
    double efficiency = (custom_gflops / cublas_gflops) * 100.0;
    printf("  Efficiency    : %.1f%% (Custom / cuBLAS)\n", efficiency);
    printf("  Correctness   : %s\n", correct ? "✓ PASS" : "✗ FAIL");
    
    cublasDestroy(cublas_handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C_custom); free(h_C_opt); free(h_C_cublas);
}
