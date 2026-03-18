#pragma once

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace cuda_utils {

inline void cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
    if (err == cudaSuccess) {
        return;
    }
    throw std::runtime_error(
        std::string("CUDA error at ") + file + ":" + std::to_string(line) +
        " for " + expr + ": " + cudaGetErrorString(err));
}

inline void cublas_check(cublasStatus_t status, const char* expr, const char* file, int line) {
    if (status == CUBLAS_STATUS_SUCCESS) {
        return;
    }
    throw std::runtime_error(
        std::string("cuBLAS error at ") + file + ":" + std::to_string(line) +
        " for " + expr + ", code=" + std::to_string(static_cast<int>(status)));
}

inline void alloc_device(float** ptr, size_t bytes) {
    cuda_check(cudaMalloc(ptr, bytes), "cudaMalloc", __FILE__, __LINE__);
}

inline void alloc_device_half(__half** ptr, size_t count) {
    cuda_check(cudaMalloc(ptr, count * sizeof(__half)), "cudaMalloc", __FILE__, __LINE__);
}

inline void free_device(float* ptr) {
    if (ptr != nullptr) {
        cuda_check(cudaFree(ptr), "cudaFree", __FILE__, __LINE__);
    }
}

inline void free_device_half(__half* ptr) {
    if (ptr != nullptr) {
        cuda_check(cudaFree(ptr), "cudaFree", __FILE__, __LINE__);
    }
}

}  // namespace cuda_utils

#define CUDA_CHECK(expr) ::cuda_utils::cuda_check((expr), #expr, __FILE__, __LINE__)
#define CUBLAS_CHECK(expr) ::cuda_utils::cublas_check((expr), #expr, __FILE__, __LINE__)
