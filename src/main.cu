#include "transformer.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace {

#define CUDA_CHECK(expr)                                                                 \
    do {                                                                                 \
        cudaError_t err__ = (expr);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            throw std::runtime_error(cudaGetErrorString(err__));                         \
        }                                                                                \
    } while (0)

}  // namespace

int main(int argc, char** argv) {
    try {
        int seq_len = 128;
        int hidden = 768;
        int heads = 12;
        int ffn = 3072;
        int iters = 100;

        if (argc > 1) seq_len = std::atoi(argv[1]);
        if (argc > 2) hidden = std::atoi(argv[2]);
        if (argc > 3) heads = std::atoi(argv[3]);
        if (argc > 4) ffn = std::atoi(argv[4]);
        if (argc > 5) iters = std::atoi(argv[5]);

        TransformerConfig cfg{seq_len, hidden, heads, ffn, 1e-5f};
        TransformerBlock block(cfg);

        const size_t bytes = static_cast<size_t>(seq_len) * hidden * sizeof(float);
        std::vector<float> h_input(static_cast<size_t>(seq_len) * hidden, 0.01f);
        std::vector<float> h_output(static_cast<size_t>(seq_len) * hidden, 0.0f);

        float* d_input = nullptr;
        float* d_output = nullptr;
        CUDA_CHECK(cudaMalloc(&d_input, bytes));
        CUDA_CHECK(cudaMalloc(&d_output, bytes));
        CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));

        for (int i = 0; i < 10; ++i) {
            block.forward(d_input, d_output);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; ++i) {
            block.forward(d_input, d_output);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

        CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, bytes, cudaMemcpyDeviceToHost));

        printf("Transformer block forward benchmark\n");
        printf("seq=%d hidden=%d heads=%d ffn=%d iters=%d\n", seq_len, hidden, heads, ffn, iters);
        printf("avg latency: %.3f ms\n", ms / static_cast<float>(iters));
        printf("throughput: %.2f tokens/s\n", (seq_len * 1000.0f * iters) / ms);
        printf("output checksum sample: %.6f %.6f %.6f\n", h_output[0], h_output[1], h_output[2]);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));

        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
