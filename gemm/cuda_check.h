#ifndef CUDA_CHECK
#include <cstdio>
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
        } \
    } while (0);
#endif // !CUDA_CHECK
