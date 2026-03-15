#include <cuda_runtime.h>
#include <cassert>
#include <device_launch_parameters.h>
#include <cuda_pipeline.h>

#include <stdexcept>

template<const size_t block_size = 16, 
         const size_t thread_m = 12, 
         const size_t thread_n = 8>
__global__ void sgemm_double_buffering(const size_t M, const size_t N, const size_t K,
        const float alpha, const float* __restrict__ A, const float* __restrict__ B,
        const float beta, float* __restrict__ C) {

    __shared__ float As[2][block_size * thread_m][block_size];
    __shared__ float Bs[2][block_size][block_size * thread_n];

    const size_t tid = threadIdx.y * block_size + threadIdx.x;
    const size_t num_threads = block_size * block_size;

    float sum[thread_m][thread_n] = {0.0f};

    const size_t BM = block_size * thread_m;
    const size_t BN = block_size * thread_n;
    const size_t BK = block_size;

    auto load_tile = [&](int stage_idx, size_t k_offset) {
        #pragma unroll
        for (size_t i = 0; i < BM * BK / (num_threads * 4); ++i) {
            size_t local_idx = tid * 4 + i * num_threads * 4;
            size_t row = local_idx / BK;
            size_t col = local_idx % BK;
            size_t g_row = blockIdx.y * BM + row;
            size_t g_col = k_offset + col;

            if (g_row < M && g_col < K) {
                __pipeline_memcpy_async(&As[stage_idx][row][col], &A[g_row * K + g_col], sizeof(float) * 4);
            }
        }
        #pragma unroll
        for (size_t i = 0; i < BK * BN / (num_threads * 4); ++i) {
            size_t local_idx = tid * 4 + i * num_threads * 4;
            size_t row = local_idx / BN;
            size_t col = local_idx % BN;
            size_t g_row = k_offset + row;
            size_t g_col = blockIdx.x * BN + col;

            if (g_row < K && g_col < N) {
                __pipeline_memcpy_async(&Bs[stage_idx][row][col], &B[g_row * N + g_col], sizeof(float) * 4);
            }
        }
        __pipeline_commit();
    };

    load_tile(0, 0);

    size_t bk = BK;
    int write_stage = 1;
    int read_stage = 0;

    for (; bk < K; bk += BK) {
        load_tile(write_stage, bk);

        __pipeline_wait_prior(1); 
        __syncthreads();

        #pragma unroll
        for (size_t k = 0; k < BK; ++k) {
            #pragma unroll
            for (size_t m = 0; m < thread_m; ++m) {
                float a_val = As[read_stage][threadIdx.y * thread_m + m][k];
                #pragma unroll
                for (size_t n = 0; n < thread_n; ++n) {
                    sum[m][n] += a_val * Bs[read_stage][k][threadIdx.x * thread_n + n]; // bank conflict here
                }
            }
        }

        read_stage = write_stage;
        write_stage = 1 - write_stage;
    }

    __pipeline_wait_prior(0);
    __syncthreads();

    #pragma unroll
    for (size_t k = 0; k < BK; ++k) {
        #pragma unroll
        for (size_t m = 0; m < thread_m; ++m) {
            float a_val = As[read_stage][threadIdx.y * thread_m + m][k];
            #pragma unroll
            for (size_t n = 0; n < thread_n; ++n) {
                sum[m][n] += a_val * Bs[read_stage][k][threadIdx.x * thread_n + n]; // bank conflict here
            }
        }
    }

    #pragma unroll
    for (size_t m = 0; m < thread_m; ++m) {
        size_t g_row = blockIdx.y * BM + threadIdx.y * thread_m + m;
        if (g_row < M) {
            #pragma unroll
            for (size_t v = 0; v < thread_n / 4; ++v) {
                size_t g_col = blockIdx.x * BN + threadIdx.x * thread_n + v * 4;
                if (g_col < N) {
                    size_t c_idx = g_row * N + g_col;
                    float4 res;
                    res.x = alpha * sum[m][v * 4 + 0];
                    res.y = alpha * sum[m][v * 4 + 1];
                    res.z = alpha * sum[m][v * 4 + 2];
                    res.w = alpha * sum[m][v * 4 + 3];

                    if (beta != 0.0f) {
                        float4 old_c = reinterpret_cast<const float4*>(&C[c_idx])[0];
                        res.x += beta * old_c.x; res.y += beta * old_c.y;
                        res.z += beta * old_c.z; res.w += beta * old_c.w;
                    }
                    reinterpret_cast<float4*>(&C[c_idx])[0] = res;
                }
            }
        }
    }
}

void launch_sgemm_double_buffering(
    int m,
    int n,
    int k,
    float alpha,
    const float* a,
    const float* b,
    float beta,
    float* c,
    cudaStream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0) {
        throw std::invalid_argument("GEMM dimensions must be positive");
    }
    if (a == nullptr || b == nullptr || c == nullptr) {
        throw std::invalid_argument("GEMM input pointers must not be null");
    }

    constexpr int kBlock = 16;
    constexpr int kThreadM = 12;
    constexpr int kThreadN = 8;
    constexpr int kTileM = kBlock * kThreadM;
    constexpr int kTileN = kBlock * kThreadN;

    const dim3 block(kBlock, kBlock, 1);
    const dim3 grid((n + kTileN - 1) / kTileN, (m + kTileM - 1) / kTileM, 1);

    sgemm_double_buffering<kBlock, kThreadM, kThreadN><<<grid, block, 0, stream>>>(
        static_cast<size_t>(m),
        static_cast<size_t>(n),
        static_cast<size_t>(k),
        alpha,
        a,
        b,
        beta,
        c);
}
