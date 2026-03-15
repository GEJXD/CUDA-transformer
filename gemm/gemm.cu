#include <cuda_runtime.h>
#include <cassert>
#include <device_launch_parameters.h>
#include <cuda_pipeline.h>

template<typename T>
__global__ void sgemm_naive(const size_t M, const size_t N, const size_t K, 
        const T alpha, const T *A, const T *B,
        const T beta, T *C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < N && y < M) {
        T sum = T(0);
        for (int k = 0; k < K;k ++) {
            sum += A[y * K + k] * B[k * N + x];
        }
        C[y * N + x] = alpha * sum + beta * C[y * N + x];
    }
}

template<typename T, size_t block_size = 32>
__global__ void sgemm_smem_tiling(const size_t M, const size_t N, const size_t K, 
        const T alpha, const T *A, const T *B,
        const T beta, T *C) {
    __shared__ T As[block_size][block_size];
    __shared__ T Bs[block_size][block_size];

    size_t row = blockIdx.y * block_size + threadIdx.y;
    size_t col = blockIdx.x * block_size + threadIdx.x;

    T sum{0};
    for (int bk = 0;bk < K;bk += block_size) {
        size_t a_row = row;
        size_t a_col = bk + threadIdx.x;
        As[threadIdx.y][threadIdx.x] = A[a_row * K + a_col];

        size_t b_row = bk + threadIdx.y;
        size_t b_col = col;
        Bs[threadIdx.y][threadIdx.x] = B[b_row * N + b_col];
        __syncthreads();

        for (size_t k = 0;k < block_size;k ++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
    }

    if (row < M && col < N) {
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

template<typename T, const size_t block_size = 32, const size_t thread_m = 4>
__global__ void sgemm_block_tiling(const size_t M, const size_t N, const size_t K,
        const T alpha, const T * __restrict__ A, const T * __restrict__ B,
        const T beta, T * __restrict__ C) {
    assert(blockDim.x == block_size);
    assert(blockDim.y == block_size);

    __shared__ T As[block_size * thread_m][block_size];
    __shared__ T Bs[block_size][block_size];

    size_t base_row = (blockIdx.y * block_size + threadIdx.y) * thread_m;
    size_t col = blockIdx.x * block_size + threadIdx.x;

    T sum[thread_m];
    for (size_t i = 0;i < thread_m;i ++) {
        sum[i] = T(0);
    }

    for (size_t bk = 0;bk < K;bk += block_size) {
        for (size_t m = 0;m < thread_m;m ++) {
            size_t a_row = base_row + m;
            size_t a_col = bk + threadIdx.x;
            if (a_row < M && a_col < K) {
                As[threadIdx.y * thread_m + m][threadIdx.x] = A[a_row * K + a_col];
            } else {
                As[threadIdx.y * thread_m + m][threadIdx.x] = T(0);
            }
        }

        size_t b_row = bk + threadIdx.y;
        size_t b_col = col;
        if (b_row < K && b_col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + b_col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = T(0);
        }

        __syncthreads();

        for (size_t k = 0;k < block_size;k ++) {
            T b_val = Bs[k][threadIdx.x];
            for (size_t m = 0;m < thread_m;m ++) {
                sum[m] += As[threadIdx.y * thread_m + m][k] * b_val;
            }
        }
        __syncthreads();
    }

    for (size_t m = 0;m < thread_m;m ++) {
        size_t row = base_row + m;
        if (row < M && col < N) {
            T c_val = (beta == T(0)) ? T(0) : beta * C[row * N + col];
            C[row * N + col] = alpha * sum[m] + c_val;
        }
    }
}

template<typename T, 
         const size_t block_size = 16, 
         const size_t thread_m = 12, 
         const size_t thread_n = 8>
__global__ void sgemm_block_tiling2(const size_t M, const size_t N, const size_t K,
        const T alpha, const T * __restrict__ A, const T * __restrict__ B,
        const T beta, T * __restrict__ C) {
    
    assert(blockDim.x == block_size && blockDim.y == block_size);

    __shared__ T As[block_size * thread_m][block_size];
    __shared__ T Bs[block_size][block_size * thread_n];

    const size_t base_row = (blockIdx.y * block_size + threadIdx.y) * thread_m;
    const size_t base_col = (blockIdx.x * block_size + threadIdx.x) * thread_n;

    T sum[thread_m][thread_n];
    #pragma unroll
    for (size_t m = 0; m < thread_m; ++m) {
        #pragma unroll
        for (size_t n = 0; n < thread_n; ++n) {
            sum[m][n] = T(0);
        }
    }

    #pragma unroll
    for (size_t bk = 0; bk < K; bk += block_size) {
        #pragma unroll
        for (size_t m = 0; m < thread_m; ++m) {
            const size_t a_row = base_row + m;
            const size_t a_col = bk + threadIdx.x;
            const size_t a_idx = a_row * K + a_col;
            As[threadIdx.y * thread_m + m][threadIdx.x] = 
                (a_row < M && a_col < K) ? A[a_idx] : T(0);
        }

        #pragma unroll
        for (size_t n = 0; n < thread_n; ++n) {
            const size_t b_row = bk + threadIdx.y;
            // adjust threads have 4 bytes stride, memory uncoalesced
            const size_t b_col = base_col + n;
            const size_t b_idx = b_row * N + b_col;
            Bs[threadIdx.y][threadIdx.x * thread_n + n] = 
                (b_row < K && b_col < N) ? B[b_idx] : T(0);
        }
        __syncthreads();

        #pragma unroll
        for (size_t k = 0; k < block_size; ++k) {
            #pragma unroll
            for (size_t m = 0; m < thread_m; ++m) {
                const T a_val = As[threadIdx.y * thread_m + m][k];
                #pragma unroll
                for (size_t n = 0; n < thread_n; ++n) {
                    const T b_val = Bs[k][threadIdx.x * thread_n + n];
                    sum[m][n] += a_val * b_val;
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (size_t m = 0; m < thread_m; ++m) {
        const size_t row = base_row + m;
        if (row < M) {
            #pragma unroll
            for (size_t n = 0; n < thread_n; ++n) {
                const size_t col = base_col + n;
                if (col < N) {
                    const size_t c_idx = row * N + col;
                    const T c_old = (beta == T(0)) ? T(0) : beta * C[c_idx];
                    C[c_idx] = alpha * sum[m][n] + c_old;
                }
            }
        }
    }
}

template<typename T, 
         const size_t block_size = 16, 
         const size_t thread_m = 12, 
         const size_t thread_n = 8>
__global__ void sgemm_block_tiling2_flatten(const size_t M, const size_t N, const size_t K,
        const T alpha, const T * __restrict__ A, const T * __restrict__ B,
        const T beta, T * __restrict__ C) {
    __shared__ T As[block_size * thread_m][block_size + 1];
    __shared__ T Bs[block_size][block_size * thread_n + 1];

    const size_t base_row = (blockIdx.y * block_size + threadIdx.y) * thread_m;
    const size_t base_col = (blockIdx.x * block_size + threadIdx.x) * thread_n;

    T sum[thread_m][thread_n];
    #pragma unroll
    for (size_t m = 0; m < thread_m; ++m) {
        #pragma unroll
        for (size_t n = 0; n < thread_n; ++n) {
            sum[m][n] = T(0);
        }
    }

    const size_t BM = block_size * thread_m;
    const size_t BN = block_size * thread_n;
    const size_t BK = block_size;

    const size_t block_base_row = blockIdx.y * BM;
    const size_t block_base_col = blockIdx.x * BN;

    #pragma unroll
    for (size_t bk = 0; bk < K; bk += block_size) {
        // thread 0 ~ (block_size - 1) store the first row;
        const size_t tid = threadIdx.y * block_size + threadIdx.x;
        const size_t num_threads = blockDim.x * blockDim.y;
        
        const size_t a_tile_row = tid / block_size;
        const size_t a_tile_col = tid % block_size;
        const size_t a_stride = num_threads / block_size;
        #pragma unroll
        // stride-loop
        for (size_t i = 0;i < BM;i += a_stride) {
            const size_t a_row_global = block_base_row + a_tile_row + i;
            const size_t a_col_global = bk + a_tile_col;
            const size_t a_idx = a_row_global * K + a_col_global;

            As[a_tile_row + i][a_tile_col] = 
                (a_row_global < M && a_col_global < K) ? A[a_idx] : T(0);
        }

        const size_t b_tile_row = tid / BN;
        const size_t b_tile_col = tid % BN;
        const size_t b_stride = num_threads / BN;

        #pragma unroll
        for (size_t i = 0;i < BK;i += b_stride) {
            const size_t b_row_global = bk + b_tile_row + i;
            const size_t b_col_global = block_base_col + b_tile_col;
            const size_t b_idx = b_row_global * N + b_col_global;

            Bs[b_tile_row + i][b_tile_col] = 
                (b_row_global < K && b_col_global < N) ? B[b_idx] : T(0);
        }
        __syncthreads();

        #pragma unroll
        for (size_t k = 0; k < block_size; ++k) {
            #pragma unroll
            for (size_t m = 0; m < thread_m; ++m) {
                const T a_val = As[threadIdx.y * thread_m + m][k];
                #pragma unroll
                for (size_t n = 0; n < thread_n; ++n) {
                    const T b_val = Bs[k][threadIdx.x * thread_n + n];
                    sum[m][n] += a_val * b_val;
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (size_t m = 0; m < thread_m; ++m) {
        const size_t row = base_row + m;
        if (row < M) {
            #pragma unroll
            for (size_t n = 0; n < thread_n; ++n) {
                const size_t col = base_col + n;
                if (col < N) {
                    const size_t c_idx = row * N + col;
                    const T c_old = (beta == T(0)) ? T(0) : beta * C[c_idx];
                    C[c_idx] = alpha * sum[m][n] + c_old;
                }
            }
        }
    }
}

template<const size_t block_size = 16,
         const size_t thread_m = 12,
         const size_t thread_n = 8>
__global__ void sgemm_vectorized(const size_t M, const size_t N, const size_t K,
        const float alpha, const float* __restrict__ A, const float* __restrict__ B,
        const float beta, float* __restrict__ C) {
    
    __shared__ float As[block_size * thread_m][block_size + 1];
    __shared__ float Bs[block_size][block_size * thread_n];

    const size_t tid = threadIdx.y * block_size + threadIdx.x;
    const size_t num_threads = block_size * block_size;

    float sum[thread_m][thread_n] = {0.0f};

    const size_t BM = block_size * thread_m;
    const size_t BN = block_size * thread_n;
    const size_t BK = block_size;

    for (size_t bk = 0; bk < K; bk += BK) {
        // --- 1. 向量化搬运 A (BM x BK = 128 x 16) ---
        // 这里的目标是让每个线程通过 float4 搬运
        // 总元素 2048, 256个线程 -> 每个线程搬运 8 个元素 (2个 float4)
        #pragma unroll
        for (size_t i = 0; i < BM * BK / (num_threads * 4); ++i) {
            size_t local_idx = tid * 4 + i * num_threads * 4;
            size_t row = local_idx / BK;
            size_t col = local_idx % BK;
            
            size_t g_row = blockIdx.y * BM + row;
            size_t g_col = bk + col;

            float4 tmp = reinterpret_cast<const float4*>(&A[g_row * K + g_col])[0];
            
            As[row][col]     = tmp.x;
            As[row][col + 1] = tmp.y;
            As[row][col + 2] = tmp.z;
            As[row][col + 3] = tmp.w;
        }

        // --- 2. 向量化搬运 B (BK x BN = 16 x 128) ---
        // 总元素 2048, 每个线程搬运 2个 float4
        #pragma unroll
        for (size_t i = 0; i < BK * BN / (num_threads * 4); ++i) {
            size_t local_idx = tid * 4 + i * num_threads * 4;
            size_t row = local_idx / BN;
            size_t col = local_idx % BN;

            size_t g_row = bk + row;
            size_t g_col = blockIdx.x * BN + col;

            float4 tmp = reinterpret_cast<const float4*>(&B[g_row * N + g_col])[0];
            
            reinterpret_cast<float4*>(&Bs[row][col])[0] = tmp;
        }

        __syncthreads();

        #pragma unroll
        for (size_t k = 0; k < BK; ++k) {
            #pragma unroll
            for (size_t m = 0; m < thread_m; ++m) {
                float a_val = As[threadIdx.y * thread_m + m][k];
                #pragma unroll
                for (size_t n = 0; n < thread_n; ++n) {
                    sum[m][n] += a_val * Bs[k][threadIdx.x * thread_n + n];
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (size_t m = 0; m < thread_m; ++m) {
        const size_t g_row = blockIdx.y * BM + threadIdx.y * thread_m + m;
    
        if (g_row < M) {
            #pragma unroll
            for (size_t v = 0; v < thread_n / 4; ++v) {
                const size_t g_col = blockIdx.x * BN + threadIdx.x * thread_n + v * 4;
            
                if (g_col < N) {
                    const size_t c_idx = g_row * N + g_col;                
                    float4 res;
                    res.x = alpha * sum[m][v * 4 + 0];
                    res.y = alpha * sum[m][v * 4 + 1];
                    res.z = alpha * sum[m][v * 4 + 2];
                    res.w = alpha * sum[m][v * 4 + 3];

                    if (beta != 0.0f) {
                        float4 old_c = reinterpret_cast<const float4*>(&C[c_idx])[0];
                        res.x += beta * old_c.x;
                        res.y += beta * old_c.y;
                        res.z += beta * old_c.z;
                        res.w += beta * old_c.w;
                    }

                    reinterpret_cast<float4*>(&C[c_idx])[0] = res;
                }
            }
        }
    }
}

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

// useless kernel
template<size_t SWIZZLE_BITS>
__device__ __forceinline__ size_t apply_swizzle(size_t row, size_t col) {
    constexpr size_t SWIZZLE_MASK = (1 << SWIZZLE_BITS) - 1;
    return col ^ (row & SWIZZLE_MASK);
}

constexpr size_t DEFAULT_SWIZZLE_BITS = 3;
__device__ __forceinline__ size_t swizzle_addr(size_t row, size_t col) {
    return apply_swizzle<DEFAULT_SWIZZLE_BITS>(row, col);
}

template<typename T, 
         const size_t block_size = 32, 
         const size_t thread_m = 2, 
         const size_t thread_n = 2,
         const size_t SWIZZLE_BITS = 3>
__global__ void sgemm_block_tiling2_swizzled(
        const size_t M, const size_t N, const size_t K,
        const T alpha, const T * __restrict__ A, const T * __restrict__ B,
        const T beta, T * __restrict__ C) {
    
    assert(blockDim.x == block_size && blockDim.y == block_size);

    constexpr size_t SWIZZLE_MASK = (1 << SWIZZLE_BITS) - 1;
    constexpr size_t SHARED_MEM_PAD = 8;
    
    __shared__ T As[block_size * thread_m][block_size + SHARED_MEM_PAD];
    __shared__ T Bs[block_size][block_size * thread_n + SHARED_MEM_PAD];

    const size_t base_row = (blockIdx.y * block_size + threadIdx.y) * thread_m;
    const size_t base_col = (blockIdx.x * block_size + threadIdx.x) * thread_n;

    T sum[thread_m][thread_n] = {};

    for (size_t bk = 0; bk < K; bk += block_size) {
        for (size_t m = 0; m < thread_m; ++m) {
            const size_t a_row = base_row + m;
            const size_t a_col = bk + threadIdx.x;
            const size_t shared_row = threadIdx.y * thread_m + m;
            const size_t shared_col = threadIdx.x;
            const size_t a_idx = a_row * K + a_col;
            
            const size_t col_swizzled = shared_col ^ (shared_row & SWIZZLE_MASK);
            As[shared_row][col_swizzled] = 
                (a_row < M && a_col < K) ? A[a_idx] : T(0);
        }

        for (size_t n = 0; n < thread_n; ++n) {
            const size_t b_row = bk + threadIdx.y;
            const size_t b_col = base_col + n;
            const size_t shared_row = threadIdx.y;
            const size_t shared_col = threadIdx.x * thread_n + n;
            const size_t b_idx = b_row * N + b_col;
            
            const size_t col_swizzled = shared_col ^ (shared_row & SWIZZLE_MASK);
            Bs[shared_row][col_swizzled] = 
                (b_row < K && b_col < N) ? B[b_idx] : T(0);
        }

        __syncthreads();

        for (size_t k = 0; k < block_size; ++k) {
            for (size_t m = 0; m < thread_m; ++m) {
                const size_t shared_row_a = threadIdx.y * thread_m + m;
                const size_t col_swizzled_a = k ^ (shared_row_a & SWIZZLE_MASK);
                const T a_val = As[shared_row_a][col_swizzled_a];
                
                for (size_t n = 0; n < thread_n; ++n) {
                    const size_t shared_col_b = threadIdx.x * thread_n + n;
                    const size_t col_swizzled_b = shared_col_b ^ (k & SWIZZLE_MASK);
                    const T b_val = Bs[k][col_swizzled_b];
                    sum[m][n] += a_val * b_val;
                }
            }
        }
        
        __syncthreads();
    }

    for (size_t m = 0; m < thread_m; ++m) {
        const size_t row = base_row + m;
        if (row < M) {
            for (size_t n = 0; n < thread_n; ++n) {
                const size_t col = base_col + n;
                if (col < N) {
                    const size_t c_idx = row * N + col;
                    const T c_old = (beta == T(0)) ? T(0) : beta * C[c_idx];
                    C[c_idx] = alpha * sum[m][n] + c_old;
                }
            }
        }
    }
}
