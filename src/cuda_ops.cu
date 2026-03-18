#include "cuda_ops.h"

#include <cassert>
#include <cmath>

namespace {

constexpr int kThreads = 256;

__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__global__ void add_bias_kernel(float* x, const float* bias, int rows, int cols) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * cols;
    if (idx >= total) {
        return;
    }
    const int col = idx % cols;
    x[idx] += bias[col];
}

// gelu activate function:
// GELU(x) = 0.5x(1 + tanh(sqrt(2 / pi)(x + 0.044715x^3)))
// sqrt(2 / pi) = 0.7978845608028654;
__global__ void gelu_kernel(float* x, int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) {
        return;
    }
    const float v = x[idx];
    const float c0 = 0.7978845608f;
    const float c1 = 0.044715f;
    x[idx] = 0.5f * v * (1.0f + tanhf(c0 * (v + c1 * v * v * v)));
}

// Res(x) = F(x) + x
// also can be used for vector add
__global__ void residual_add_kernel(const float* a, const float* b, float* out, int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) {
        return;
    }
    out[idx] = a[idx] + b[idx];
}

__global__ void scale_kernel(float* x, float scale, int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) {
        return;
    }
    x[idx] *= scale;
}

__global__ void transpose_kernel(const float* in, float* out, int rows, int cols) {
    __shared__ float tile[32][33];

    const int x = blockIdx.x * 32 + threadIdx.x;
    const int y = blockIdx.y * 32 + threadIdx.y;

    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = in[y * cols + x];
    }
    __syncthreads();

    const int tx = blockIdx.y * 32 + threadIdx.x;
    const int ty = blockIdx.x * 32 + threadIdx.y;
    if (tx < rows && ty < cols) {
        out[ty * rows + tx] = tile[threadIdx.x][threadIdx.y];
    }
}

// safe softmax:
// Softmax(x) = e^xi - max(x) / Sum(x - max(x));
__global__ void softmax_rows_kernel(float* x, int rows, int cols) {
    __shared__ float shared_max[32];
    __shared__ float shared_sum[32];

    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warp_count = blockDim.x >> 5;

    float local_max = -INFINITY;
    for (int col = tid; col < cols; col += blockDim.x) {
        local_max = fmaxf(local_max, x[row * cols + col]);
    }

    float warp_max = warp_reduce_max(local_max);
    if (lane == 0) {
        shared_max[warp] = warp_max;
    }
    __syncthreads();

    float block_max = -INFINITY;
    if (warp == 0) {
        block_max = (lane < warp_count) ? shared_max[lane] : -INFINITY;
        block_max = warp_reduce_max(block_max);
        if (lane == 0) {
            shared_max[0] = block_max;
        }
    }
    __syncthreads();
    block_max = shared_max[0];

    float local_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        const float v = __expf(x[row * cols + col] - block_max);
        x[row * cols + col] = v;
        local_sum += v;
    }

    float warp_sum = warp_reduce_sum(local_sum);
    if (lane == 0) {
        shared_sum[warp] = warp_sum;
    }
    __syncthreads();

    float block_sum = 0.0f;
    if (warp == 0) {
        block_sum = (lane < warp_count) ? shared_sum[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (lane == 0) {
            shared_sum[0] = block_sum;
        }
    }
    __syncthreads();
    // avoid block_sum is zero
    block_sum = shared_sum[0] + 1e-6f;

    for (int col = tid; col < cols; col += blockDim.x) {
        x[row * cols + col] /= block_sum;
    }
}

// LayerNorm with Affine Transformer
// y = x * gamma + beta
__global__ void layernorm_kernel(
    const float* x,
    const float* gamma,
    const float* beta,
    float* y,
    int rows,
    int cols,
    float eps) {
    __shared__ float shared_sum[32];
    __shared__ float shared_sq_sum[32];

    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warp_count = blockDim.x >> 5;

    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        const float v = x[row * cols + col];
        local_sum += v;
        local_sq_sum += v * v;
    }

    float warp_sum = warp_reduce_sum(local_sum);
    float warp_sq_sum = warp_reduce_sum(local_sq_sum);

    if (lane == 0) {
        shared_sum[warp] = warp_sum;
        shared_sq_sum[warp] = warp_sq_sum;
    }
    __syncthreads();

    float total_sum = 0.0f;
    float total_sq_sum = 0.0f;
    if (warp == 0) {
        total_sum = (lane < warp_count) ? shared_sum[lane] : 0.0f;
        total_sq_sum = (lane < warp_count) ? shared_sq_sum[lane] : 0.0f;
        total_sum = warp_reduce_sum(total_sum);
        total_sq_sum = warp_reduce_sum(total_sq_sum);
        if (lane == 0) {
            shared_sum[0] = total_sum;
            shared_sq_sum[0] = total_sq_sum;
        }
    }
    __syncthreads();

    const float mean = shared_sum[0] / static_cast<float>(cols);
    const float ex2 = shared_sq_sum[0] / static_cast<float>(cols);
    const float var = fmaxf(ex2 - mean * mean, 0.0f);
    const float inv_std = rsqrtf(var + eps);

    for (int col = tid; col < cols; col += blockDim.x) {
        const float norm = (x[row * cols + col] - mean) * inv_std;
        y[row * cols + col] = norm * gamma[col] + beta[col];
    }
}

// __global__ void pack_heads_kernel(
//     const float* in,
//     float* out,
//     int seq_len,
//     int hidden,
//     int num_heads,
//     int head_dim) {
//     const int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     const int total = seq_len * hidden;
//     if (idx >= total) {
//         return;
//     }
//
//     const int token = idx / hidden;
//     const int h = idx % hidden;
//     const int head = h / head_dim;
//     const int dim = h % head_dim;
//
//     out[(head * seq_len + token) * head_dim + dim] = in[idx];
// }
//
// __global__ void unpack_heads_kernel(
//     const float* in,
//     float* out,
//     int seq_len,
//     int hidden,
//     int num_heads,
//     int head_dim) {
//     const int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     const int total = seq_len * hidden;
//     if (idx >= total) {
//         return;
//     }
//
//     const int token = idx / hidden;
//     const int h = idx % hidden;
//     const int head = h / head_dim;
//     const int dim = h % head_dim;
//
//     out[idx] = in[(head * seq_len + token) * head_dim + dim];
// }

}  // namespace

void launch_add_bias(float* x, const float* bias, int rows, int cols, cudaStream_t stream) {
    const int total = rows * cols;
    const int blocks = (total + kThreads - 1) / kThreads;
    add_bias_kernel<<<blocks, kThreads, 0, stream>>>(x, bias, rows, cols);
}

void launch_gelu(float* x, int size, cudaStream_t stream) {
    const int blocks = (size + kThreads - 1) / kThreads;
    gelu_kernel<<<blocks, kThreads, 0, stream>>>(x, size);
}

void launch_residual_add(const float* a, const float* b, float* out, int size, cudaStream_t stream) {
    const int blocks = (size + kThreads - 1) / kThreads;
    residual_add_kernel<<<blocks, kThreads, 0, stream>>>(a, b, out, size);
}

void launch_scale(float* x, float scale, int size, cudaStream_t stream) {
    const int blocks = (size + kThreads - 1) / kThreads;
    scale_kernel<<<blocks, kThreads, 0, stream>>>(x, scale, size);
}

void launch_transpose(const float* in, float* out, int rows, int cols, cudaStream_t stream) {
    const dim3 block(32, 8, 1);
    const dim3 grid((cols + 31) / 32, (rows + 31) / 32, 1);
    transpose_kernel<<<grid, block, 0, stream>>>(in, out, rows, cols);
}

void launch_softmax_rows(float* x, int rows, int cols, cudaStream_t stream) {
    softmax_rows_kernel<<<rows, kThreads, 0, stream>>>(x, rows, cols);
}

void launch_layernorm(
    const float* x,
    const float* gamma,
    const float* beta,
    float* y,
    int rows,
    int cols,
    float eps,
    cudaStream_t stream) {
    layernorm_kernel<<<rows, kThreads, 0, stream>>>(x, gamma, beta, y, rows, cols, eps);
}

// void launch_pack_heads(const float* in, float* out, int seq_len, int hidden, int num_heads, cudaStream_t stream) {
//     assert(hidden % num_heads == 0);
//     const int total = seq_len * hidden;
//     const int blocks = (total + kThreads - 1) / kThreads;
//     const int head_dim = hidden / num_heads;
//     pack_heads_kernel<<<blocks, kThreads, 0, stream>>>(in, out, seq_len, hidden, num_heads, head_dim);
// }
//
// void launch_unpack_heads(const float* in, float* out, int seq_len, int hidden, int num_heads, cudaStream_t stream) {
//     assert(hidden % num_heads == 0);
//     const int total = seq_len * hidden;
//     const int blocks = (total + kThreads - 1) / kThreads;
//     const int head_dim = hidden / num_heads;
//     unpack_heads_kernel<<<blocks, kThreads, 0, stream>>>(in, out, seq_len, hidden, num_heads, head_dim);
// }
