#include "block_common.h"

#include "cuda_utils.h"
#include "cuda_ops.h"
#include "gemm.h"

#include <random>
#include <vector>

namespace {

// convert float to fp16
__global__ void float_to_half_kernel_impl(const float* in, __half* out, int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) {
        return;
    }
    out[idx] = __float2half(in[idx]);
}

// transforme [Batch, Seq_len, Hidden] to [Batch, Num_heads, Seq_len, Hidden];
__global__ void pack_heads_batched_kernel_impl(
    const float* in,
    float* out,
    int batch,
    int seq_len,
    int hidden,
    int num_heads,
    int head_dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch * seq_len * hidden;
    if (idx >= total) {
        return;
    }

    const int h = idx % hidden;
    const int token = (idx / hidden) % seq_len;
    const int b = idx / (hidden * seq_len);
    const int head = h / head_dim;
    const int dim = h % head_dim;

    const size_t out_idx =
        (((static_cast<size_t>(b) * num_heads + head) * seq_len + token) * head_dim + dim);
    out[out_idx] = in[idx];
}

// transforme [Batch, Num_heads, Seq_len, Hidden] to [Batch, Seq_len, Hidden];
__global__ void unpack_heads_batched_kernel_impl(
    const float* in,
    float* out,
    int batch,
    int seq_len,
    int hidden,
    int num_heads,
    int head_dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch * seq_len * hidden;
    if (idx >= total) {
        return;
    }

    const int h = idx % hidden;
    const int token = (idx / hidden) % seq_len;
    const int b = idx / (hidden * seq_len);
    const int head = h / head_dim;
    const int dim = h % head_dim;

    const size_t in_idx =
        (((static_cast<size_t>(b) * num_heads + head) * seq_len + token) * head_dim + dim);
    out[idx] = in[in_idx];
}

// compute Prob * V
// where Prob equals to Softmax(QK^T / sacle)
__global__ void attention_context_batched_kernel_impl(
    const float* probs,
    const float* v_heads,
    float* ctx_heads,
    int batch,
    int num_heads,
    int q_len,
    int kv_len,
    int head_dim) {
    const int dim = blockIdx.x * blockDim.x + threadIdx.x;
    const int token = blockIdx.y * blockDim.y + threadIdx.y;
    const int bh = blockIdx.z;
    const int total_bh = batch * num_heads;

    if (bh >= total_bh || token >= q_len || dim >= head_dim) {
        return;
    }

    const size_t prob_base = static_cast<size_t>(bh) * q_len * kv_len + static_cast<size_t>(token) * kv_len;
    const size_t v_base = static_cast<size_t>(bh) * kv_len * head_dim;
    const size_t out_base = static_cast<size_t>(bh) * q_len * head_dim;

    float sum = 0.0f;
    for (int j = 0; j < kv_len; ++j) {
        const float p = probs[prob_base + j];
        sum += p * v_heads[v_base + static_cast<size_t>(j) * head_dim + dim];
    }

    ctx_heads[out_base + static_cast<size_t>(token) * head_dim + dim] = sum;
}

__global__ void apply_causal_mask_kernel(float* scores, int total_bh, int q_len, int kv_len) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = total_bh * q_len * kv_len;
    if (idx >= total) {
        return;
    }

    const int col = idx % kv_len;
    const int row = (idx / kv_len) % q_len;
    if (col > row) {
        scores[idx] = -1.0e20f;
    }
}

void launch_apply_causal_mask(float* scores, int batch, int num_heads, int q_len, int kv_len, cudaStream_t stream) {
    const int total_bh = batch * num_heads;
    const int total = total_bh * q_len * kv_len;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    apply_causal_mask_kernel<<<blocks, threads, 0, stream>>>(scores, total_bh, q_len, kv_len);
}

}  // namespace

void init_param(float* d_ptr, size_t count, float low, float high) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(low, high);

    std::vector<float> host(count);
    for (size_t i = 0; i < count; ++i) {
        host[i] = dist(rng);
    }
    CUDA_CHECK(cudaMemcpy(d_ptr, host.data(), count * sizeof(float), cudaMemcpyHostToDevice));
}

void init_constant(float* d_ptr, size_t count, float value) {
    std::vector<float> host(count, value);
    CUDA_CHECK(cudaMemcpy(d_ptr, host.data(), count * sizeof(float), cudaMemcpyHostToDevice));
}

void float_to_half_kernel(const float* in, __half* out, int size, cudaStream_t stream) {
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    float_to_half_kernel_impl<<<blocks, threads, 0, stream>>>(in, out, size);
}

void gemm_fp16_tensor_core(
    cublasHandle_t handle,
    int m,
    int n,
    int k,
    const __half* a,
    const __half* b,
    float* c,
    cudaStream_t stream) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n,
        m,
        k,
        &alpha,
        b,
        CUDA_R_16F,
        n,
        a,
        CUDA_R_16F,
        k,
        &beta,
        c,
        CUDA_R_32F,
        n,
        CUBLAS_COMPUTE_32F_FAST_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void pack_heads_batched_kernel(
    const float* in,
    float* out,
    int batch,
    int seq_len,
    int hidden,
    int num_heads,
    int head_dim,
    cudaStream_t stream) {
    const int threads = 256;
    const int total = batch * seq_len * hidden;
    const int blocks = (total + threads - 1) / threads;
    pack_heads_batched_kernel_impl<<<blocks, threads, 0, stream>>>(
        in,
        out,
        batch,
        seq_len,
        hidden,
        num_heads,
        head_dim);
}

void unpack_heads_batched_kernel(
    const float* in,
    float* out,
    int batch,
    int seq_len,
    int hidden,
    int num_heads,
    int head_dim,
    cudaStream_t stream) {
    const int threads = 256;
    const int total = batch * seq_len * hidden;
    const int blocks = (total + threads - 1) / threads;
    unpack_heads_batched_kernel_impl<<<blocks, threads, 0, stream>>>(
        in,
        out,
        batch,
        seq_len,
        hidden,
        num_heads,
        head_dim);
}

void attention_context_batched_kernel(
    const float* probs,
    const float* v_heads,
    float* ctx_heads,
    int batch,
    int num_heads,
    int q_len,
    int kv_len,
    int head_dim,
    cudaStream_t stream) {
    const dim3 block(16, 16, 1);
    const dim3 grid((head_dim + block.x - 1) / block.x,
                    (q_len + block.y - 1) / block.y,
                    batch * num_heads);
    attention_context_batched_kernel_impl<<<grid, block, 0, stream>>>(
        probs,
        v_heads,
        ctx_heads,
        batch,
        num_heads,
        q_len,
        kv_len,
        head_dim);
}

void launch_attention_scores_batched_gemm(
    const float* q_heads,
    const float* k_heads,
    float* k_heads_t,
    float* scores,
    int batch,
    int num_heads,
    int q_len,
    int kv_len,
    int head_dim,
    float scale,
    bool causal_mask,
    cudaStream_t stream) {
    const int total_bh = batch * num_heads;
    for (int bh = 0; bh < total_bh; ++bh) {
        const size_t q_offset = static_cast<size_t>(bh) * q_len * head_dim;
        const size_t k_offset = static_cast<size_t>(bh) * kv_len * head_dim;
        const size_t kt_offset = static_cast<size_t>(bh) * head_dim * kv_len;
        const size_t score_offset = static_cast<size_t>(bh) * q_len * kv_len;

        const float* q_ptr = q_heads + q_offset;
        const float* k_ptr = k_heads + k_offset;
        float* kt_ptr = k_heads_t + kt_offset;
        float* score_ptr = scores + score_offset;

        launch_transpose(k_ptr, kt_ptr, kv_len, head_dim, stream);
        launch_sgemm_double_buffering(
            q_len,
            kv_len,
            head_dim,
            scale,
            q_ptr,
            kt_ptr,
            0.0f,
            score_ptr,
            stream);
    }

    if (causal_mask) {
        launch_apply_causal_mask(scores, batch, num_heads, q_len, kv_len, stream);
    }
}
