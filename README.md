# CUDA Transformer Forward

This repo implements a high-performance skeleton for **Transformer Block** forward inference.

## Project Structure

- `gemm.cu`：Core matrix multiplication and host launcher. More detaile see the `./gemm/README.md`
- `src/cuda_ops.cu`：softmax/layernorm/gelu/transpose/residual/head pack-unpack etc.
- `src/transformer.cu`：TransformerBlock forward pass flow.
- `src/main.cu`：entry point

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Run

```bash
./build/transformer_bench [seq_len] [hidden] [heads] [ffn] [iters]
```

Example：

```bash
./build/transformer_bench 128 768 12 3072 100
```

## Note
The `gemm kernel` leverages __pipeline_memcpy_async for asynchronous memory copies, float4 vectorized memory access, and a double-buffering optimization for tiling cache transfers from Global Memory to Shared Memory. These techniques enable the kernel to achieve approximately 86% of the performance of the highly optimized cuBLAS library.
For a comprehensive introduction to the gemm implementation, please refer to ./gemm/README.md.
Most other operators in this codebase have implemented Warp-level reduction primitives to minimize synchronization overhead and maximize intra-warp data sharing.
TODO: Implementation of the FlashAttention kernel is pending.
