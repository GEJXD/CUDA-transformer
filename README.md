# CUDA Seq2Seq Transformer Forward

This repo implements a CUDA forward inference skeleton for full Transformer encoder-decoder architecture:
- encoder stack (self-attention + FFN)
- decoder stack (masked self-attention + encoder-decoder cross-attention + FFN)

## Project Structure

- `src/gemm.cu`: core matrix multiplication kernel and launcher
- `src/cuda_ops.cu`: softmax/layernorm/gelu/transpose/residual/head pack-unpack kernels
- `src/encoder_block.cu`: reusable EncoderBlock used by encoder layers
- `src/transformer.cu`: encoder-decoder stack with masked self-attention + cross-attention
- `src/main.cu`: seq2seq benchmark entry

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Run

### Seq2Seq CLI

There is only one benchmark entry command:

```bash
./build/transformer_bench [precision] [batch] [src_seq_len] [tgt_seq_len] [hidden] [heads] [ffn] [encoder_layers] [decoder_layers] [iters]
```

Or with named options:

```bash
./build/transformer_bench --precision [fp32|fp16] --batch [N] --src-seq-len [N] --tgt-seq-len [N] --hidden [N] --heads [N] --ffn [N] --encoder-layers [N] --decoder-layers [N] --iters [N]
```

```bash
./build/transformer_bench --help
```

`precision` supports:
- `fp32`: FP32 baseline
- `fp16`: mixed precision path (FP16 GEMM on Tensor Cores, FP32 accumulation)

### Minimal Reproducible Benchmark

| Case                    | Command                                                                                                                                                                          |
|-------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Seq2Seq baseline        | `./build/transformer_bench fp32 4 128 128 768 12 3072 6 6 20`                                                                                                                    |
| Seq2Seq mixed precision | `./build/transformer_bench fp16 4 256 128 768 12 3072 12 12 20`                                                                                                                  |
| Seq2Seq named options   | `./build/transformer_bench --precision fp16 --batch 4 --src-seq-len 256 --tgt-seq-len 128 --hidden 768 --heads 12 --ffn 3072 --encoder-layers 12 --decoder-layers 12 --iters 20` |

Output includes:
- average forward latency
- throughput (tokens/s)
- theoretical FLOPs per iteration
- achieved throughput (TFLOPS)
- per-layer average latency

## Notes

- The GEMM kernel uses `__pipeline_memcpy_async`, float4 vectorized memory access, and double buffering for global-to-shared tiling.
- For GEMM details, see `gemm/README.md`.
- Other operators use warp-level reductions to reduce synchronization overhead.
- TODO: FlashAttention kernel implementation.
