cmake_minimum_required(VERSION 3.18)
project(transformer_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_EXPORT_COMPILE_COMMANDS true)

add_executable(transformer_bench
    src/main.cu
    src/transformer.cu
    src/encoder_block.cu
    src/decoder_block.cu
    src/block_common.cu
    src/cuda_ops.cu
    src/gemm.cu
)

target_include_directories(transformer_bench PRIVATE include)

target_link_libraries(transformer_bench PRIVATE cublas)

target_compile_options(transformer_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math --expt-relaxed-constexpr>
)
