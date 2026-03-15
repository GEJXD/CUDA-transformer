# Compiler & flags
NVCC = nvcc
CXX = g++
NVCC_FLAGS = -O3 -arch=sm_86 --use_fast_math
LDFLAGS = -lcublas

BENCH_SRC = main_cublas_benchmark.cu
HOST_SRC = host.cu

BENCH_OBJ = $(BENCH_SRC:.cu=.o)
HOST_OBJ = $(HOST_SRC:.cu=.o)

# Targets
BENCH_TARGET = gemm_benchmark
HOST_TARGET = gemm_host

# Header dependencies (shared by both targets)
HEADERS = gemm.cu cuda_check.h

.PHONY: all clean run bench host run-large run-small

all: $(BENCH_TARGET)

bench: $(BENCH_TARGET)

host: $(HOST_TARGET)

$(BENCH_TARGET): $(BENCH_OBJ)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS)

$(HOST_TARGET): $(HOST_OBJ)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cu $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

run: $(BENCH_TARGET)
	./$(BENCH_TARGET)

run-large: $(BENCH_TARGET)
	./$(BENCH_TARGET) 2048 2048 2048

run-small: $(BENCH_TARGET)
	./$(BENCH_TARGET) 256 256 256

clean:
	rm -f $(BENCH_OBJ) $(HOST_OBJ) $(BENCH_TARGET) $(HOST_TARGET) *.o
