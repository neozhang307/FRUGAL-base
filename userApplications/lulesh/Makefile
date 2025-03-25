VCPKG_INCLUDE_PATH=/home/users/lingqi.zhang/workspace_x64/optimize-cuda-memory-usage-v1/build/vcpkg_installed/x64-linux/include
MEMOPT_INCLUDE_PATH=/home/users/lingqi.zhang/workspace_x64/optimize-cuda-memory-usage-v1/public

CUDA_PATH=/usr/local/cuda
NVCC=~/bin_86/llvm/bin/clang++
NVCC_FLAGS=-g -O3 -I. -I$(VCPKG_INCLUDE_PATH) -I$(MEMOPT_INCLUDE_PATH) --cuda-path=$(CUDA_PATH) --cuda-gpu-arch=sm_90 -mllvm --memred-mode=trace

all: compiler-pass

compiler-pass: lulesh.cu
	@echo "Analyzing lulesh.cu"
	$(NVCC) -c $(NVCC_FLAGS) -o lulesh.o lulesh.cu
	rm lulesh.o
	mv .memred.memory.analysis.out ../..
