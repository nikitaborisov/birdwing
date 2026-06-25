# ================================================================
# Vendored GPU-NTT (git submodule)
# ================================================================

GPU_NTT_DIR   := $(ROOT_DIR)/GPU-NTT
GPU_NTT_BUILD := $(GPU_NTT_DIR)/build
GPU_NTT_LIB   := $(GPU_NTT_BUILD)/src/libntt-1.0.a
GPU_NTT_INC   := $(GPU_NTT_DIR)/src/include

CMAKE_CUDA_ARCHITECTURES ?= native

$(GPU_NTT_DIR)/CMakeLists.txt:
	@if [ ! -f "$@" ]; then \
		echo "Initializing GPU-NTT submodule..."; \
		git submodule update --init --recursive GPU-NTT; \
	fi

$(GPU_NTT_LIB): $(GPU_NTT_DIR)/CMakeLists.txt
	cmake -S $(GPU_NTT_DIR) -B $(GPU_NTT_BUILD) \
		-DCMAKE_BUILD_TYPE=Release \
		-DGPUNTT_BUILD_EXAMPLES=OFF \
		-DGPUNTT_BUILD_BENCHMARKS=OFF \
		-DCMAKE_CUDA_ARCHITECTURES=$(CMAKE_CUDA_ARCHITECTURES)
	cmake --build $(GPU_NTT_BUILD) --parallel

gpu-ntt: $(GPU_NTT_LIB)

.PHONY: gpu-ntt
