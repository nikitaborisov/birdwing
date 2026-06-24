# ================================================================
# Benchmark build
# ================================================================

FULL_MUL_BENCH_SRC    := bench/gpu_full_multiply_benchmark.cpp
FULL_MUL_BENCH_TARGET := build/bench_full_multiply
FULL_MUL_BENCH_OBJ    := $(OBJ_DIR)/gpu_full_multiply_benchmark.o

$(BENCH_OBJ): $(BENCH_SRC) | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(BENCH_TARGET): $(BENCH_OBJ) $(CPP_OBJS) $(CU_OBJS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_PATHS) \
		-o $@ $^ $(LIBS)

$(FULL_MUL_BENCH_OBJ): $(FULL_MUL_BENCH_SRC) | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(FULL_MUL_BENCH_TARGET): $(FULL_MUL_BENCH_OBJ) $(CPP_OBJS) $(CU_OBJS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_PATHS) \
		-o $@ $^ $(LIBS)

# ================================================================
# Convenience targets
# ================================================================

bench: $(BENCH_TARGET)
bench_full: $(FULL_MUL_BENCH_TARGET)

.PHONY: bench bench_full