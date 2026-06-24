# ================================================================
# Benchmark build
# ================================================================

FULL_MUL_BENCH_SRC := bench/gpu_full_multiply_benchmark.cpp
BENCH_FULL_SRCS    := $(FULL_MUL_BENCH_SRC) $(CPP_SRCS) $(CU_SRCS)

BENCH_FULL_32 := build/bench_full_multiply_32
BENCH_FULL_64 := build/bench_full_multiply_64

$(BENCH_OBJ): $(BENCH_SRC) | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(BENCH_TARGET): $(BENCH_OBJ) $(CPP_OBJS) $(CU_OBJS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_PATHS) \
		-o $@ $^ $(LIBS)

$(BENCH_FULL_32): $(BENCH_FULL_SRCS) | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -DLIMB_BITS=32 $(INCLUDES) $(LIB_PATHS) \
		$^ -o $@ $(LIBS)

$(BENCH_FULL_64): $(BENCH_FULL_SRCS) | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -DLIMB_BITS=64 $(INCLUDES) $(LIB_PATHS) \
		$^ -o $@ $(LIBS)

# ================================================================
# Convenience targets
# ================================================================

bench: $(BENCH_TARGET)
bench_full_32: $(BENCH_FULL_32)
bench_full_64: $(BENCH_FULL_64)
bench_full: bench_full_32 bench_full_64

.PHONY: bench bench_full bench_full_32 bench_full_64
