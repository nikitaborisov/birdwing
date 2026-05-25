# ================================================================
# Benchmark build
# ================================================================

$(BENCH_OBJ): $(BENCH_SRC) | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(BENCH_TARGET): $(BENCH_OBJ) $(CPP_OBJS) $(CU_OBJS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_PATHS) \
		-o $@ $^ $(LIBS)

# ================================================================
# Convenience target
# ================================================================

bench: $(BENCH_TARGET)

.PHONY: bench