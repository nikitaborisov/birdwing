# ================================================================
# Test build rules
# ================================================================

# Build C++ tests
$(TEST_BUILD)/%: $(TEST_DIR)/%.cpp $(CPP_OBJS) $(CU_OBJS) | $(TEST_BUILD) $(GPU_NTT_LIB)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) \
		$< $(CPP_OBJS) $(CU_OBJS) \
		-o $@ $(LIB_PATHS) $(LIBS)

# Build CUDA tests
$(TEST_BUILD)/%: $(TEST_DIR)/%.cu $(CPP_OBJS) $(CU_OBJS) | $(TEST_BUILD) $(GPU_NTT_LIB)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) \
		$< $(CPP_OBJS) $(CU_OBJS) \
		-o $@ $(LIB_PATHS) $(LIBS)

# ================================================================
# Run tests
# ================================================================

# Run all:
#   make test
#
# Run one:
#   make test TEST=test_name

test: gpu-ntt $(TEST_BINS)
ifdef TEST
	@echo "Running test: $(TEST)"
	@$(TEST_BUILD)/$(TEST)
else
	@echo "Running all tests..."
	@for t in $(TEST_BINS); do \
		echo "===== $$t ====="; \
		$$t || exit 1; \
	done
	@echo "All tests passed!"
endif

# ================================================================
# zero_pad dual-width tests
# ================================================================

ZP_SRC      := $(TEST_DIR)/test_zero_pad.cu
ZP_SRCS_DEP := $(SRC_DIR)/zero_pad.cu

$(TEST_BUILD)/test_zero_pad_32: $(ZP_SRC) $(ZP_SRCS_DEP) | $(TEST_BUILD) $(GPU_NTT_LIB)
	$(NVCC) $(NVCCFLAGS) -DLIMB_BITS=32 $(INCLUDES) \
		$^ -o $@ $(LIB_PATHS) $(LIBS)

$(TEST_BUILD)/test_zero_pad_64: $(ZP_SRC) $(ZP_SRCS_DEP) | $(TEST_BUILD) $(GPU_NTT_LIB)
	$(NVCC) $(NVCCFLAGS) -DLIMB_BITS=64 $(INCLUDES) \
		$^ -o $@ $(LIB_PATHS) $(LIBS)

test_zero_pad: \
	$(TEST_BUILD)/test_zero_pad_32 \
	$(TEST_BUILD)/test_zero_pad_64
	@echo "===== zero_pad 32-bit ====="
	@$(TEST_BUILD)/test_zero_pad_32
	@echo "===== zero_pad 64-bit ====="
	@$(TEST_BUILD)/test_zero_pad_64

# ================================================================
# crt_gpu dual-width tests
# ================================================================

CRT_SRC      := $(TEST_DIR)/test_crt_gpu.cu
CRT_SRCS_DEP := $(SRC_DIR)/crt_gpu.cu $(SRC_DIR)/crt_utils.cpp

$(TEST_BUILD)/test_crt_gpu_32: $(CRT_SRC) $(CRT_SRCS_DEP) | $(TEST_BUILD) $(GPU_NTT_LIB)
	$(NVCC) $(NVCCFLAGS) -DLIMB_BITS=32 $(INCLUDES) \
		$^ -o $@ $(LIB_PATHS) $(LIBS)

$(TEST_BUILD)/test_crt_gpu_64: $(CRT_SRC) $(CRT_SRCS_DEP) | $(TEST_BUILD) $(GPU_NTT_LIB)
	$(NVCC) $(NVCCFLAGS) -DLIMB_BITS=64 $(INCLUDES) \
		$^ -o $@ $(LIB_PATHS) $(LIBS)

test_crt_gpu: \
	$(TEST_BUILD)/test_crt_gpu_32 \
	$(TEST_BUILD)/test_crt_gpu_64
	@echo "===== crt_gpu 32-bit ====="
	@$(TEST_BUILD)/test_crt_gpu_32
	@echo "===== crt_gpu 64-bit ====="
	@$(TEST_BUILD)/test_crt_gpu_64

# ================================================================
# Carry Propagation dual-width tests
# ================================================================

CARRY_SRC      := $(TEST_DIR)/test_carry_prop.cu
CARRY_SRCS_DEP := $(SRC_DIR)/carry_prop.cu

$(TEST_BUILD)/test_carry_prop_32: $(CARRY_SRC) $(CARRY_SRCS_DEP) | $(TEST_BUILD) $(GPU_NTT_LIB)
	$(NVCC) $(NVCCFLAGS) -DLIMB_BITS=32 $(INCLUDES) \
		$^ -o $@ $(LIB_PATHS) $(LIBS)

$(TEST_BUILD)/test_carry_prop_64: $(CARRY_SRC) $(CARRY_SRCS_DEP) | $(TEST_BUILD) $(GPU_NTT_LIB)
	$(NVCC) $(NVCCFLAGS) -DLIMB_BITS=64 $(INCLUDES) \
		$^ -o $@ $(LIB_PATHS) $(LIBS)

test_carry_prop: \
	$(TEST_BUILD)/test_carry_prop_32 \
	$(TEST_BUILD)/test_carry_prop_64
	@echo "===== carry_prop 32-bit ====="
	@$(TEST_BUILD)/test_carry_prop_32
	@echo "===== carry_prop 64-bit ====="
	@$(TEST_BUILD)/test_carry_prop_64

# ================================================================
# ntt_limits dual-width tests
# ================================================================

NTT_LIMITS_SRC     := $(TEST_DIR)/test_ntt_limits.cpp
NTT_LIMITS_OBJS    := $(OBJ_DIR)/ntt_limits_test.o $(OBJ_DIR)/ntt_limits.o

$(OBJ_DIR)/ntt_limits_test.o: $(NTT_LIMITS_SRC) | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -DLIMB_BITS=32 $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/ntt_limits_test_64.o: $(NTT_LIMITS_SRC) | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -DLIMB_BITS=64 $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/ntt_limits_32.o: $(SRC_DIR)/ntt_limits.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -DLIMB_BITS=32 $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/ntt_limits_64.o: $(SRC_DIR)/ntt_limits.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -DLIMB_BITS=64 $(INCLUDES) -c $< -o $@

$(TEST_BUILD)/test_ntt_limits_32: $(OBJ_DIR)/ntt_limits_test.o $(OBJ_DIR)/ntt_limits_32.o | $(TEST_BUILD)
	$(CXX) $(CXXFLAGS) -DLIMB_BITS=32 $(INCLUDES) \
		$^ -o $@

$(TEST_BUILD)/test_ntt_limits_64: $(OBJ_DIR)/ntt_limits_test_64.o $(OBJ_DIR)/ntt_limits_64.o | $(TEST_BUILD)
	$(CXX) $(CXXFLAGS) -DLIMB_BITS=64 $(INCLUDES) \
		$^ -o $@

test_ntt_limits: \
	$(TEST_BUILD)/test_ntt_limits_32 \
	$(TEST_BUILD)/test_ntt_limits_64
	@echo "===== ntt_limits 32-bit ====="
	@$(TEST_BUILD)/test_ntt_limits_32
	@echo "===== ntt_limits 64-bit ====="
	@$(TEST_BUILD)/test_ntt_limits_64

# ================================================================
# End-to-end dual-width builds
# ================================================================

E2E_SRCS := tests/test_full_multiply.cpp $(CPP_SRCS) $(CU_SRCS)

main_32: | $(OBJ_DIR) $(GPU_NTT_LIB)
	$(NVCC) $(NVCCFLAGS) -DLIMB_BITS=32 $(INCLUDES) $(LIB_PATHS) \
		$(E2E_SRCS) -o build/test_full_multiply_32 $(LIBS)

main_64: | $(OBJ_DIR) $(GPU_NTT_LIB)
	$(NVCC) $(NVCCFLAGS) -DLIMB_BITS=64 $(INCLUDES) $(LIB_PATHS) \
		$(E2E_SRCS) -o build/test_full_multiply_64 $(LIBS)

# ================================================================
# Phony targets
# ================================================================

.PHONY: test test_zero_pad test_crt_gpu test_carry_prop test_ntt_limits main_32 main_64