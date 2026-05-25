# ================================================================
# Test build rules
# ================================================================

# Build C++ tests
$(TEST_BUILD)/%: $(TEST_DIR)/%.cpp $(CPP_OBJS) $(CU_OBJS) | $(TEST_BUILD)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) \
		$< $(CPP_OBJS) $(CU_OBJS) \
		-o $@ $(LIB_PATHS) $(LIBS)

# Build CUDA tests
$(TEST_BUILD)/%: $(TEST_DIR)/%.cu $(CPP_OBJS) $(CU_OBJS) | $(TEST_BUILD)
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

test: $(TEST_BINS)
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

$(TEST_BUILD)/test_zero_pad_32: $(ZP_SRC) $(ZP_SRCS_DEP) | $(TEST_BUILD)
	$(NVCC) $(NVCCFLAGS) -DLIMB_BITS=32 $(INCLUDES) \
		$^ -o $@ $(LIB_PATHS) $(LIBS)

$(TEST_BUILD)/test_zero_pad_64: $(ZP_SRC) $(ZP_SRCS_DEP) | $(TEST_BUILD)
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
# Phony targets
# ================================================================

.PHONY: test test_zero_pad