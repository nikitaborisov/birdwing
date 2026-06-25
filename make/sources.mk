# ================================================================
# Source files
# ================================================================

MAIN_SRC  := main.cpp
BENCH_SRC := bench/gpu_ntt_benchmark.cu

CPP_SRCS := $(wildcard $(SRC_DIR)/*.cpp)
CU_SRCS  := $(wildcard $(SRC_DIR)/*.cu)

TEST_SRCS_CPP := $(filter-out $(TEST_DIR)/test_ntt_limits.cpp,$(wildcard $(TEST_DIR)/*.cpp))
TEST_SRCS_CU  := $(wildcard $(TEST_DIR)/*.cu)

# ================================================================
# Object files
# ================================================================

CPP_OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CPP_SRCS))
CU_OBJS  := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(CU_SRCS))

MAIN_OBJ  := $(OBJ_DIR)/main.o
BENCH_OBJ := $(OBJ_DIR)/gpu_ntt_benchmark.o

OBJS := $(CPP_OBJS) $(CU_OBJS) $(MAIN_OBJ)

# ================================================================
# Test binaries
# ================================================================

TEST_BINS_CPP := \
    $(patsubst $(TEST_DIR)/%.cpp,$(TEST_BUILD)/%,$(TEST_SRCS_CPP))

TEST_BINS_CU := \
    $(patsubst $(TEST_DIR)/%.cu,$(TEST_BUILD)/%,$(TEST_SRCS_CU))

TEST_BINS := $(TEST_BINS_CPP) $(TEST_BINS_CU)