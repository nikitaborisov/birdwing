# ================================================================
# Toolchain
# ================================================================

NVCC := nvcc
CXX  := g++

CUDA_PATH ?= /usr/local/cuda

# ================================================================
# Build flags
# ================================================================

CXXFLAGS := -std=c++17 -O3 -Wall -Wextra

NVCCFLAGS := -std=c++17 -O3 -lineinfo \
             -Xcompiler -Wall \
             -Xcompiler -Wextra \

ifdef PROFILE
    CXXFLAGS  += -DPROFILE -g
    NVCCFLAGS += -DPROFILE -g -lineinfo
endif

ifdef TIMING
    CXXFLAGS  += -DTIMING
    NVCCFLAGS += -DTIMING
endif

ifdef DEBUG
    CXXFLAGS  += -DDEBUG -g
    NVCCFLAGS += -DDEBUG -g -lineinfo
endif

# ================================================================
# Directories
# ================================================================

SRC_DIR     := src
TEST_DIR    := tests
OBJ_DIR     := build
TEST_BUILD  := $(OBJ_DIR)/tests

# ================================================================
# Includes / Libraries
# ================================================================

INCLUDES := \
    -Iinclude \
    -I$(CUDA_PATH)/include \
    -I$(HOME)/.local/include \
    -I$(HOME)/GPU-NTT/src/include \
    -I$(HOME)/GPU-NTT/src/include/common \
    -I$(HOME)/GPU-NTT/src/include/ntt_merge

LIB_PATHS := \
    -L$(HOME)/.local/lib \
    -L$(HOME)/gmp-local/lib

LIBS := \
    -lntt-1.0 \
    -lgmp

# ================================================================
# Targets
# ================================================================

TARGET       := main
BENCH_TARGET := bench_ntt