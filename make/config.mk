# ================================================================
# Project root
# ================================================================

MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ROOT_DIR     := $(abspath $(MAKEFILE_DIR)/..)

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

GPU_NTT_DIR   := $(ROOT_DIR)/GPU-NTT
GPU_NTT_BUILD := $(GPU_NTT_DIR)/build
GPU_NTT_LIB   := $(GPU_NTT_BUILD)/src/libntt-1.0.a
GPU_NTT_INC   := $(GPU_NTT_DIR)/src/include

GMP_PREFIX ?=
ifneq ($(GMP_PREFIX),)
    GMP_INCLUDES := -I$(GMP_PREFIX)/include
    GMP_LIB_PATH := -L$(GMP_PREFIX)/lib
else
    GMP_INCLUDES :=
    GMP_LIB_PATH :=
endif

INCLUDES := \
    -Iinclude \
    -I$(CUDA_PATH)/include \
    -I$(GPU_NTT_INC) \
    -I$(GPU_NTT_INC)/gpuntt/ntt_merge \
    $(GMP_INCLUDES)

LIB_PATHS := \
    $(GMP_LIB_PATH)

LIBS := \
    $(GPU_NTT_LIB) \
    -lgmp \
    -lcudart

# ================================================================
# Targets
# ================================================================

TARGET       := main
BENCH_TARGET := bench_ntt