#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#ifndef DEBUG
#define DEBUG 0
#endif

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _err = (expr);                                             \
        if (_err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(_err));                                 \
            abort();                                                           \
        }                                                                      \
    } while (0)

#if DEBUG
#define CUDA_CHECK_KERNEL()                                                    \
    do {                                                                       \
        cudaError_t _err = cudaGetLastError();                                 \
        if (_err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA kernel launch error at %s:%d: %s\n",         \
                    __FILE__, __LINE__, cudaGetErrorString(_err));             \
            abort();                                                           \
        }                                                                      \
        _err = cudaDeviceSynchronize();                                        \
        if (_err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA kernel exec error at %s:%d: %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(_err));             \
            abort();                                                           \
        }                                                                      \
    } while (0)
#else
#define CUDA_CHECK_KERNEL()                                                    \
    do {                                                                       \
        cudaError_t _err = cudaGetLastError();                                 \
        if (_err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA kernel launch error at %s:%d: %s\n",         \
                    __FILE__, __LINE__, cudaGetErrorString(_err));             \
            abort();                                                           \
        }                                                                      \
    } while (0)
#endif
