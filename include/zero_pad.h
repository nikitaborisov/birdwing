#pragma once

#include "config.h"
#include <cuda_runtime.h>

void zero_pad_gpu(
    const uint32_t* d_src,
    TestDataTypeUint* d_dst,
    size_t L,
    size_t N,
    cudaStream_t stream);

#if defined(NATIVE_HOST_LIMBS)
void zero_pad_gpu_u64(
    const uint64_t* d_src,
    TestDataTypeUint* d_dst,
    size_t L,
    size_t N,
    TestDataTypeUint modulus,
    cudaStream_t stream);
#endif
