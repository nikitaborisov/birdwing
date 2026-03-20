#include "zero_pad.h"

__global__ void zero_pad_kernel(
    const TestDataTypeUint* __restrict__ src,
    TestDataTypeUint* __restrict__ dst,
    size_t L,   // original length
    size_t N)   // padded length
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        dst[idx] = (idx < L) ? src[idx] : 0;
}

void zero_pad_gpu(
    const TestDataTypeUint* d_src,
    TestDataTypeUint* d_dst,
    size_t L,
    size_t N
) {
    int threads_per_block = 256;
    int blocks = (N + threads_per_block - 1) / threads_per_block;
    zero_pad_kernel<<<blocks, threads_per_block>>>(d_src, d_dst, L, N);
}