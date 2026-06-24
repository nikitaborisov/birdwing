#include "zero_pad.h"

__global__ void zero_pad_kernel(
    const uint32_t* __restrict__ src,   // always 32-bit input
    TestDataTypeUint* __restrict__ dst, // 32 or 64-bit output
    size_t L,
    size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        dst[idx] = (idx < L) ? (TestDataTypeUint)src[idx] : 0;
}

void zero_pad_gpu(
    const uint32_t* d_src,
    TestDataTypeUint* d_dst,
    size_t L,
    size_t N,
    cudaStream_t stream
) {
    int threads_per_block = 256;
    int blocks = (N + threads_per_block - 1) / threads_per_block;
    zero_pad_kernel<<<blocks, threads_per_block, 0, stream>>>(d_src, d_dst, L, N);
}

#if defined(NATIVE_HOST_LIMBS)

__global__ void zero_pad_kernel_u64(
    const uint64_t* __restrict__ src,
    TestDataTypeUint* __restrict__ dst,
    size_t L,
    size_t N,
    TestDataTypeUint modulus)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        if (idx < L)
            dst[idx] = (TestDataTypeUint)(src[idx] % modulus);
        else
            dst[idx] = 0;
    }
}

void zero_pad_gpu_u64(
    const uint64_t* d_src,
    TestDataTypeUint* d_dst,
    size_t L,
    size_t N,
    TestDataTypeUint modulus,
    cudaStream_t stream)
{
    int threads_per_block = 256;
    int blocks = (N + threads_per_block - 1) / threads_per_block;
    zero_pad_kernel_u64<<<blocks, threads_per_block, 0, stream>>>(
        d_src, d_dst, L, N, modulus);
}

#endif