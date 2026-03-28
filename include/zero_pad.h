#include "config.h"

void zero_pad_gpu(
    const TestDataTypeUint* d_src,
    TestDataTypeUint* d_dst,
    size_t L,
    size_t N,
    cudaStream_t stream
);