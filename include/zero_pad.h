#include "config.h"

void zero_pad_gpu(
    // changed from TestDataTypeUint to uint32_t
    const uint32_t* d_src,
    TestDataTypeUint* d_dst,
    size_t L,
    size_t N,
    cudaStream_t stream
);