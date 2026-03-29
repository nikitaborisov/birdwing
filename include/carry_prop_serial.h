#include "config.h"

__global__ void carry_prop_serial_kernel(
    const uint64_t*   __restrict__ C_hi,
    const uint64_t*   __restrict__ C_lo,
    TestDataTypeUint* __restrict__ out,
    size_t N,
    unsigned __int128 M);