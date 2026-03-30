#pragma once

#include "config.h"
#include <cstddef>
#include <cstdint>

__global__ void carry_intra_segment_kernel(
    const uint64_t* __restrict__ d_C_hi,
    const uint64_t* __restrict__ d_C_lo,
    uint32_t*       __restrict__ d_out,
    int64_t*        __restrict__ d_segment_carry,
    size_t N,
    unsigned __int128 M);

__global__ void carry_inter_segment_kernel(
    int64_t* __restrict__ d_segment_carry,
    size_t   num_segs);

__global__ void carry_fixup_kernel(
    TestDataTypeUint* __restrict__ d_out,
    const int64_t*    __restrict__ d_segment_carry,
    size_t N);