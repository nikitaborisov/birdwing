// carry_prop.h
#pragma once
#include <stdint.h>
#include <stddef.h>
#include "config.h"

// Host-side launch geometry for intra/fixup carry kernels (one thread per segment).
// grid_dim = min(num_segs, max(64, num_segs/512)); block_dim = ceil(num_segs/grid_dim).
inline void carry_launch_dims(size_t num_segs, int& grid_dim, int& block_dim) {
    if (num_segs == 0) {
        grid_dim = 1;
        block_dim = 1;
        return;
    }
    size_t g = num_segs / 512;
    if (g < 64) g = 64;
    if (g > num_segs) g = num_segs;
    grid_dim = (int)g;
    block_dim = (int)((num_segs + (size_t)grid_dim - 1) / (size_t)grid_dim);
}

__global__ void carry_intra_segment_kernel(
    const uint64_t*   __restrict__ C_hi,
    const uint64_t*   __restrict__ C_lo,
    OutputLimbType* __restrict__ out,
    int64_t*          __restrict__ seg_carry,
    size_t N);

__global__ void carry_inter_segment_kernel(
    int64_t* __restrict__ seg_carry,
    size_t   num_segs);

__global__ void carry_fixup_kernel(
    OutputLimbType* __restrict__ out,
    const int64_t*    __restrict__ seg_carry_in,
    int64_t*          __restrict__ seg_carry_out,
    size_t N,
    size_t num_segs,
    int*              __restrict__ escape_flag);

#if defined(NATIVE_HOST_LIMBS)
__global__ void carry_intra_segment_kernel_u160(
    const uint64_t* __restrict__ C_lo,
    const uint64_t* __restrict__ C_mid,
    const uint32_t* __restrict__ C_hi,
    OutputLimbType* __restrict__ out,
    uint64_t*         __restrict__ seg_carry_lo,
    uint64_t*         __restrict__ seg_carry_mid,
    uint32_t*         __restrict__ seg_carry_hi,
    size_t N);

__global__ void carry_inter_segment_kernel_u160(
    uint64_t* __restrict__ seg_carry_lo,
    uint64_t* __restrict__ seg_carry_mid,
    uint32_t* __restrict__ seg_carry_hi,
    size_t   num_segs);

__global__ void carry_fixup_kernel_u160(
    OutputLimbType* __restrict__ out,
    const uint64_t*   __restrict__ seg_carry_in_lo,
    const uint64_t*   __restrict__ seg_carry_in_mid,
    const uint32_t*   __restrict__ seg_carry_in_hi,
    uint64_t*         __restrict__ seg_carry_out_lo,
    uint64_t*         __restrict__ seg_carry_out_mid,
    uint32_t*         __restrict__ seg_carry_out_hi,
    size_t N,
    size_t num_segs,
    int*              __restrict__ escape_flag);
#endif