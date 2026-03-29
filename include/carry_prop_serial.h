#include "config.h"

__global__ void carry_intra_segment_kernel(
    const uint64_t* __restrict__ C_hi,
    const uint64_t* __restrict__ C_lo,
    uint32_t*       __restrict__ out,
    int64_t*        __restrict__ seg_carry,
    size_t N,
    unsigned __int128 M);

__global__ void carry_inter_segment_kernel(
    int64_t* __restrict__ seg_carry,
    size_t   num_segs);

__global__ void carry_fixup_kernel(
    uint32_t*       __restrict__ out,
    const int64_t*  __restrict__ seg_carry,
    size_t N);