// carry_prop.h
#pragma once
#include <stdint.h>
#include <stddef.h>
#include "config.h"

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
    int64_t*          __restrict__ seg_carry,
    size_t N,
    size_t num_segs,
    int*              __restrict__ escape_flag);