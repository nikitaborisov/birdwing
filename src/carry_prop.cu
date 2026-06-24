// carry_prop.cu
#include <cuda_runtime.h>
#include <stdint.h>
#include "config.h"

// Pass 1: each block resolves carry within its own segment independently.
// Writes reduced limbs to out[] and the segment's outgoing carry to seg_carry[].
__global__ void carry_intra_segment_kernel(
    const uint64_t* __restrict__ C_hi,
    const uint64_t* __restrict__ C_lo,
    OutputLimbType* __restrict__ out,
    int64_t*          __restrict__ seg_carry,
    size_t N,
    unsigned __int128 M,
    unsigned __int128 M_half)
{
    if (threadIdx.x != 0) return;

    size_t seg_start = (size_t)blockIdx.x * CARRY_SEG;
    size_t seg_end   = min(seg_start + CARRY_SEG, N);

    __int128 carry = 0;

    for (size_t i = seg_start; i < seg_end; i++) {
        __int128 val = ((__uint128_t)C_hi[i] << 64) | C_lo[i];
        if (val > (__int128)M_half) val -= (__int128)M;

        __int128 temp = val + carry;
        OutputLimbType limb = (OutputLimbType)(temp & OUTPUT_LIMB_MASK);
        out[i] = limb;
        carry  = temp >> OUTPUT_LIMB_BITS;
    }
    seg_carry[blockIdx.x] = (int64_t)carry;
}

// Pass 2: serial scan over just (N/CARRY_SEG) carry values — tiny, fast.
// Converts per-segment raw carries into the addend each segment needs to receive.
__global__ void carry_inter_segment_kernel(
    int64_t* __restrict__ seg_carry,
    size_t   num_segs)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int64_t running = 0;
    for (size_t s = 0; s < num_segs; s++) {
        int64_t incoming = running;
        running          = seg_carry[s];
        seg_carry[s]     = incoming;
    }
    // seg_carry[s] now holds the carry that must be added to the first limb of seg s
    // seg_carry[0] will always be 0 (nothing flows into the first segment)
}

// Pass 3: each block adds the incoming carry into its segment's limbs,
// propagating within the segment if an addition overflows a limb.
__global__ void carry_fixup_kernel(
    OutputLimbType* __restrict__ out,
    int64_t*          __restrict__ seg_carry,
    size_t N,
    size_t num_segs,
    int*              __restrict__ escape_flag)
{
    if (threadIdx.x != 0) return;

    size_t seg       = blockIdx.x;
    size_t seg_start = seg * CARRY_SEG;
    size_t seg_end   = min(seg_start + CARRY_SEG, N);

    int64_t incoming = seg_carry[seg];
    if (incoming == 0) return;

    for (size_t i = seg_start; i < seg_end && incoming != 0; i++) {
        __int128 sum  = (__int128)out[i] + incoming;
        out[i]        = (OutputLimbType)(sum & OUTPUT_LIMB_MASK);
        incoming      = (int64_t)(sum >> OUTPUT_LIMB_BITS);
    }

    if (incoming != 0 && seg + 1 < num_segs) {
        atomicAdd(reinterpret_cast<unsigned long long*>(&seg_carry[seg + 1]),
                  static_cast<unsigned long long>(incoming));
        *escape_flag = 1;
    }
}