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
    size_t N)
{
    if (threadIdx.x != 0) return;

    size_t seg_start = (size_t)blockIdx.x * CARRY_SEG;
    size_t seg_end   = min(seg_start + CARRY_SEG, N);

    __int128 carry = 0;

    for (size_t i = seg_start; i < seg_end; i++) {
        __int128 val = ((__uint128_t)C_hi[i] << 64) | C_lo[i];

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
// The host retries this kernel until escape_flag stays zero: when incoming
// carry cannot fit in the remaining limbs of a segment, the overflow is
// atomically added to seg_carry[seg+1] and another pass is needed so the
// next segment can absorb it (carry can ripple across many segments).
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
    // Consume now: the host may rerun this kernel when escape_flag is set.
    // Leaving the slot nonzero would re-apply the same incoming carry on the
    // next pass (double-counting at segment boundaries).
    seg_carry[seg] = 0;
    if (incoming == 0) return;

    for (size_t i = seg_start; i < seg_end && incoming != 0; i++) {
        __int128 sum  = (__int128)out[i] + incoming;
        out[i]        = (OutputLimbType)(sum & OUTPUT_LIMB_MASK);
        incoming      = (int64_t)(sum >> OUTPUT_LIMB_BITS);
    }

    if (incoming != 0 && seg + 1 < num_segs) {
        // Carry escaped this segment; defer to the next fixup iteration.
        atomicAdd(reinterpret_cast<unsigned long long*>(&seg_carry[seg + 1]),
                  static_cast<unsigned long long>(incoming));
        *escape_flag = 1;
    }
}

#if defined(NATIVE_HOST_LIMBS)

__device__ __forceinline__
void u160_add_dev(uint64_t& lo, uint64_t& mid, uint32_t& hi,
                  uint64_t b_lo, uint64_t b_mid, uint32_t b_hi) {
    uint64_t old = lo;
    lo += b_lo;
    uint64_t c = lo < old ? 1ULL : 0ULL;
    old = mid;
    mid += b_mid + c;
    c = (mid < old || (c && mid == old)) ? 1ULL : 0ULL;
    hi += b_hi + (uint32_t)c;
}

__device__ __forceinline__
void atomic_add_seg_u160(uint64_t* seg_lo, uint64_t* seg_mid, uint32_t* seg_hi,
                         size_t idx, uint64_t a_lo, uint64_t a_mid, uint32_t a_hi) {
    if (a_lo == 0 && a_mid == 0 && a_hi == 0)
        return;
    unsigned long long old_lo =
        atomicAdd(reinterpret_cast<unsigned long long*>(&seg_lo[idx]),
                  static_cast<unsigned long long>(a_lo));
    uint64_t carry = (old_lo + a_lo < old_lo) ? 1ULL : 0ULL;
    if (a_mid != 0 || carry != 0) {
        unsigned long long old_mid =
            atomicAdd(reinterpret_cast<unsigned long long*>(&seg_mid[idx]),
                      static_cast<unsigned long long>(a_mid + carry));
        carry = (old_mid + a_mid + carry < old_mid) ? 1ULL : 0ULL;
    }
    if (a_hi != 0 || carry != 0)
        atomicAdd(reinterpret_cast<unsigned int*>(&seg_hi[idx]), a_hi + (uint32_t)carry);
}

__global__ void carry_intra_segment_kernel_u160(
    const uint64_t* __restrict__ C_lo,
    const uint64_t* __restrict__ C_mid,
    const uint32_t* __restrict__ C_hi,
    OutputLimbType* __restrict__ out,
    uint64_t*         __restrict__ seg_carry_lo,
    uint64_t*         __restrict__ seg_carry_mid,
    uint32_t*         __restrict__ seg_carry_hi,
    size_t N)
{
    if (threadIdx.x != 0) return;

    size_t seg_start = (size_t)blockIdx.x * CARRY_SEG;
    size_t seg_end   = min(seg_start + CARRY_SEG, N);

    uint64_t c_lo = 0, c_mid = 0;
    uint32_t c_hi = 0;

    for (size_t i = seg_start; i < seg_end; i++) {
        uint64_t lo = C_lo[i], mid = C_mid[i];
        uint32_t hi = C_hi[i];
        u160_add_dev(lo, mid, hi, c_lo, c_mid, c_hi);

        out[i] = (OutputLimbType)lo;

        c_lo = mid;
        c_mid = (uint64_t)hi;
        c_hi = 0;
    }
    seg_carry_lo[blockIdx.x]  = c_lo;
    seg_carry_mid[blockIdx.x] = c_mid;
    seg_carry_hi[blockIdx.x]  = c_hi;
}

__global__ void carry_inter_segment_kernel_u160(
    uint64_t* __restrict__ seg_carry_lo,
    uint64_t* __restrict__ seg_carry_mid,
    uint32_t* __restrict__ seg_carry_hi,
    size_t num_segs)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint64_t run_lo = 0, run_mid = 0;
    uint32_t run_hi = 0;

    for (size_t s = 0; s < num_segs; s++) {
        uint64_t inc_lo = run_lo, inc_mid = run_mid;
        uint32_t inc_hi = run_hi;

        run_lo  = seg_carry_lo[s];
        run_mid = seg_carry_mid[s];
        run_hi  = seg_carry_hi[s];

        seg_carry_lo[s]  = inc_lo;
        seg_carry_mid[s] = inc_mid;
        seg_carry_hi[s]  = inc_hi;
    }
}

__global__ void carry_fixup_kernel_u160(
    OutputLimbType* __restrict__ out,
    uint64_t*         __restrict__ seg_carry_lo,
    uint64_t*         __restrict__ seg_carry_mid,
    uint32_t*         __restrict__ seg_carry_hi,
    size_t N,
    size_t num_segs,
    int*              __restrict__ escape_flag)
{
    if (threadIdx.x != 0) return;

    size_t seg       = blockIdx.x;
    size_t seg_start = seg * CARRY_SEG;
    size_t seg_end   = min(seg_start + CARRY_SEG, N);

    uint64_t c_lo = seg_carry_lo[seg];
    uint64_t c_mid = seg_carry_mid[seg];
    uint32_t c_hi = seg_carry_hi[seg];
    // Same consume-on-read as carry_fixup_kernel (see comment there).
    seg_carry_lo[seg] = 0;
    seg_carry_mid[seg] = 0;
    seg_carry_hi[seg] = 0;
    if (c_lo == 0 && c_mid == 0 && c_hi == 0) return;

    // Stop early once carry is fully absorbed (mirrors the u128 fixup loop).
    for (size_t i = seg_start; i < seg_end && (c_lo != 0 || c_mid != 0 || c_hi != 0); i++) {
        uint64_t lo = out[i];
        uint64_t mid = 0;
        uint32_t hi = 0;
        u160_add_dev(lo, mid, hi, c_lo, c_mid, c_hi);
        out[i] = (OutputLimbType)lo;
        c_lo = mid;
        c_mid = (uint64_t)hi;
        c_hi = 0;
    }

    if ((c_lo != 0 || c_mid != 0 || c_hi != 0) && seg + 1 < num_segs) {
        // Carry escaped this segment; defer to the next fixup iteration.
        atomic_add_seg_u160(seg_carry_lo, seg_carry_mid, seg_carry_hi,
                            seg + 1, c_lo, c_mid, c_hi);
        *escape_flag = 1;
    }
}

#endif // NATIVE_HOST_LIMBS