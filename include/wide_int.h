#pragma once

#include <cstdint>

// 160-bit unsigned integer: bits [0,64), [64,128), [128,160)
struct U160 {
    uint64_t lo;
    uint64_t mid;
    uint32_t hi;
};

inline uint64_t u160_to_u64_lo(const U160& x) { return x.lo; }
inline uint64_t u160_to_u64_mid(const U160& x) { return x.mid; }
inline uint32_t u160_to_u32_hi(const U160& x) { return x.hi; }

inline bool u160_eq(const U160& a, const U160& b) {
    return a.lo == b.lo && a.mid == b.mid && a.hi == b.hi;
}

inline void u160_from_u128(uint64_t lo, uint64_t hi, U160& out) {
    out.lo = lo;
    out.mid = hi;
    out.hi = 0;
}

// Host-side helpers (mirrored on device in crt_gpu.cu / carry_prop.cu)
inline void add160_host(U160& a, uint64_t b_lo, uint64_t b_mid, uint32_t b_hi) {
    uint64_t old = a.lo;
    a.lo += b_lo;
    uint64_t c = a.lo < old ? 1ULL : 0ULL;

    old = a.mid;
    a.mid += b_mid + c;
    c = (a.mid < old || (c && a.mid == old)) ? 1ULL : 0ULL;

    a.hi += b_hi + (uint32_t)c;
}

inline void mul128x64_host(uint64_t M_hi, uint64_t M_lo, uint64_t k,
                           uint64_t& out_lo, uint64_t& out_mid, uint32_t& out_hi) {
    unsigned __int128 p0 = (unsigned __int128)M_lo * k;
    out_lo = (uint64_t)p0;
    unsigned __int128 p1 = (unsigned __int128)M_hi * k + (uint64_t)(p0 >> 64);
    out_mid = (uint64_t)p1;
    out_hi = (uint32_t)(p1 >> 64);
}
