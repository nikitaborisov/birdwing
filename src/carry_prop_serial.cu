#include "carry_prop_serial.h"

__global__ void carry_prop_serial_kernel(
    const uint64_t*   __restrict__ C_hi,
    const uint64_t*   __restrict__ C_lo,
    TestDataTypeUint* __restrict__ out,
    size_t N,
    unsigned __int128 M)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const __int128 BASE = (__int128)1 << 32;
    __int128 carry = 0;

    for (size_t i = 0; i < N; i++) {
        __int128 val = ((__uint128_t)C_hi[i] << 64) | C_lo[i];
        if (val > (__int128)(M / 2)) val -= (__int128)M;

        __int128 temp = val + carry;
        __int128 limb = temp % BASE;
        if (limb < 0) { limb += BASE; temp -= BASE; }
        out[i] = (TestDataTypeUint)limb;
        carry  = (temp - limb) / BASE;
    }
    if (carry != 0) out[N] = (TestDataTypeUint)carry;
}