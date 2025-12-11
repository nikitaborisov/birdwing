#include "crt_utils.h"
#include <cstdio>
#include <cassert>

using namespace std;

// ------------------ 128-bit helpers (for printing/testing) ------------------

void print_u128(unsigned __int128 x) {
    // Minimal decimal printing for debugging
    char buf[64];
    int idx = 63;
    buf[idx--] = '\0';
    if (x == 0) {
        std::printf("0");
        return;
    }
    while (x > 0) {
        unsigned __int128 q = x / 10;
        unsigned int digit = (unsigned int)(x - q * 10);
        buf[idx--] = '0' + digit;
        x = q;
    }
    std::printf("%s", &buf[idx+1]);
}

// TODO probably can get rid of this later and just hardcode the inverses for selected primes

uint64_t modinv_u64(uint64_t a, uint64_t m) {
    // Extended Euclid: works as long as a, m < 2^63 OR we use __int128 carefully.
    int64_t t0 = 0, t1 = 1;
    int64_t r0 = (int64_t)m, r1 = (int64_t)a;

    while (r1 != 0) {
        int64_t q = r0 / r1;
        int64_t r2 = r0 - q * r1;
        r0 = r1;
        r1 = r2;

        int64_t t2 = t0 - q * t1;
        t0 = t1;
        t1 = t2;
    }

    assert(r0 == 1); // a and m must be coprime (true for prime moduli)

    if (t0 < 0)
        t0 += (int64_t)m;

    return (uint64_t)t0;
}