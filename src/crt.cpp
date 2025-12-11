#include "crt.h"
#include <cassert>

// ------------------ CRT2Params constructor ------------------
CRT2Params::CRT2Params(uint64_t _p1, uint64_t _p2)
    : p1(_p1), p2(_p2)
{
    modulus = (unsigned __int128)p1 * (unsigned __int128)p2;
    p1_inv_mod_p2 = modinv_u64(p1 % p2, p2);
}

// ------------------ Combine (a mod p1, b mod p2) -> x mod p1*p2 ------------------

unsigned __int128 crt_combine_2(const CRT2Params &params,
                                       uint64_t a_mod_p1,
                                       uint64_t b_mod_p2) {
    uint64_t p1 = params.p1;
    uint64_t p2 = params.p2;
    uint64_t inv = params.p1_inv_mod_p2;

    // t = (b - a) mod p2
    uint64_t t;
    if (b_mod_p2 >= a_mod_p1)
        t = b_mod_p2 - a_mod_p1;
    else
        t = b_mod_p2 + (p2 - a_mod_p1);  // wrap around

    // t = t * inv (mod p2)
    unsigned __int128 tt = (unsigned __int128)t * (unsigned __int128)inv;
    t = (uint64_t)(tt % p2);   // still safely within uint64_t

    // x = a + p1 * t, guaranteed < p1*p2
    unsigned __int128 x = (unsigned __int128)a_mod_p1 +
                          (unsigned __int128)p1 * (unsigned __int128)t;

    // x is already in [0, p1*p2), so no need to reduce
    return x;
}

// ------------------ General CRT for many primes ------------------
unsigned __int128 crt_combine_many(const vector<uint64_t> &primes, const vector<uint64_t> &residues) {
    assert(primes.size() == residues.size());
    size_t k = primes.size();
    assert(k > 0);

    // Current combined modulus and solution:
    unsigned __int128 M = 1;           // product of processed primes
    unsigned __int128 x = 0;           // solution mod M

    for (size_t i = 0; i < k; ++i) {
        uint64_t p = primes[i];
        uint64_t r = residues[i];

        // We have: x ≡ current x (mod M), want x ≡ r (mod p)
        // Let x' be the new solution, M' = M * p.

        // First, reduce current x modulo p
        uint64_t x_mod_p = (uint64_t)(x % p);

        // t = (r - x_mod_p) mod p
        uint64_t t;
        if (r >= x_mod_p)
            t = r - x_mod_p;
        else
            t = r + (p - x_mod_p);

        // Compute M_mod_p = M % p as a 64-bit value
        uint64_t M_mod_p = (uint64_t)(M % p);

        // inv = (M_mod_p)^{-1} mod p
        uint64_t inv = modinv_u64(M_mod_p, p);

        // k_i = t * inv (mod p)
        unsigned __int128 tmp = (unsigned __int128)t * (unsigned __int128)inv;
        uint64_t k_i = (uint64_t)(tmp % p);

        // Update x and M:
        // x' = x + M * k_i   (mod M*p) but x' is chosen in [0, M*p)
        x += M * (unsigned __int128)k_i;

        // Update modulus
        M *= (unsigned __int128)p;
    }

    return x;
}