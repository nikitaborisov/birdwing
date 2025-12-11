#include <cstdint>
#include <cstdio>
#include <cassert>
#include <vector>

using namespace std;

// ------------------ 128-bit helpers (for printing/testing) ------------------

static inline void print_u128(unsigned __int128 x) {
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

// ------------------ CRT params for two primes ------------------

struct CRT2Params {
    uint64_t p1;
    uint64_t p2;
    uint64_t p1_inv_mod_p2;      // p1^{-1} mod p2
    unsigned __int128 modulus;   // p1 * p2

    CRT2Params(uint64_t _p1, uint64_t _p2)
        : p1(_p1), p2(_p2) {
        // product fits in 128 bits easily
        modulus = (unsigned __int128)p1 * (unsigned __int128)p2;
        p1_inv_mod_p2 = modinv_u64(p1 % p2, p2);
    }
};

// ------------------ Combine (a mod p1, b mod p2) -> x mod p1*p2 ------------------

inline unsigned __int128 crt_combine_2(const CRT2Params &params,
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

// ------------------ Example / test ------------------

int main() {
    // Example 50-bit-ish primes (same as before)
    uint64_t p1 = ( (uint64_t)1 << 43 ) * 3 * 25 + 1;  // 2^43 * 3 * 5^2 + 1
    uint64_t p2 = ( (uint64_t)1 << 44 ) * 9 * 7 + 1;   // 2^44 * 3^2 * 7 + 1

    CRT2Params params_2(p1, p2);

    unsigned __int128 x_true = 12345678901234567890ULL; // example

    uint64_t x_mod_p1 = (uint64_t)(x_true % p1);
    uint64_t x_mod_p2 = (uint64_t)(x_true % p2);

    // Test 2-prime version
    unsigned __int128 x_recovered_2 = crt_combine_2(params_2, x_mod_p1, x_mod_p2);

    std::printf("2-prime CRT:\n");
    std::printf("x_true      = ");
    print_u128(x_true);
    std::printf("\n");
    std::printf("x_recovered = ");
    print_u128(x_recovered_2);
    std::printf("\n");
    std::printf("2-prime CRT %s\n\n",
                (x_true == x_recovered_2) ? "PASSED" : "FAILED");

    // Now test multi-prime CRT with the same two primes
    std::vector<uint64_t> primes = {p1, p2};
    std::vector<uint64_t> residues = {x_mod_p1, x_mod_p2};

    unsigned __int128 x_recovered_many = crt_combine_many(primes, residues);

    std::printf("multi-prime CRT (k=2):\n");
    std::printf("x_true      = ");
    print_u128(x_true);
    std::printf("\n");
    std::printf("x_recovered = ");
    print_u128(x_recovered_many);
    std::printf("\n");
    std::printf("multi-prime CRT %s\n",
                (x_true == x_recovered_many) ? "PASSED" : "FAILED");

    // Three roughly 40–45-bit NTT-friendly primes
    p1 = ((uint64_t)1 << 43) * 3 * 25 + 1;
    p2 = ((uint64_t)1 << 44) * 9 * 7 + 1;
    uint64_t p3 = ((uint64_t)1 << 40) * 17 + 1;   // arbitrary ~40-bit prime

    // Check combined size fits in 128 bits
    unsigned __int128 P = (unsigned __int128) p1 * p2 * p3;

    std::printf("\n=== 3-prime CRT test ===\n");
    std::printf("p1 = "); print_u128(p1); std::printf("\n");
    std::printf("p2 = "); print_u128(p2); std::printf("\n");
    std::printf("p3 = "); print_u128(p3); std::printf("\n");
    std::printf("Product = "); print_u128(P); std::printf("\n");

    // Generate a random test value < p1*p2*p3
    x_true =
        ((unsigned __int128)123456789123ULL << 64) ^
        (unsigned __int128)987654321ULL;  // just some ~120-bit number

    x_true %= P; // ensure x_true < product

    // Compute residues
    uint64_t r1 = (uint64_t)(x_true % p1);
    uint64_t r2 = (uint64_t)(x_true % p2);
    uint64_t r3 = (uint64_t)(x_true % p3);

    primes = {p1, p2, p3};
    residues = {r1, r2, r3};

    // Recover using multi-prime CRT
    unsigned __int128 x_rec = crt_combine_many(primes, residues);

    std::printf("x_true      = "); print_u128(x_true); std::printf("\n");
    std::printf("x_recovered = "); print_u128(x_rec);  std::printf("\n");

    if (x_true == x_rec)
        std::printf("3-prime CRT PASSED\n");
    else
        std::printf("3-prime CRT FAILED\n");

    return 0;
}
