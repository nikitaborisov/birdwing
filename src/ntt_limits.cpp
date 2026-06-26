#include "ntt_limits.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <climits>
#include <iostream>
#include <limits>

using namespace std;

extern vector<TestDataTypeUint> moduli;
extern vector<TestDataTypeUint> roots_of_unity_max;

static int valuation2_u64(uint64_t n)
{
    int k = 0;
    while ((n & 1) == 0) {
        k++;
        n >>= 1;
    }
    return k;
}

int max_logN_for_prime(uint64_t p)
{
    return valuation2_u64(p - 1);
}

static TestDataTypeUint mod_square(TestDataTypeUint r, TestDataTypeUint p)
{
    if constexpr (sizeof(TestDataTypeUint) == 8) {
        return (TestDataTypeUint)(((__uint128_t)r * r) % p);
    }
    return (TestDataTypeUint)(((uint64_t)r * r) % p);
}

static int max_root_order(TestDataTypeUint root, TestDataTypeUint p)
{
    TestDataTypeUint r = root % p;
    if (r == 0)
        return 0;

    int k = 0;
    while (k < 62) {
        if (r == 1)
            return k;
        r = mod_square(r, p);
        k++;
    }
    return k;
}

int max_root_logN(TestDataTypeUint root, TestDataTypeUint p)
{
    return max_root_order(root, p);
}

static unsigned __int128 crt_modulus_product()
{
    unsigned __int128 M = 1;
    for (int i = 0; i < NUM_MODULI; i++)
        M *= (unsigned __int128)moduli[i];
    return M;
}

static long double crt_modulus_product_ld()
{
    long double M = 1.0L;
    for (int i = 0; i < NUM_MODULI; i++)
        M *= (long double)moduli[i];
    return M;
}

static unsigned __int128 host_limb_product_sq()
{
    const uint64_t limb_max = (INPUT_LIMB_BITS == 64)
        ? UINT64_MAX
        : (uint64_t)OUTPUT_LIMB_MASK;
    return (unsigned __int128)limb_max * limb_max;
}

static unsigned __int128 max_convolution_coefficient(size_t L_A, size_t L_B)
{
    const size_t terms = min(L_A, L_B);
    return (unsigned __int128)terms * host_limb_product_sq();
}

static long double max_convolution_coefficient_ld(size_t L_A, size_t L_B)
{
    const long double terms = (long double)min(L_A, L_B);
    const long double limb_max = (INPUT_LIMB_BITS == 64)
        ? (long double)UINT64_MAX
        : (long double)OUTPUT_LIMB_MASK;
    return terms * limb_max * limb_max;
}

static bool crt_bounds_use_long_double()
{
#if INPUT_LIMB_BITS == 64
    return true;
#else
    return false;
#endif
}

static bool ntt_logN_supported(int logN, string* why)
{
    for (int i = 0; i < NUM_MODULI; i++) {
        const TestDataTypeUint p = moduli[i];
        const int by_prime = max_logN_for_prime(p);
        const int by_root = max_root_logN(roots_of_unity_max[i], p);
        const int max_log = min(by_prime, by_root);
        if (logN > max_log) {
            if (why) {
                *why = "NTT size N=2^" + to_string(logN) + " exceeds modulus " +
                       to_string(i) + " (p=" + to_string(p) +
                       ", max 2^" + to_string(max_log) + ")";
            }
            return false;
        }
    }
    return true;
}

int max_supported_logN()
{
    int limit = numeric_limits<int>::max();
    for (int i = 0; i < NUM_MODULI; i++) {
        const TestDataTypeUint p = moduli[i];
        const int by_prime = max_logN_for_prime(p);
        const int by_root = max_root_logN(roots_of_unity_max[i], p);
        limit = min({limit, by_prime, by_root});
    }
    return limit;
}

size_t max_supported_N()
{
    const int logN = max_supported_logN();
    if (logN < 0)
        return 0;
    return size_t(1) << logN;
}

size_t padded_ntt_size(size_t L_A, size_t L_B)
{
    const size_t L_C = L_A + L_B - 1;
    size_t N = 1;
    while (N < L_C)
        N <<= 1;
    return N;
}

static size_t max_limb_count_by_crt()
{
    if (crt_bounds_use_long_double()) {
        const long double M = crt_modulus_product_ld();
        const long double limb_sq = (long double)UINT64_MAX * (long double)UINT64_MAX;
        const long double max_L = M / limb_sq;
        if (max_L > (long double)numeric_limits<size_t>::max())
            return numeric_limits<size_t>::max();
        return static_cast<size_t>(max_L);
    }

    const unsigned __int128 M = crt_modulus_product();
    const unsigned __int128 limb_sq = host_limb_product_sq();
    if (limb_sq == 0)
        return 0;

    const unsigned __int128 max_L = (M - 1) / limb_sq;
    if (max_L > numeric_limits<size_t>::max())
        return numeric_limits<size_t>::max();
    return static_cast<size_t>(max_L);
}

#if !defined(NATIVE_HOST_LIMBS)
static unsigned __int128 worst_case_int64_segment_carry(size_t L)
{
    const unsigned __int128 max_sq =
        (unsigned __int128)OUTPUT_LIMB_MASK * OUTPUT_LIMB_MASK;
    const unsigned __int128 coeff = (unsigned __int128)L * max_sq;

    unsigned __int128 carry = 0;
    for (size_t i = 0; i < CARRY_SEG; i++) {
        const unsigned __int128 t = coeff + carry;
        carry = t >> OUTPUT_LIMB_BITS;
    }
    return carry;
}
#endif

size_t max_limb_count_for_int64_segment_carry()
{
#if defined(NATIVE_HOST_LIMBS)
    return numeric_limits<size_t>::max();
#else
    static const size_t kMax = []() {
        size_t lo = 1;
        size_t hi = size_t(1) << 33;
        while (lo < hi) {
            const size_t mid = lo + (hi - lo + 1) / 2;
            if (worst_case_int64_segment_carry(mid) <=
                (unsigned __int128)INT64_MAX) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        return lo;
    }();
    return kMax;
#endif
}

#if !defined(NATIVE_HOST_LIMBS)
static_assert(CARRY_SEG > 0, "CARRY_SEG must be positive");
#endif

size_t max_supported_limb_count()
{
    const int max_log = max_supported_logN();
    if (max_log <= 0)
        return 0;

    const size_t max_by_ntt = size_t(1) << (max_log - 1);
    const size_t max_by_crt = max_limb_count_by_crt();
#if defined(NATIVE_HOST_LIMBS)
    return min(max_by_ntt, max_by_crt);
#else
    return min({max_by_ntt, max_by_crt, max_limb_count_for_int64_segment_carry()});
#endif
}

bool crt_coefficient_bound_satisfied(size_t L_A, size_t L_B, string* why)
{
    if (L_A == 0 || L_B == 0) {
        if (why)
            *why = "operand limb count must be positive";
        return false;
    }

    if (crt_bounds_use_long_double()) {
        const long double coeff_max = max_convolution_coefficient_ld(L_A, L_B);
        const long double M = crt_modulus_product_ld();
        if (coeff_max < M)
            return true;
    } else {
        const unsigned __int128 coeff_max = max_convolution_coefficient(L_A, L_B);
        const unsigned __int128 M = crt_modulus_product();
        if (coeff_max < M)
            return true;
    }

    if (why) {
        *why = "max convolution coefficient for L_A=" + to_string(L_A) +
               ", L_B=" + to_string(L_B) +
               " (terms=" + to_string(min(L_A, L_B)) +
               ", limb=" + to_string(INPUT_LIMB_BITS) +
               "-bit) exceeds CRT modulus product";
    }
    return false;
}

bool ntt_size_supported(size_t N, string* why)
{
    if (N == 0) {
        if (why)
            *why = "NTT size N must be positive";
        return false;
    }

    if ((N & (N - 1)) != 0) {
        if (why) {
            *why = "NTT size N=" + to_string(N) + " is not a power of two";
        }
        return false;
    }

    const int logN = static_cast<int>(lround(log2(static_cast<double>(N))));
    if ((size_t(1) << logN) != N) {
        if (why)
            *why = "NTT size N=" + to_string(N) + " is not an exact power of two";
        return false;
    }

    return ntt_logN_supported(logN, why);
}

bool multiply_size_supported(size_t L_A, size_t L_B, string* why)
{
    if (L_A == 0 || L_B == 0) {
        if (why)
            *why = "operand limb count must be positive";
        return false;
    }

    const size_t N = padded_ntt_size(L_A, L_B);
    if (!ntt_size_supported(N, why))
        return false;

    if (!crt_coefficient_bound_satisfied(L_A, L_B, why))
        return false;

#if !defined(NATIVE_HOST_LIMBS)
    const size_t L = min(L_A, L_B);
    if (L > max_limb_count_for_int64_segment_carry()) {
        if (why) {
            *why = "L=" + to_string(L) +
                   " exceeds int64 segment-carry range (max L=" +
                   to_string(max_limb_count_for_int64_segment_carry()) + ")";
        }
        return false;
    }
#endif

    return true;
}

void ensure_ntt_size_supported(size_t N)
{
    string why;
    if (!ntt_size_supported(N, &why)) {
        cerr << "[NTT] Unsupported size: " << why << "\n";
        cerr << "[NTT] Max supported N=2^" << max_supported_logN()
             << " (" << max_supported_N() << "), max L=2^"
             << (max_supported_logN() - 1) << " (" << max_supported_limb_count()
             << ")\n";
        exit(1);
    }
}

void ensure_multiply_size_supported(size_t L_A, size_t L_B)
{
    string why;
    if (!multiply_size_supported(L_A, L_B, &why)) {
        cerr << "[NTT] Unsupported multiply: " << why << "\n";
        cerr << "[NTT] Max supported N=2^" << max_supported_logN()
             << " (" << max_supported_N() << "), max L (square)=2^"
             << (max_supported_logN() - 1) << " (" << max_supported_limb_count()
             << "), max L (CRT only)=" << max_limb_count_by_crt() << "\n";
        exit(1);
    }
}
