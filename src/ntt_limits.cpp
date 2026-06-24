#include "ntt_limits.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
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

size_t max_supported_limb_count()
{
    const int max_log = max_supported_logN();
    if (max_log <= 0)
        return 0;
    // L = 2^k needs N = 2^(k+1); require k+1 <= max_log.
    return size_t(1) << (max_log - 1);
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
