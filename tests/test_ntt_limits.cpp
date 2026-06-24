#include "../include/ntt_limits.h"
#include "../include/config.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

// Tables mirrored from src/gpu_ntt.cu for integration tests.
#if LIMB_BITS == 64
vector<TestDataTypeUint> moduli = {0x6723cbb800001ULL, 0x6723cb6800001ULL};
vector<TestDataTypeUint> roots_of_unity_max = {622482970039944ULL, 1317955505843176ULL};
#else
vector<TestDataTypeUint> moduli = {0x2d000001, 0x23800001, 0x26800001};
vector<TestDataTypeUint> roots_of_unity_max = {663, 721, 19};
#endif

#define GREEN "\033[1;32m"
#define RED   "\033[1;31m"
#define RESET "\033[0m"

static int failures = 0;

static void check(bool cond, const string& msg)
{
    if (cond) {
        cout << GREEN << "[PASS] " << RESET << msg << "\n";
    } else {
        cout << RED << "[FAIL] " << RESET << msg << "\n";
        failures++;
    }
}

static void check_eq_u64(uint64_t got, uint64_t want, const string& msg)
{
    check(got == want, msg + " (got " + to_string(got) + ", want " + to_string(want) + ")");
}

static void check_eq_int(int got, int want, const string& msg)
{
    check(got == want, msg + " (got " + to_string(got) + ", want " + to_string(want) + ")");
}

// ---------------------------------------------------------------------------
// Pure helpers (no dependency on runtime moduli tables)
// ---------------------------------------------------------------------------

static void test_max_logN_for_prime()
{
    cout << "\n=== max_logN_for_prime ===\n";

#if LIMB_BITS == 64
    check_eq_int(max_logN_for_prime(0x6723cbb800001ULL), 23,
                 "64-bit modulus 0 has v2(p-1)=23");
    check_eq_int(max_logN_for_prime(0x6723cb6800001ULL), 23,
                 "64-bit modulus 1 has v2(p-1)=23");
#else
    check_eq_int(max_logN_for_prime(0x2d000001ULL), 24,
                 "32-bit prime 0x2d000001 has v2(p-1)=24");
    check_eq_int(max_logN_for_prime(0x23800001ULL), 23,
                 "32-bit prime 0x238000001 has v2(p-1)=23");
    check_eq_int(max_logN_for_prime(0x26800001ULL), 23,
                 "32-bit prime 0x268000001 has v2(p-1)=23");
#endif
}

static void test_max_root_logN()
{
    cout << "\n=== max_root_logN ===\n";

#if LIMB_BITS == 64
    check_eq_int(max_root_logN(622482970039944ULL, 0x6723cbb800001ULL), 23,
                 "64-bit root 0 has order 2^23");
    check_eq_int(max_root_logN(1317955505843176ULL, 0x6723cb6800001ULL), 23,
                 "64-bit root 1 has order 2^23");
#else
    check_eq_int(max_root_logN(663, 0x2d000001ULL), 23,
                 "root 663 has order 2^23 mod 0x2d000001");
    check_eq_int(max_root_logN(721, 0x23800001ULL), 23,
                 "root 721 has order 2^23 mod 0x23800001");
    check_eq_int(max_root_logN(19, 0x26800001ULL), 23,
                 "root 19 has order 2^23 mod 0x26800001");
#endif
}

// ---------------------------------------------------------------------------
// padded_ntt_size
// ---------------------------------------------------------------------------

static void test_padded_ntt_size()
{
    cout << "\n=== padded_ntt_size ===\n";

    check_eq_u64(padded_ntt_size(1, 1), 1, "1 x 1 -> N=1");
    check_eq_u64(padded_ntt_size(4, 4), 8, "4 x 4 -> N=8");
    check_eq_u64(padded_ntt_size(5, 5), 16, "5 x 5 -> N=16");

    const size_t L = size_t(1) << 22;
    check_eq_u64(padded_ntt_size(L, L), size_t(1) << 23,
                 "L=2^22 square multiply -> N=2^23");

    const size_t L_big = size_t(1) << 23;
    check_eq_u64(padded_ntt_size(L_big, L_big), size_t(1) << 24,
                 "L=2^23 square multiply -> N=2^24");
}

// ---------------------------------------------------------------------------
// Integration with configured moduli / roots (from gpu_ntt.cu)
// ---------------------------------------------------------------------------

static void test_configured_limits()
{
    cout << "\n=== configured moduli / roots ===\n";

    check_eq_int(max_supported_logN(), 23, "max_supported_logN()==23");
    check_eq_u64(max_supported_N(), size_t(1) << 23, "max_supported_N()==2^23");
    check_eq_u64(max_supported_limb_count(), size_t(1) << 22,
                 "max_supported_limb_count()==2^22");
}

static void test_crt_coefficient_bound()
{
    cout << "\n=== crt_coefficient_bound_satisfied ===\n";

    string why;

    const size_t L_ok = size_t(1) << 22;
    check(crt_coefficient_bound_satisfied(L_ok, L_ok, &why),
          "L=2^22 square multiply satisfies CRT bound");

#if LIMB_BITS == 32
    const size_t L_crt_only = size_t(1) << 23;
    check(crt_coefficient_bound_satisfied(L_crt_only, L_crt_only, &why),
          "L=2^23 satisfies CRT bound alone (32-bit moduli)");
    const size_t L_bad = size_t(1) << 24;
#else
    const size_t L_bad = size_t(1) << 38;
#endif
    check(!crt_coefficient_bound_satisfied(L_bad, L_bad, &why),
          "L exceeds CRT bound");
    if (!crt_coefficient_bound_satisfied(L_bad, L_bad, &why)) {
        check(why.find("CRT modulus product") != string::npos,
              "CRT rejection message mentions modulus product");
    }

    check(crt_coefficient_bound_satisfied(L_bad, 1, &why),
          "rectangular multiply: short operand keeps CRT bound small");
}

static void test_multiply_size_supported()
{
    cout << "\n=== multiply_size_supported ===\n";

    string why;

    const size_t L_ok = size_t(1) << 22;
    check(multiply_size_supported(L_ok, L_ok, &why),
          "L=2^22 square multiply supported");

    const size_t L_bad = size_t(1) << 23;
    check(!multiply_size_supported(L_bad, L_bad, &why),
          "L=2^23 square multiply rejected (NTT and/or CRT)");
}

static void test_ntt_size_supported_accept()
{
    cout << "\n=== ntt_size_supported (accept) ===\n";

    string why;

    for (int logN = 1; logN <= 23; logN++) {
        const size_t N = size_t(1) << logN;
        check(ntt_size_supported(N, &why),
              "N=2^" + to_string(logN) + " supported");
    }

    const size_t L_max = size_t(1) << 22;
    const size_t N_edge = padded_ntt_size(L_max, L_max);
    check(ntt_size_supported(N_edge, &why),
          "padded N at L=2^22 is supported");
}

static void test_ntt_size_supported_reject()
{
    cout << "\n=== ntt_size_supported (reject) ===\n";

    string why;

    check(!ntt_size_supported(0, &why), "N=0 rejected");
    check(!ntt_size_supported(3, &why), "N=3 rejected (not power of two)");
    check(!ntt_size_supported(1u << 24, &why), "N=2^24 rejected");

    const size_t L_too_big = size_t(1) << 23;
    const size_t N_too_big = padded_ntt_size(L_too_big, L_too_big);
    check(!ntt_size_supported(N_too_big, &why),
          "padded N at L=2^23 rejected (needs N=2^24)");

    if (!ntt_size_supported(1u << 24, &why)) {
        check(why.find("2^24") != string::npos || why.find("24") != string::npos,
              "rejection message mentions oversize N");
    }
}

// ---------------------------------------------------------------------------
// Benchmark L_arg convention (L < 64 means 1<<L limbs)
// ---------------------------------------------------------------------------

static size_t resolve_limb_count(size_t L_arg)
{
    if (L_arg < 64)
        return size_t(1) << L_arg;
    return L_arg;
}

static void test_l_arg_convention()
{
    cout << "\n=== L_arg convention vs limits ===\n";

    string why;

    const size_t L_ok = resolve_limb_count(22);
    check_eq_u64(L_ok, size_t(1) << 22, "L_arg=22 -> L=2^22");
    check(multiply_size_supported(L_ok, L_ok, &why),
          "L_arg=22 fits in configured multiply limits");

    const size_t L_bad = resolve_limb_count(23);
    check_eq_u64(L_bad, size_t(1) << 23, "L_arg=23 -> L=2^23");
    check(!multiply_size_supported(L_bad, L_bad, &why),
          "L_arg=23 exceeds configured multiply limits");
}

int main()
{
    cout << "=== ntt_limits tests (LIMB_BITS=" << LIMB_BITS << ") ===\n";

    test_max_logN_for_prime();
    test_max_root_logN();
    test_padded_ntt_size();
    test_crt_coefficient_bound();
    test_multiply_size_supported();
    test_configured_limits();
    test_ntt_size_supported_accept();
    test_ntt_size_supported_reject();
    test_l_arg_convention();

    cout << "\n";
    if (failures == 0) {
        cout << GREEN << "All ntt_limits tests passed." << RESET << "\n";
        return 0;
    }

    cout << RED << failures << " test(s) failed." << RESET << "\n";
    return 1;
}
