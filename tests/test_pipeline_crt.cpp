// Compare GPU CRT output (64-bit pipeline) against CPU convolution mod 2^128.
//
// Coefficients can exceed 64 bits (e.g. L=2^15 with ~30-bit limbs peaks near 2^75),
// so the CPU reference uses unsigned __int128 — uint64_t / long long is not enough.
#include "../include/gpu_ntt.h"
#include "../include/ntt_limits.h"
#include "../include/config.h"

#include "gpuntt/ntt_merge/ntt_cpu.cuh"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

using namespace std;
using namespace gpuntt;
using Clock = chrono::steady_clock;

// GCC/Clang extended integer — no portable std::uint128_t yet.
using u128 = unsigned __int128;

static int passed = 0, failed = 0;

static void check(const char* name, bool ok) {
    if (ok) { printf("  PASS  %s\n", name); passed++; }
    else     { printf("  FAIL  %s\n", name); failed++; }
}

static double ms_since(const Clock::time_point& t0) {
    return chrono::duration<double, milli>(Clock::now() - t0).count();
}

static constexpr size_t SCHOOLBOOK_N_MAX = 4096;

#if !defined(NATIVE_HOST_LIMBS)

static vector<uint32_t> random_limbs(size_t n, uint64_t seed) {
    mt19937_64 rng(seed);
    vector<uint32_t> v(n);
    for (size_t i = 0; i < n; i++)
        v[i] = (uint32_t)(rng() % (1ULL << 30));
    return v;
}

static u128 hilo_to_u128(uint64_t hi, uint64_t lo) {
    return (static_cast<u128>(hi) << 64) | lo;
}

static uint64_t u128_hi(u128 x) { return static_cast<uint64_t>(x >> 64); }
static uint64_t u128_lo(u128 x) { return static_cast<uint64_t>(x); }

static bool exceeds_64_bits(u128 x) {
    return x > static_cast<u128>(UINT64_MAX);
}

// Garner CRT in host u128 — mirrors crt_combine_kernel.
static u128 host_crt_reference(
    const uint64_t* residues,
    const vector<uint64_t>& primes)
{
    u128 x = 0;
    u128 Mprefix = 1;
    uint64_t x_mod[NUM_MODULI] = {};

    auto mulmod128 = [](u128 a, u128 b, uint64_t p) -> uint64_t {
        return static_cast<uint64_t>((a * b) % p);
    };
    auto modinv = [](uint64_t a, uint64_t m) -> uint64_t {
        int64_t old_r = a, r = m;
        int64_t old_s = 1, s = 0;
        while (r != 0) {
            int64_t q = old_r / r;
            int64_t tmp = r; r = old_r - q * r; old_r = tmp;
            tmp = s; s = old_s - q * s; old_s = tmp;
        }
        return static_cast<uint64_t>((old_s % static_cast<int64_t>(m) + m) % m);
    };

    for (int j = 0; j < NUM_MODULI; j++) {
        uint64_t p   = primes[j];
        uint64_t r   = residues[j];
        uint64_t inv = (j == 0) ? 1ULL : modinv(static_cast<uint64_t>(Mprefix % p), p);

        uint64_t x_mod_p = x_mod[j];
        uint64_t t = (r >= x_mod_p) ? (r - x_mod_p) : (r + p - x_mod_p);
        uint64_t k_j = mulmod128(t, inv, p);

        for (int k = j + 1; k < NUM_MODULI; k++) {
            uint64_t pk = primes[k];
            uint64_t contrib = mulmod128(Mprefix % pk, k_j, pk);
            x_mod[k] += contrib;
            if (x_mod[k] >= pk) x_mod[k] -= pk;
        }

        x += Mprefix * k_j;
        Mprefix *= p;
    }
    return x;
}

// Padded schoolbook convolution; u128 unsigned wrap = mod 2^128.
static vector<u128> cpu_convolution_schoolbook(
    const vector<uint32_t>& A,
    const vector<uint32_t>& B,
    size_t N)
{
    vector<u128> a(N, 0), b(N, 0);
    for (size_t i = 0; i < A.size(); i++) a[i] = A[i];
    for (size_t i = 0; i < B.size(); i++) b[i] = B[i];

    vector<u128> c(N, 0);
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            if (i + j < N)
                c[i + j] += a[i] * b[j];
        }
    }
    return c;
}

// O(N log N) CPU reference via NTT + Garner CRT (same math as the GPU path).
static vector<u128> cpu_convolution_ntt(
    const vector<uint32_t>& A,
    const vector<uint32_t>& B,
    size_t N,
    const NTTPrecomputed& pre)
{
    vector<TestDataType> a(N, 0), b(N, 0);
    for (size_t i = 0; i < A.size(); i++) a[i] = static_cast<TestDataType>(A[i]);
    for (size_t i = 0; i < B.size(); i++) b[i] = static_cast<TestDataType>(B[i]);

    vector<vector<uint64_t>> residues(NUM_MODULI, vector<uint64_t>(N));
    vector<uint64_t> primes(NUM_MODULI);
    for (int m = 0; m < NUM_MODULI; m++)
        primes[m] = moduli[m];

    for (int m = 0; m < NUM_MODULI; m++) {
        NTTCPU<TestDataType> ntt(pre.params[m]);
        auto fa = ntt.ntt(a);
        auto fb = ntt.ntt(b);
        auto prod = ntt.mult(fa, fb);
        auto inv = ntt.intt(prod);
        for (size_t k = 0; k < N; k++)
            residues[m][k] = static_cast<uint64_t>(inv[k]);
    }

    vector<u128> out(N, 0);
    uint64_t r[NUM_MODULI];
    for (size_t k = 0; k < N; k++) {
        for (int m = 0; m < NUM_MODULI; m++)
            r[m] = residues[m][k];
        out[k] = host_crt_reference(r, primes);
    }
    return out;
}

static bool compare_crt_to_cpu(
    const vector<uint64_t>& gpu_hi,
    const vector<uint64_t>& gpu_lo,
    const vector<u128>& cpu,
    size_t L_C,
    const char* label)
{
    size_t mismatch = 0;
    for (size_t k = 0; k < L_C; k++) {
        u128 gpu_val = hilo_to_u128(gpu_hi[k], gpu_lo[k]);
        u128 cpu_val = cpu[k];
        if (gpu_val != cpu_val) {
            if (mismatch < 4) {
                printf("  [%s] coeff %zu: GPU=(%llu,%llu) CPU=(%llu,%llu)\n",
                       label, k,
                       (unsigned long long)u128_hi(gpu_val),
                       (unsigned long long)u128_lo(gpu_val),
                       (unsigned long long)u128_hi(cpu_val),
                       (unsigned long long)u128_lo(cpu_val));
            }
            mismatch++;
        }
    }
    if (mismatch > 4)
        printf("  [%s] ... %zu more mismatches\n", label, mismatch - 4);
    return mismatch == 0;
}

static void report_peak_coeff(const vector<u128>& cpu, size_t L_C, const char* label) {
    u128 peak = 0;
    size_t peak_k = 0;
    size_t wide_count = 0;
    for (size_t k = 0; k < L_C; k++) {
        if (cpu[k] > peak) {
            peak = cpu[k];
            peak_k = k;
        }
        if (exceeds_64_bits(cpu[k]))
            wide_count++;
    }
    printf("  [%s] peak coeff k=%zu value=(%llu,%llu)  coeffs with hi!=0: %zu / %zu\n",
           label, peak_k,
           (unsigned long long)u128_hi(peak),
           (unsigned long long)u128_lo(peak),
           wide_count, L_C);
}

// Schoolbook is O(N^2); use NTT CPU ref once N is large.
static void run_case(size_t L, uint64_t seed) {
    vector<uint32_t> A = random_limbs(L, seed);
    vector<uint32_t> B = random_limbs(L, seed + 1);

    size_t L_C = A.size() + B.size() - 1;
    size_t N = padded_ntt_size(A.size(), B.size());
    ensure_multiply_size_supported(A.size(), B.size());

    uint32_t* a_pinned = nullptr;
    uint32_t* b_pinned = nullptr;
    cudaMallocHost(&a_pinned, A.size() * sizeof(uint32_t));
    cudaMallocHost(&b_pinned, B.size() * sizeof(uint32_t));
    memcpy(a_pinned, A.data(), A.size() * sizeof(uint32_t));
    memcpy(b_pinned, B.data(), B.size() * sizeof(uint32_t));

    auto t_pre = Clock::now();
    NTTPrecomputed pre = precompute_ntt(N);
    upload_ntt_precomputed(pre);
    NTTContext ctx = allocate_ntt_context(pre, A.size(), B.size());
    printf("  L=%zu N=%zu precompute+alloc: %.1f ms\n", L, N, ms_since(t_pre));

    vector<uint64_t> gpu_hi, gpu_lo;
    vector<OutputLimbType> dummy_out;
    auto t_gpu = Clock::now();
    execute_ntt_multiply(ctx, a_pinned, b_pinned, dummy_out,
                         nullptr, &gpu_hi, &gpu_lo);
    printf("  L=%zu GPU through CRT: %.1f ms\n", L, ms_since(t_gpu));

    char label[64];
    snprintf(label, sizeof(label), "L=%zu", L);

    vector<u128> cpu;
    auto t_cpu = Clock::now();
    if (N <= SCHOOLBOOK_N_MAX) {
        cpu = cpu_convolution_schoolbook(A, B, N);
        printf("  L=%zu CPU schoolbook (u128): %.1f ms\n", L, ms_since(t_cpu));
    } else {
        cpu = cpu_convolution_ntt(A, B, N, pre);
        printf("  L=%zu CPU NTT+CRT (u128): %.1f ms\n", L, ms_since(t_cpu));
    }

    report_peak_coeff(cpu, L_C, label);

    bool ok = compare_crt_to_cpu(gpu_hi, gpu_lo, cpu, L_C, label);
    check(label, ok);

    cleanup_ntt_context(ctx);
    cleanup_ntt_precomputed(pre);
    cudaFreeHost(a_pinned);
    cudaFreeHost(b_pinned);
}

int main() {
    printf("=== pipeline CRT vs CPU convolution mod 2^128 (LIMB_BITS=%d) ===\n", LIMB_BITS);
    static_assert(sizeof(u128) == 16, "need 128-bit integer type for CPU reference");

#if LIMB_BITS != 64
    printf("SKIP: this test targets the 64-bit pipeline\n");
    return 0;
#else
    run_case(4, 100);
    run_case(8, 200);
    run_case(64, 300);
    run_case(256, 400);
    run_case(1ULL << 15, 600);  // N=2^16; peak coeff >> 64 bits

    printf("\n%d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
#endif
}

#else // NATIVE_HOST_LIMBS

#include "../include/wide_int.h"

static vector<uint64_t> random_limbs_u64(size_t n, uint64_t seed, bool narrow) {
    mt19937_64 rng(seed);
    vector<uint64_t> v(n);
    for (size_t i = 0; i < n; i++) {
        if (narrow)
            v[i] = rng() % (1ULL << 30);
        else if (i % 8 == 0)
            v[i] = 0;
        else if (i % 8 == 1)
            v[i] = 1;
        else if (i % 8 == 2)
            v[i] = UINT64_MAX;
        else
            v[i] = rng();
    }
    return v;
}

static U160 host_crt_reference_u160(
    const uint64_t* residues,
    const vector<uint64_t>& primes)
{
    U160 x{0, 0, 0};
    uint64_t M_hi = 0, M_lo = 1;
    uint64_t x_mod[NUM_MODULI] = {};

    auto mulmod = [](uint64_t a, uint64_t b, uint64_t p) -> uint64_t {
        return (uint64_t)(((unsigned __int128)a * b) % p);
    };
    auto modinv = [](uint64_t a, uint64_t m) -> uint64_t {
        int64_t old_r = a, r = m;
        int64_t old_s = 1, s = 0;
        while (r != 0) {
            int64_t q = old_r / r;
            int64_t tmp = r; r = old_r - q * r; old_r = tmp;
            tmp = s; s = old_s - q * s; old_s = tmp;
        }
        return (uint64_t)((old_s % (int64_t)m + m) % m);
    };

    for (int j = 0; j < NUM_MODULI; j++) {
        uint64_t p = primes[j];
        uint64_t r = residues[j];
        unsigned __int128 Mprefix = ((unsigned __int128)M_hi << 64) | M_lo;
        uint64_t inv = (j == 0) ? 1ULL : modinv((uint64_t)(Mprefix % p), p);

        uint64_t x_mod_p = x_mod[j];
        uint64_t t = (r >= x_mod_p) ? (r - x_mod_p) : (r + p - x_mod_p);
        uint64_t k_j = mulmod(t, inv, p);

        for (int k = j + 1; k < NUM_MODULI; k++) {
            uint64_t pk = primes[k];
            uint64_t contrib = mulmod((uint64_t)(Mprefix % pk), k_j, pk);
            x_mod[k] += contrib;
            if (x_mod[k] >= pk) x_mod[k] -= pk;
        }

        uint64_t tmp_lo, tmp_mid;
        uint32_t tmp_hi;
        mul128x64_host(M_hi, M_lo, k_j, tmp_lo, tmp_mid, tmp_hi);
        add160_host(x, tmp_lo, tmp_mid, tmp_hi);

        unsigned __int128 Mfull = ((unsigned __int128)M_hi << 64) | M_lo;
        Mfull *= p;
        M_lo = (uint64_t)Mfull;
        M_hi = (uint64_t)(Mfull >> 64);
    }
    return x;
}

static void add_u160(U160& a, const U160& b) {
    add160_host(a, b.lo, b.mid, b.hi);
}

static U160 mul_u64_u64(uint64_t a, uint64_t b) {
    unsigned __int128 p = (unsigned __int128)a * b;
    U160 out{(uint64_t)p, (uint64_t)(p >> 64), 0};
    return out;
}

static vector<U160> cpu_convolution_schoolbook_u64(
    const vector<uint64_t>& A,
    const vector<uint64_t>& B,
    size_t N)
{
    vector<U160> a(N, {0, 0, 0}), b(N, {0, 0, 0});
    for (size_t i = 0; i < A.size(); i++) a[i] = U160{A[i], 0, 0};
    for (size_t i = 0; i < B.size(); i++) b[i] = U160{B[i], 0, 0};

    vector<U160> c(N, {0, 0, 0});
    for (size_t i = 0; i < A.size(); i++) {
        for (size_t j = 0; j < B.size(); j++) {
            if (i + j < N) {
                U160 prod = mul_u64_u64(A[i], B[j]);
                add_u160(c[i + j], prod);
            }
        }
    }
    return c;
}

static vector<U160> cpu_convolution_ntt_u64(
    const vector<uint64_t>& A,
    const vector<uint64_t>& B,
    size_t N,
    const NTTPrecomputed& pre)
{
    vector<vector<uint64_t>> residues(NUM_MODULI, vector<uint64_t>(N));
    vector<uint64_t> primes(NUM_MODULI);
    for (int m = 0; m < NUM_MODULI; m++)
        primes[m] = moduli[m];

    for (int m = 0; m < NUM_MODULI; m++) {
        vector<TestDataType> a(N, 0), b(N, 0);
        for (size_t i = 0; i < A.size(); i++)
            a[i] = static_cast<TestDataType>(A[i] % moduli[m]);
        for (size_t i = 0; i < B.size(); i++)
            b[i] = static_cast<TestDataType>(B[i] % moduli[m]);

        NTTCPU<TestDataType> ntt(pre.params[m]);
        auto fa = ntt.ntt(a);
        auto fb = ntt.ntt(b);
        auto prod = ntt.mult(fa, fb);
        auto inv = ntt.intt(prod);
        for (size_t k = 0; k < N; k++)
            residues[m][k] = static_cast<uint64_t>(inv[k]);
    }

    vector<U160> out(N, {0, 0, 0});
    uint64_t r[NUM_MODULI];
    for (size_t k = 0; k < N; k++) {
        for (int m = 0; m < NUM_MODULI; m++)
            r[m] = residues[m][k];
        out[k] = host_crt_reference_u160(r, primes);
    }
    return out;
}

static bool compare_crt_u160_to_cpu(
    const vector<uint64_t>& gpu_lo,
    const vector<uint64_t>& gpu_mid,
    const vector<uint32_t>& gpu_hi,
    const vector<U160>& cpu,
    size_t L_C,
    const char* label)
{
    size_t mismatch = 0;
    for (size_t k = 0; k < L_C; k++) {
        U160 gpu_val{gpu_lo[k], gpu_mid[k], gpu_hi[k]};
        if (!u160_eq(gpu_val, cpu[k])) {
            if (mismatch < 4) {
                printf("  [%s] coeff %zu: GPU=(%llu,%llu,%u) CPU=(%llu,%llu,%u)\n",
                       label, k,
                       (unsigned long long)gpu_val.lo,
                       (unsigned long long)gpu_val.mid,
                       gpu_val.hi,
                       (unsigned long long)cpu[k].lo,
                       (unsigned long long)cpu[k].mid,
                       cpu[k].hi);
            }
            mismatch++;
        }
    }
    if (mismatch > 4)
        printf("  [%s] ... %zu more mismatches\n", label, mismatch - 4);
    return mismatch == 0;
}

static void run_case_native(size_t L, uint64_t seed, bool narrow) {
    vector<uint64_t> A = random_limbs_u64(L, seed, narrow);
    vector<uint64_t> B = random_limbs_u64(L, seed + 1, narrow);

    size_t L_C = A.size() + B.size() - 1;
    size_t N = padded_ntt_size(A.size(), B.size());
    ensure_multiply_size_supported(A.size(), B.size());

    uint64_t* a_pinned = nullptr;
    uint64_t* b_pinned = nullptr;
    cudaMallocHost(&a_pinned, A.size() * sizeof(uint64_t));
    cudaMallocHost(&b_pinned, B.size() * sizeof(uint64_t));
    memcpy(a_pinned, A.data(), A.size() * sizeof(uint64_t));
    memcpy(b_pinned, B.data(), B.size() * sizeof(uint64_t));

    auto t_pre = Clock::now();
    NTTPrecomputed pre = precompute_ntt(N);
    upload_ntt_precomputed(pre);
    NTTContext ctx = allocate_ntt_context(pre, A.size(), B.size());
    printf("  L=%zu N=%zu precompute+alloc: %.1f ms\n", L, N, ms_since(t_pre));

    vector<uint64_t> gpu_lo, gpu_mid;
    vector<uint32_t> gpu_hi;
    vector<OutputLimbType> dummy_out;
    auto t_gpu = Clock::now();
    execute_ntt_multiply(ctx, a_pinned, b_pinned, dummy_out,
                         nullptr, nullptr, &gpu_lo, &gpu_mid, &gpu_hi);
    printf("  L=%zu GPU through CRT: %.1f ms\n", L, ms_since(t_gpu));

    char label[64];
    snprintf(label, sizeof(label), "L=%zu", L);

    vector<U160> cpu;
    auto t_cpu = Clock::now();
    cpu = cpu_convolution_ntt_u64(A, B, N, pre);
    printf("  L=%zu CPU NTT+CRT (U160): %.1f ms\n", L, ms_since(t_cpu));

    bool ok = compare_crt_u160_to_cpu(gpu_lo, gpu_mid, gpu_hi, cpu, L_C, label);
    check(label, ok);

    cleanup_ntt_context(ctx);
    cleanup_ntt_precomputed(pre);
    cudaFreeHost(a_pinned);
    cudaFreeHost(b_pinned);
}

int main() {
    printf("=== pipeline CRT vs CPU convolution (64native, U160) ===\n");

    run_case_native(4, 100, false);
    run_case_native(8, 200, false);
    run_case_native(64, 300, false);
    run_case_native(256, 400, true);
    run_case_native(1ULL << 15, 600, true);

    printf("\n%d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}

#endif // NATIVE_HOST_LIMBS
