// test_crt_gpu.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cassert>
#include "crt_gpu.h"
#include "config.h"

using namespace std;

static int passed = 0, failed = 0;

void check(const char* name, bool ok) {
    if (ok) { printf("  PASS  %s\n", name); passed++; }
    else     { printf("  FAIL  %s\n", name); failed++; }
}

// ----------------------------------------------------------------
// Host-side Garner reconstruction using __int128 — reference answer
// ----------------------------------------------------------------
unsigned __int128 host_garner(
    const uint64_t* residues,       // residues[j] = x mod p_j
    const uint64_t* primes,
    int num_moduli)
{
    unsigned __int128 x = 0;
    unsigned __int128 M = 1;

    // same Garner algorithm as the kernel, but in host __int128
    uint64_t x_mod[NUM_MODULI] = {};

    for (int j = 0; j < num_moduli; j++) {
        uint64_t p   = primes[j];
        uint64_t r   = residues[j];
        uint64_t inv = (j == 0) ? 1ULL : 0ULL; // recomputed below

        // recompute inv: modinv of M mod p_j
        // for the test we just reuse compute_garner_params' output
        uint64_t x_mod_p = x_mod[j];
        uint64_t t = (r >= x_mod_p) ? (r - x_mod_p) : (r + p - x_mod_p);

        // k_j = t * inv mod p — we'll pass precomputed inv in
        (void)inv; // handled via garner params below
        (void)t;

        M *= p;
    }
    return x; // placeholder — see note below
}

// ----------------------------------------------------------------
// Simpler reference: just do x = CRT directly via __int128 arithmetic
// Given residues and primes, reconstruct x in [0, M)
// ----------------------------------------------------------------
unsigned __int128 host_crt_reference(
    const vector<uint64_t>& residues,
    const vector<uint64_t>& primes)
{
    // Use the same Garner algorithm as the kernel
    unsigned __int128 x = 0;
    unsigned __int128 M = 1;
    uint64_t x_mod[NUM_MODULI] = {};

    // precompute invs
    // inv[j] = (p_0 * p_1 * ... * p_{j-1})^{-1} mod p_j
    auto mulmod128 = [](unsigned __int128 a, unsigned __int128 b, uint64_t p) -> uint64_t {
        return (uint64_t)((a * b) % p);
    };
    auto modinv = [](uint64_t a, uint64_t m) -> uint64_t {
        // extended Euclidean
        int64_t old_r = a, r = m;
        int64_t old_s = 1, s = 0;
        while (r != 0) {
            int64_t q = old_r / r;
            int64_t tmp = r; r = old_r - q * r; old_r = tmp;
            tmp = s; s = old_s - q * s; old_s = tmp;
        }
        return (uint64_t)((old_s % (int64_t)m + m) % m);
    };

    unsigned __int128 Mprefix = 1;
    for (int j = 0; j < NUM_MODULI; j++) {
        uint64_t p   = primes[j];
        uint64_t r   = residues[j];
        uint64_t inv = (j == 0) ? 1ULL : modinv((uint64_t)(Mprefix % p), p);

        uint64_t x_mod_p = x_mod[j];
        uint64_t t = (r >= x_mod_p) ? (r - x_mod_p) : (r + p - x_mod_p);
        uint64_t k_j = mulmod128(t, inv, p);

        for (int k = j + 1; k < NUM_MODULI; k++) {
            uint64_t pk = primes[k];
            uint64_t contrib = mulmod128(Mprefix % pk, k_j, pk);
            x_mod[k] += contrib;
            if (x_mod[k] >= pk) x_mod[k] -= pk;
        }

        x += (unsigned __int128)Mprefix * k_j;
        Mprefix *= p;
    }
    return x;
}

// ----------------------------------------------------------------
// Upload known residues to device and run CRT
// Returns reconstructed (hi, lo) per coefficient
// ----------------------------------------------------------------
struct CRTResult { uint64_t hi, lo; };

vector<CRTResult> run_crt(
    const CRTGarnerParams& garner,
    const vector<vector<uint64_t>>& residues_per_modulus,  // [modulus][coeff]
    int N)
{
    // allocate and upload residue buffers as TestDataTypeUint
    vector<TestDataTypeUint*> c_dev(NUM_MODULI);
    for (int j = 0; j < NUM_MODULI; j++) {
        cudaMalloc(&c_dev[j], N * sizeof(TestDataTypeUint));
        // convert uint64_t -> TestDataTypeUint for upload
        vector<TestDataTypeUint> tmp(N);
        for (int i = 0; i < N; i++)
            tmp[i] = (TestDataTypeUint)residues_per_modulus[j][i];
        cudaMemcpy(c_dev[j], tmp.data(), N * sizeof(TestDataTypeUint),
                   cudaMemcpyHostToDevice);
    }

    upload_garner_params(garner);
    upload_residue_ptrs(c_dev);

    uint64_t *d_C_hi, *d_C_lo;
    cudaMalloc(&d_C_hi, N * sizeof(uint64_t));
    cudaMalloc(&d_C_lo, N * sizeof(uint64_t));

    crt_combine_gpu(d_C_hi, d_C_lo, N);

    vector<uint64_t> h_hi(N), h_lo(N);
    cudaMemcpy(h_hi.data(), d_C_hi, N * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_lo.data(), d_C_lo, N * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    for (int j = 0; j < NUM_MODULI; j++) cudaFree(c_dev[j]);
    cudaFree(d_C_hi);
    cudaFree(d_C_lo);

    vector<CRTResult> out(N);
    for (int i = 0; i < N; i++) out[i] = {h_hi[i], h_lo[i]};
    return out;
}

// ----------------------------------------------------------------
// Tests
// ----------------------------------------------------------------

// Reconstruct value from (hi, lo)
unsigned __int128 from_hilo(uint64_t hi, uint64_t lo) {
    return ((unsigned __int128)hi << 64) | lo;
}

void test_single_known_value(const CRTGarnerParams& garner) {
    // x = 12, compute residues manually, check CRT recovers 12
    uint64_t x = 12;
    int N = 1;

    vector<vector<uint64_t>> residues(NUM_MODULI, vector<uint64_t>(N));
    for (int j = 0; j < NUM_MODULI; j++)
        residues[j][0] = x % garner.primes[j];

    auto out = run_crt(garner, residues, N);

    unsigned __int128 recovered = from_hilo(out[0].hi, out[0].lo);
    check("single known value (x=12)", recovered == x);
}

void test_zero(const CRTGarnerParams& garner) {
    int N = 4;
    vector<vector<uint64_t>> residues(NUM_MODULI, vector<uint64_t>(N, 0));

    auto out = run_crt(garner, residues, N);

    bool ok = true;
    for (int i = 0; i < N; i++)
        ok &= (out[i].hi == 0 && out[i].lo == 0);
    check("all zeros", ok);
}

void test_max_32bit(const CRTGarnerParams& garner) {
    // x = 2^32 - 1, largest single 32-bit value
    uint64_t x = 0xFFFFFFFFULL;
    int N = 1;

    vector<vector<uint64_t>> residues(NUM_MODULI, vector<uint64_t>(N));
    for (int j = 0; j < NUM_MODULI; j++)
        residues[j][0] = x % garner.primes[j];

    auto out = run_crt(garner, residues, N);
    unsigned __int128 recovered = from_hilo(out[0].hi, out[0].lo);
    check("max 32-bit value (2^32-1)", recovered == (unsigned __int128)x);
}

void test_large_value(const CRTGarnerParams& garner) {
    // x close to max reconstructible: N * (2^32)^2 = 2^26 * 2^64 = 2^90
    // use x = 2^89 as a stress test
    unsigned __int128 x = (unsigned __int128)1 << 89;

    // check x < M = p0 * p1 (... * p2)
    unsigned __int128 M = 1;
    for (int j = 0; j < NUM_MODULI; j++) M *= garner.primes[j];
    if (x >= M) {
        printf("  SKIP  large value test (x >= M for these primes)\n");
        return;
    }

    int N = 1;
    vector<vector<uint64_t>> residues(NUM_MODULI, vector<uint64_t>(N));
    for (int j = 0; j < NUM_MODULI; j++)
        residues[j][0] = (uint64_t)(x % garner.primes[j]);

    auto out = run_crt(garner, residues, N);
    unsigned __int128 recovered = from_hilo(out[0].hi, out[0].lo);
    check("large value (2^89)", recovered == x);
}

void test_multiple_coefficients(const CRTGarnerParams& garner) {
    // Several known values in one batch
    vector<uint64_t> values = {0, 1, 12, 255, 65535, 0xFFFFFFFFULL};
    int N = values.size();

    vector<vector<uint64_t>> residues(NUM_MODULI, vector<uint64_t>(N));
    for (int j = 0; j < NUM_MODULI; j++)
        for (int i = 0; i < N; i++)
            residues[j][i] = values[i] % garner.primes[j];

    auto out = run_crt(garner, residues, N);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        unsigned __int128 recovered = from_hilo(out[i].hi, out[i].lo);
        ok &= (recovered == (unsigned __int128)values[i]);
    }
    check("multiple coefficients batch", ok);
}

void test_against_host_reference(const CRTGarnerParams& garner) {
    // Generate random-ish residues, check GPU matches host_crt_reference
    int N = 64;
    vector<uint64_t> primes_vec(garner.primes, garner.primes + NUM_MODULI);

    vector<vector<uint64_t>> residues(NUM_MODULI, vector<uint64_t>(N));
    for (int j = 0; j < NUM_MODULI; j++)
        for (int i = 0; i < N; i++)
            residues[j][i] = ((uint64_t)(i * 7 + j * 13 + 1)) % garner.primes[j];

    auto out = run_crt(garner, residues, N);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        vector<uint64_t> res_i(NUM_MODULI);
        for (int j = 0; j < NUM_MODULI; j++) res_i[j] = residues[j][i];

        unsigned __int128 expected  = host_crt_reference(res_i, primes_vec);
        unsigned __int128 recovered = from_hilo(out[i].hi, out[i].lo);
        ok &= (recovered == expected);
    }
    check("GPU matches host reference (N=64)", ok);
}

int main() {
    printf("=== crt_gpu tests (LIMB_BITS=%d) ===\n", LIMB_BITS);

    // Use your actual moduli from config
    #if LIMB_BITS == 64
    // 62-bit NTT-friendly primes: p = k * 2^M + 1, M >= 23
    vector<TestDataTypeUint> moduli_vec = {0x6723cbb800001, 0x6723cb6800001};
    vector<TestDataTypeUint> roots_of_unity_2_23 = {11, 6};
    #else
    vector<TestDataTypeUint> moduli_vec = {0x23800001, 0x26800001, 0x2d000001};
    vector<TestDataTypeUint> roots_of_unity_2_23 = {663, 721, 19};
    #endif
    CRTGarnerParams garner = compute_garner_params(moduli_vec);

    test_zero(garner);
    test_single_known_value(garner);
    test_max_32bit(garner);
    test_large_value(garner);
    test_multiple_coefficients(garner);
    test_against_host_reference(garner);

    printf("\n%d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}