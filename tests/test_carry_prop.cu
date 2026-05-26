// test_carry_prop.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cstring>
#include "carry_prop.h"
#include "config.h"

using namespace std;

static int passed = 0, failed = 0;

void check(const char* name, bool ok) {
    if (ok) { printf("  PASS  %s\n", name); passed++; }
    else     { printf("  FAIL  %s\n", name); failed++; }
}

// ----------------------------------------------------------------
// CPU reference carry propagation
// Given 128-bit values per coefficient, reduce mod M (centered),
// then carry-propagate into LIMB_BITS-wide limbs.
// ----------------------------------------------------------------
vector<TestDataTypeUint> cpu_carry_prop(
    const vector<uint64_t>& C_hi,
    const vector<uint64_t>& C_lo,
    unsigned __int128 M,
    unsigned __int128 M_half,
    size_t N)
{
    vector<TestDataTypeUint> out(N, 0);
    __int128 carry = 0;

    for (size_t i = 0; i < N; i++) {
        __int128 val = ((__uint128_t)C_hi[i] << 64) | C_lo[i];
        if (val > (__int128)M_half) val -= (__int128)M;

        __int128 temp = val + carry;
        out[i] = (TestDataTypeUint)(temp & LIMB_MASK);
        carry  = temp >> LIMB_BITS;
    }
    return out;
}

// ----------------------------------------------------------------
// Run the three carry prop kernels and return result on host
// ----------------------------------------------------------------
vector<TestDataTypeUint> run_carry_prop(
    const vector<uint64_t>& h_C_hi,
    const vector<uint64_t>& h_C_lo,
    unsigned __int128 M,
    unsigned __int128 M_half,
    size_t N)
{
    uint64_t *d_C_hi, *d_C_lo;
    cudaMalloc(&d_C_hi, N * sizeof(uint64_t));
    cudaMalloc(&d_C_lo, N * sizeof(uint64_t));
    cudaMemcpy(d_C_hi, h_C_hi.data(), N * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_lo, h_C_lo.data(), N * sizeof(uint64_t), cudaMemcpyHostToDevice);

    TestDataTypeUint* d_out;
    cudaMalloc(&d_out, N * sizeof(TestDataTypeUint));
    cudaMemset(d_out, 0, N * sizeof(TestDataTypeUint));

    size_t num_segs = (N + CARRY_SEG - 1) / CARRY_SEG;
    int64_t* d_seg_carry;
    cudaMalloc(&d_seg_carry, num_segs * sizeof(int64_t));

    carry_intra_segment_kernel<<<num_segs, 1>>>(
        d_C_hi, d_C_lo, d_out, d_seg_carry, N, M, M_half);

    carry_inter_segment_kernel<<<1, 1>>>(d_seg_carry, num_segs);

    carry_fixup_kernel<<<num_segs, 1>>>(d_out, d_seg_carry, N);

    cudaDeviceSynchronize();

    vector<TestDataTypeUint> out(N);
    cudaMemcpy(out.data(), d_out, N * sizeof(TestDataTypeUint), cudaMemcpyDeviceToHost);

    cudaFree(d_C_hi);
    cudaFree(d_C_lo);
    cudaFree(d_out);
    cudaFree(d_seg_carry);
    return out;
}

// ----------------------------------------------------------------
// Helper: build (C_hi, C_lo) from a flat __int128 value per coeff
// ----------------------------------------------------------------
void to_hilo(unsigned __int128 val, uint64_t& hi, uint64_t& lo) {
    lo = (uint64_t)val;
    hi = (uint64_t)(val >> 64);
}

// ----------------------------------------------------------------
// Choose M and M_half based on limb mode — mirrors what
// host_multiply_merge computes from the actual moduli
// ----------------------------------------------------------------
void get_M(unsigned __int128& M, unsigned __int128& M_half) {
#if LIMB_BITS == 64
    // two 50-bit primes — replace with your actual values
    uint64_t primes[] = {0x6723cbb800001, 0x6723cb6800001};
#else
    uint64_t primes[] = {0x23800001, 0x26800001, 0x2d000001};
#endif
    M = 1;
    for (auto p : primes) M *= p;
    M_half = M >> 1;
}

// ----------------------------------------------------------------
// Tests
// ----------------------------------------------------------------

void test_all_zeros() {
    size_t N = 64;
    unsigned __int128 M, M_half;
    get_M(M, M_half);

    vector<uint64_t> C_hi(N, 0), C_lo(N, 0);
    auto gpu = run_carry_prop(C_hi, C_lo, M, M_half, N);
    auto cpu = cpu_carry_prop(C_hi, C_lo, M, M_half, N);

    bool ok = (gpu == cpu);
    check("all zeros", ok);
}

void test_small_values_no_carry() {
    // coefficients small enough that no carry between limbs occurs
    size_t N = 64;
    unsigned __int128 M, M_half;
    get_M(M, M_half);

    vector<uint64_t> C_hi(N, 0), C_lo(N);
    for (size_t i = 0; i < N; i++) C_lo[i] = (uint64_t)(i + 1);  // 1..64, fits in one limb

    auto gpu = run_carry_prop(C_hi, C_lo, M, M_half, N);
    auto cpu = cpu_carry_prop(C_hi, C_lo, M, M_half, N);

    bool ok = (gpu == cpu);
    check("small values no carry", ok);
}

void test_single_overflow() {
    // one coefficient just over LIMB_MASK — forces carry into next limb
    size_t N = 64;
    unsigned __int128 M, M_half;
    get_M(M, M_half);

    vector<uint64_t> C_hi(N, 0), C_lo(N, 0);
    // set coeff 0 to LIMB_MASK + 1 — should produce limb=0, carry=1
    C_lo[0] = LIMB_MASK + 1;

    auto gpu = run_carry_prop(C_hi, C_lo, M, M_half, N);
    auto cpu = cpu_carry_prop(C_hi, C_lo, M, M_half, N);

    bool ok = (gpu == cpu);
    check("single overflow into next limb", ok);
}

void test_max_limb_values() {
    // every coefficient at LIMB_MASK — heavy carry chain across all segments
    size_t N = 1 << 10;
    unsigned __int128 M, M_half;
    get_M(M, M_half);

    vector<uint64_t> C_hi(N, 0), C_lo(N, LIMB_MASK);

    auto gpu = run_carry_prop(C_hi, C_lo, M, M_half, N);
    auto cpu = cpu_carry_prop(C_hi, C_lo, M, M_half, N);

    bool ok = (gpu == cpu);
    check("max limb values (heavy carry chain)", ok);
}

void test_negative_after_M_reduction() {
    // coefficient > M_half so it gets shifted negative — tests the val -= M path
    size_t N = 16;
    unsigned __int128 M, M_half;
    get_M(M, M_half);

    vector<uint64_t> C_hi(N, 0), C_lo(N, 0);
    // set coeff 0 to M_half + 1 — should become negative after reduction
    unsigned __int128 val = M_half + 1;
    to_hilo(val, C_hi[0], C_lo[0]);

    auto gpu = run_carry_prop(C_hi, C_lo, M, M_half, N);
    auto cpu = cpu_carry_prop(C_hi, C_lo, M, M_half, N);

    bool ok = (gpu == cpu);
    check("negative after M reduction", ok);
}

void test_cross_segment_carry() {
    // carry must propagate across a segment boundary
    // fill last coeff of segment 0 with LIMB_MASK to force carry into segment 1
    size_t N = CARRY_SEG * 2;
    unsigned __int128 M, M_half;
    get_M(M, M_half);

    vector<uint64_t> C_hi(N, 0), C_lo(N, 0);
    C_lo[CARRY_SEG - 1] = LIMB_MASK + 1;  // last coeff of seg 0 overflows

    auto gpu = run_carry_prop(C_hi, C_lo, M, M_half, N);
    auto cpu = cpu_carry_prop(C_hi, C_lo, M, M_half, N);

    bool ok = (gpu == cpu);
    check("cross-segment carry", ok);
}

void test_large_random() {
    size_t N = 1 << 20;
    unsigned __int128 M, M_half;
    get_M(M, M_half);

    vector<uint64_t> C_hi(N), C_lo(N);
    for (size_t i = 0; i < N; i++) {
        unsigned __int128 val;
        if      (i % 3 == 0) val = (uint64_t)(i * 12345 + 1);
        else if (i % 3 == 1) val = M_half - (i % 7);
        else                 val = M_half + (i % 7) + 1;
        to_hilo(val, C_hi[i], C_lo[i]);
    }

    auto gpu = run_carry_prop(C_hi, C_lo, M, M_half, N);
    auto cpu = cpu_carry_prop(C_hi, C_lo, M, M_half, N);

    size_t mismatch_count = 0;
    for (size_t i = 0; i < N && mismatch_count < 8; i++) {
        if (gpu[i] != cpu[i]) {
            printf("  MISMATCH [%zu] gpu=%u cpu=%u\n", i, (uint32_t)gpu[i], (uint32_t)cpu[i]);
            mismatch_count++;
        }
    }
    printf("  total mismatches in first N: %zu\n", mismatch_count);

    bool ok = (gpu == cpu);
    check("large random (N=2^20)", ok);
}

int main() {
    printf("=== carry_prop tests (LIMB_BITS=%d) ===\n", LIMB_BITS);

    test_all_zeros();
    test_small_values_no_carry();
    test_single_overflow();
    test_max_limb_values();
    test_negative_after_M_reduction();
    test_cross_segment_carry();
    test_large_random();

    printf("\n%d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}