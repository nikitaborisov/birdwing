// Carry propagation unit tests: 128-bit CRT coefficients -> 32-bit output limbs.
// Independent of LIMB_BITS / NUM_MODULI (same kernels in 32- and 64-bit builds).
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include "carry_prop.h"
#include "config.h"

using namespace std;

static int passed = 0, failed = 0;

static void check(const char* name, bool ok) {
    if (ok) { printf("  PASS  %s\n", name); passed++; }
    else     { printf("  FAIL  %s\n", name); failed++; }
}

// Serial CPU reference — matches scripts/carry.py.
static vector<OutputLimbType> cpu_carry_prop(
    const vector<uint64_t>& C_hi,
    const vector<uint64_t>& C_lo,
    size_t N)
{
    vector<OutputLimbType> out(N, 0);
    __int128 carry = 0;

    for (size_t i = 0; i < N; i++) {
        __int128 val = ((__uint128_t)C_hi[i] << 64) | C_lo[i];
        __int128 temp = val + carry;
        out[i] = (OutputLimbType)(temp & OUTPUT_LIMB_MASK);
        carry  = temp >> OUTPUT_LIMB_BITS;
    }
    return out;
}

static vector<OutputLimbType> run_carry_prop_gpu(
    const vector<uint64_t>& h_C_hi,
    const vector<uint64_t>& h_C_lo,
    size_t N)
{
    uint64_t *d_C_hi, *d_C_lo;
    cudaMalloc(&d_C_hi, N * sizeof(uint64_t));
    cudaMalloc(&d_C_lo, N * sizeof(uint64_t));
    cudaMemcpy(d_C_hi, h_C_hi.data(), N * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_lo, h_C_lo.data(), N * sizeof(uint64_t), cudaMemcpyHostToDevice);

    OutputLimbType* d_out;
    cudaMalloc(&d_out, N * sizeof(OutputLimbType));
    cudaMemset(d_out, 0, N * sizeof(OutputLimbType));

    size_t num_segs = (N + CARRY_SEG - 1) / CARRY_SEG;
    int64_t* d_seg_carry;
    cudaMalloc(&d_seg_carry, num_segs * sizeof(int64_t));

    carry_intra_segment_kernel<<<num_segs, 1>>>(
        d_C_hi, d_C_lo, d_out, d_seg_carry, N);

    carry_inter_segment_kernel<<<1, 1>>>(d_seg_carry, num_segs);

    int* d_escape;
    cudaMalloc(&d_escape, sizeof(int));
    for (;;) {
        cudaMemset(d_escape, 0, sizeof(int));
        carry_fixup_kernel<<<num_segs, 1>>>(d_out, d_seg_carry, N, num_segs, d_escape);
        int escaped = 0;
        cudaMemcpy(&escaped, d_escape, sizeof(int), cudaMemcpyDeviceToHost);
        if (!escaped)
            break;
    }
    cudaFree(d_escape);

    cudaDeviceSynchronize();

    vector<OutputLimbType> out(N);
    cudaMemcpy(out.data(), d_out, N * sizeof(OutputLimbType), cudaMemcpyDeviceToHost);

    cudaFree(d_C_hi);
    cudaFree(d_C_lo);
    cudaFree(d_out);
    cudaFree(d_seg_carry);
    return out;
}

static void to_hilo(unsigned __int128 val, uint64_t& hi, uint64_t& lo) {
    lo = (uint64_t)val;
    hi = (uint64_t)(val >> 64);
}

static bool check_gpu_cpu(
    const vector<uint64_t>& C_hi,
    const vector<uint64_t>& C_lo,
    size_t N,
    const char* name)
{
    auto gpu = run_carry_prop_gpu(C_hi, C_lo, N);
    auto cpu = cpu_carry_prop(C_hi, C_lo, N);
    bool ok = (gpu == cpu);
    if (!ok) {
        for (size_t i = 0; i < N && i < 8; i++) {
            if (gpu[i] != cpu[i])
                printf("  [%s] mismatch k=%zu gpu=%u cpu=%u\n",
                       name, i, gpu[i], cpu[i]);
        }
    }
    check(name, ok);
    return ok;
}

static void test_all_zeros() {
    size_t N = 64;
    vector<uint64_t> C_hi(N, 0), C_lo(N, 0);
    check_gpu_cpu(C_hi, C_lo, N, "all zeros");
}

static void test_small_values_no_carry() {
    size_t N = 64;
    vector<uint64_t> C_hi(N, 0), C_lo(N);
    for (size_t i = 0; i < N; i++) C_lo[i] = (uint64_t)(i + 1);
    check_gpu_cpu(C_hi, C_lo, N, "small values no carry");
}

static void test_single_overflow() {
    size_t N = 64;
    vector<uint64_t> C_hi(N, 0), C_lo(N, 0);
    C_lo[0] = OUTPUT_LIMB_MASK + 1;
    check_gpu_cpu(C_hi, C_lo, N, "single overflow into next limb");
}

static void test_max_limb_values() {
    size_t N = 1 << 10;
    vector<uint64_t> C_hi(N, 0), C_lo(N, OUTPUT_LIMB_MASK);
    check_gpu_cpu(C_hi, C_lo, N, "max limb values (heavy carry chain)");
}

static void test_wide_128bit_coeff() {
    // Coefficient uses high word — carry must see full 128-bit value.
    size_t N = 16;
    vector<uint64_t> C_hi(N, 0), C_lo(N, 0);
    unsigned __int128 val = (unsigned __int128)1 << 70;
    val += 12345;
    to_hilo(val, C_hi[0], C_lo[0]);
    check_gpu_cpu(C_hi, C_lo, N, "wide 128-bit coefficient (hi != 0)");
}

static void test_cross_segment_carry() {
    size_t N = CARRY_SEG * 2;
    vector<uint64_t> C_hi(N, 0), C_lo(N, 0);
    C_lo[CARRY_SEG - 1] = OUTPUT_LIMB_MASK + 1;
    check_gpu_cpu(C_hi, C_lo, N, "cross-segment carry");
}

static void test_large_random() {
    size_t N = 1 << 16;
    vector<uint64_t> C_hi(N), C_lo(N);
    for (size_t i = 0; i < N; i++) {
        unsigned __int128 val;
        if (i % 4 == 0)
            val = (unsigned __int128)(i * 12345 + 1);
        else if (i % 4 == 1)
            val = ((unsigned __int128)1 << 70) + (i % 1000);
        else if (i % 4 == 2)
            val = OUTPUT_LIMB_MASK;
        else
            val = ((unsigned __int128)(i % 17) << 64) + (i * 99991);
        to_hilo(val, C_hi[i], C_lo[i]);
    }
    check_gpu_cpu(C_hi, C_lo, N, "large random (N=2^16)");
}

int main() {
    printf("=== carry_prop unit tests (LIMB_BITS=%d, output=%d-bit limbs) ===\n",
           LIMB_BITS, OUTPUT_LIMB_BITS);
    static_assert(OUTPUT_LIMB_BITS == 32, "carry output must be 32-bit limbs");
    static_assert(sizeof(OutputLimbType) == 4, "OutputLimbType must be uint32_t");

    test_all_zeros();
    test_small_values_no_carry();
    test_single_overflow();
    test_max_limb_values();
    test_wide_128bit_coeff();
    test_cross_segment_carry();
    test_large_random();

    printf("\n%d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
