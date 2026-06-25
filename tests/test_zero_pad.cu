// test_zero_pad.cu
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cstdint>
#include "zero_pad.h"
#include "config.h"

static int passed = 0, failed = 0;

void check(const char* name, bool ok) {
    if (ok) { printf("  PASS  %s\n", name); passed++; }
    else     { printf("  FAIL  %s\n", name); failed++; }
}

#if LIMB_BITS == 64
static constexpr TestDataTypeUint kZeroPadModulus = 0x400002600000001ULL;
#else
static constexpr TestDataTypeUint kZeroPadModulus = 0x2d000001ULL;
#endif

static TestDataTypeUint expected_padded(uint32_t x)
{
    return (TestDataTypeUint)(x % kZeroPadModulus);
}

std::vector<TestDataTypeUint> run(const std::vector<uint32_t>& src, size_t N) {
    size_t L = src.size();

    uint32_t* d_src;
    TestDataTypeUint* d_dst;
    cudaMalloc(&d_src, L * sizeof(uint32_t));
    cudaMalloc(&d_dst, N * sizeof(TestDataTypeUint));

    cudaMemcpy(d_src, src.data(), L * sizeof(uint32_t), cudaMemcpyHostToDevice);

    zero_pad_gpu(d_src, d_dst, L, N, kZeroPadModulus, /*stream=*/0);
    cudaDeviceSynchronize();

    std::vector<TestDataTypeUint> out(N);
    cudaMemcpy(out.data(), d_dst, N * sizeof(TestDataTypeUint), cudaMemcpyDeviceToHost);

    cudaFree(d_src);
    cudaFree(d_dst);
    return out;
}

void test_normal_pad() {
    std::vector<uint32_t> src = {1, 2, 3, 4};
    auto out = run(src, 8);

    bool ok = true;
    for (size_t i = 0; i < 4; i++)
        ok &= (out[i] == expected_padded(src[i]));
    for (size_t i = 4; i < 8; i++) ok &= (out[i] == 0);
    check("normal pad (L=4, N=8)", ok);
}

void test_noop_pad() {
    std::vector<uint32_t> src = {10, 20, 30, 40};
    auto out = run(src, 4);

    bool ok = true;
    for (size_t i = 0; i < 4; i++)
        ok &= (out[i] == expected_padded(src[i]));
    check("no-op pad (L==N)", ok);
}

void test_single_element() {
    std::vector<uint32_t> src = {42};
    auto out = run(src, 8);

    bool ok = (out[0] == 42);
    for (size_t i = 1; i < 8; i++) ok &= (out[i] == 0);
    check("single element (L=1, N=8)", ok);
}

void test_zeros_in_source_preserved() {
    std::vector<uint32_t> src = {5, 0, 7, 0};
    auto out = run(src, 8);

    bool ok = true;
    for (size_t i = 0; i < 4; i++)
        ok &= (out[i] == expected_padded(src[i]));
    for (size_t i = 4; i < 8; i++) ok &= (out[i] == 0);
    check("zeros in source preserved", ok);
}

// verify widening: high 32 bits of each output must be zero
// (32-bit inputs should never set the upper half in 64-bit mode)
void test_no_upper_bits_set() {
    std::vector<uint32_t> src = {0xFFFFFFFF, 0xDEADBEEF, 0x12345678};
    auto out = run(src, 4);

    bool ok = true;
    for (size_t i = 0; i < 3; i++) {
        ok &= (out[i] == expected_padded(src[i]));
        if constexpr (sizeof(TestDataTypeUint) == 8)
            ok &= ((static_cast<uint64_t>(out[i]) & 0xFFFFFFFF00000000ULL) == 0);
    }
    ok &= (out[3] == 0);
    check("no upper bits set after widen", ok);
}

void test_large_N() {
    size_t L = 1 << 20;
    size_t N = 1 << 23;
    std::vector<uint32_t> src(L);
    for (size_t i = 0; i < L; i++) src[i] = (uint32_t)(i + 1);

    auto out = run(src, N);

    bool ok = true;
    for (size_t i = 0; i < L && ok; i++)
        ok &= (out[i] == expected_padded(src[i]));
    for (size_t i = L; i < N && ok; i++) ok &= (out[i] == 0);
    check("large N (L=2^20, N=2^23)", ok);
}

#if defined(NATIVE_HOST_LIMBS)

static constexpr TestDataTypeUint kTestModulus = 0x400002600000001ULL;

std::vector<TestDataTypeUint> run_u64(const std::vector<uint64_t>& src, size_t N,
                                      TestDataTypeUint modulus) {
    size_t L = src.size();
    uint64_t* d_src;
    TestDataTypeUint* d_dst;
    cudaMalloc(&d_src, L * sizeof(uint64_t));
    cudaMalloc(&d_dst, N * sizeof(TestDataTypeUint));
    cudaMemcpy(d_src, src.data(), L * sizeof(uint64_t), cudaMemcpyHostToDevice);
    zero_pad_gpu_u64(d_src, d_dst, L, N, modulus, 0);
    cudaDeviceSynchronize();
    std::vector<TestDataTypeUint> out(N);
    cudaMemcpy(out.data(), d_dst, N * sizeof(TestDataTypeUint), cudaMemcpyDeviceToHost);
    cudaFree(d_src);
    cudaFree(d_dst);
    return out;
}

void test_u64_mod_pad() {
    const TestDataTypeUint p = kTestModulus;
    std::vector<uint64_t> src = {42, 1ULL << 40, UINT64_MAX};
    auto out = run_u64(src, 8, p);
    bool ok = true;
    for (size_t i = 0; i < 3; i++)
        ok &= (out[i] == (TestDataTypeUint)(src[i] % p));
    for (size_t i = 3; i < 8; i++) ok &= (out[i] == 0);
    check("u64 mod-p pad", ok);
}

void test_u64_full_range_reduced() {
    const TestDataTypeUint p = kTestModulus;
    std::vector<uint64_t> src = {UINT64_MAX, UINT64_MAX - 1, 1};
    auto out = run_u64(src, 4, p);
    bool ok = true;
    for (size_t i = 0; i < 3; i++)
        ok &= (out[i] == (TestDataTypeUint)(src[i] % p));
    check("u64 full-range limbs reduced mod p", ok);
}

#endif

int main() {
    printf("=== zero_pad tests (LIMB_BITS=%d) ===\n", LIMB_BITS);
#if defined(NATIVE_HOST_LIMBS)
    test_u64_mod_pad();
    test_u64_full_range_reduced();
#else
    test_normal_pad();
    test_noop_pad();
    test_single_element();
    test_zeros_in_source_preserved();
    test_no_upper_bits_set();
    test_large_N();
#endif
    printf("\n%d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}