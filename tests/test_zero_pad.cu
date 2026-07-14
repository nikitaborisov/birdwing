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

// Pipeline primes (for pad smoke tests that mirror production moduli).
#if LIMB_BITS == 32
static constexpr TestDataTypeUint kPipelineModulus = 0x2d000001u;
#else
static constexpr TestDataTypeUint kPipelineModulus = 0x400002600000001ULL;
#endif

// Intentionally small modulus so limbs exceed it on both uint32 and uint64
// kernels — a cast-only (no `%`) implementation must fail these checks.
static constexpr TestDataTypeUint kRegressionModulus = 17;

std::vector<TestDataTypeUint> run_u32(const std::vector<uint32_t>& src, size_t N,
                                      TestDataTypeUint modulus) {
    size_t L = src.size();

    uint32_t* d_src;
    TestDataTypeUint* d_dst;
    cudaMalloc(&d_src, L * sizeof(uint32_t));
    cudaMalloc(&d_dst, N * sizeof(TestDataTypeUint));

    cudaMemcpy(d_src, src.data(), L * sizeof(uint32_t), cudaMemcpyHostToDevice);

    zero_pad_gpu(d_src, d_dst, L, N, modulus, /*stream=*/0);
    cudaDeviceSynchronize();

    std::vector<TestDataTypeUint> out(N);
    cudaMemcpy(out.data(), d_dst, N * sizeof(TestDataTypeUint), cudaMemcpyDeviceToHost);

    cudaFree(d_src);
    cudaFree(d_dst);
    return out;
}

#if !defined(NATIVE_HOST_LIMBS)

void test_normal_pad() {
    std::vector<uint32_t> src = {1, 2, 3, 4};
    auto out = run_u32(src, 8, kPipelineModulus);

    bool ok = true;
    for (size_t i = 0; i < 4; i++) ok &= (out[i] == src[i] % kPipelineModulus);
    for (size_t i = 4; i < 8; i++) ok &= (out[i] == 0);
    check("normal pad (L=4, N=8)", ok);
}

void test_noop_pad() {
    std::vector<uint32_t> src = {10, 20, 30, 40};
    auto out = run_u32(src, 4, kPipelineModulus);

    bool ok = true;
    for (size_t i = 0; i < 4; i++) ok &= (out[i] == src[i] % kPipelineModulus);
    check("no-op pad (L==N)", ok);
}

void test_single_element() {
    std::vector<uint32_t> src = {42};
    auto out = run_u32(src, 8, kPipelineModulus);

    bool ok = (out[0] == 42 % kPipelineModulus);
    for (size_t i = 1; i < 8; i++) ok &= (out[i] == 0);
    check("single element (L=1, N=8)", ok);
}

void test_zeros_in_source_preserved() {
    std::vector<uint32_t> src = {5, 0, 7, 0};
    auto out = run_u32(src, 8, kPipelineModulus);

    bool ok = true;
    for (size_t i = 0; i < 4; i++) ok &= (out[i] == src[i] % kPipelineModulus);
    for (size_t i = 4; i < 8; i++) ok &= (out[i] == 0);
    check("zeros in source preserved", ok);
}

// verify widening: high 32 bits of each output must be zero
// (32-bit inputs should never set the upper half in 64-bit mode)
void test_no_upper_bits_set() {
    std::vector<uint32_t> src = {0xFFFFFFFF, 0xDEADBEEF, 0x12345678};
    auto out = run_u32(src, 4, kPipelineModulus);

    bool ok = true;
    for (size_t i = 0; i < 3; i++) {
        ok &= (out[i] == (TestDataTypeUint)(src[i] % kPipelineModulus));
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

    auto out = run_u32(src, N, kPipelineModulus);

    bool ok = true;
    for (size_t i = 0; i < L && ok; i++)
        ok &= (out[i] == src[i] % kPipelineModulus);
    for (size_t i = L; i < N && ok; i++) ok &= (out[i] == 0);
    check("large N (L=2^20, N=2^23)", ok);
}

void test_limbs_reduced_mod_p() {
    const TestDataTypeUint p = kPipelineModulus;
    std::vector<uint32_t> src = {0xFFFFFFFFu, 0xDEADBEEFu, 1u};
#if LIMB_BITS == 32
    // Exercise exact boundary cases against a ~30-bit prime.
    src.push_back(static_cast<uint32_t>(p));
    src.push_back(static_cast<uint32_t>(p) + 1u);
#endif
    auto out = run_u32(src, 8, p);

    bool ok = true;
    for (size_t i = 0; i < src.size(); i++)
        ok &= (out[i] == (TestDataTypeUint)(src[i] % p));
    for (size_t i = src.size(); i < 8; i++) ok &= (out[i] == 0);
    check("uint32 limbs reduced mod p", ok);
}

// Regression: uint32 zero_pad must reduce mod p (cast-only would leave src[i]).
void test_u32_reduction_regression() {
    const TestDataTypeUint p = kRegressionModulus;
    std::vector<uint32_t> src = {
        0u,
        1u,
        static_cast<uint32_t>(p),
        static_cast<uint32_t>(p) + 1u,
        18u,
        100u,
        0xFFFFFFFFu,
    };
    auto out = run_u32(src, 16, p);

    bool ok = true;
    bool saw_strict_reduction = false;
    for (size_t i = 0; i < src.size(); i++) {
        const TestDataTypeUint want = (TestDataTypeUint)(src[i] % p);
        ok &= (out[i] == want);
        ok &= (out[i] < p);
        if (src[i] >= p)
            saw_strict_reduction |= (out[i] != (TestDataTypeUint)src[i]);
    }
    for (size_t i = src.size(); i < 16; i++) ok &= (out[i] == 0);
    ok &= saw_strict_reduction;
    check("REGRESSION u32 zero_pad reduces mod p", ok);
}

#if LIMB_BITS == 32
// Also pin reduction against a real ~30-bit pipeline prime.
void test_u32_pipeline_prime_reduction() {
    const TestDataTypeUint p = kPipelineModulus;
    std::vector<uint32_t> src = {
        0xFFFFFFFFu,
        static_cast<uint32_t>(p),
        static_cast<uint32_t>(p) + 1u,
        0xDEADBEEFu,
    };
    auto out = run_u32(src, 8, p);

    bool ok = true;
    for (size_t i = 0; i < src.size(); i++) {
        ok &= (out[i] == (TestDataTypeUint)(src[i] % p));
        ok &= (out[i] < p);
        ok &= (out[i] != (TestDataTypeUint)src[i]);  // all inputs >= p
    }
    for (size_t i = src.size(); i < 8; i++) ok &= (out[i] == 0);
    check("REGRESSION u32 reduces vs pipeline prime", ok);
}
#endif

#endif // !NATIVE_HOST_LIMBS

#if defined(NATIVE_HOST_LIMBS)

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
    const TestDataTypeUint p = kPipelineModulus;
    std::vector<uint64_t> src = {42, 1ULL << 40, UINT64_MAX};
    auto out = run_u64(src, 8, p);
    bool ok = true;
    for (size_t i = 0; i < 3; i++)
        ok &= (out[i] == (TestDataTypeUint)(src[i] % p));
    for (size_t i = 3; i < 8; i++) ok &= (out[i] == 0);
    check("u64 mod-p pad", ok);
}

// Regression: uint64 zero_pad must reduce mod p (cast-only would leave src[i]).
void test_u64_reduction_regression() {
    const TestDataTypeUint p = kRegressionModulus;
    std::vector<uint64_t> src = {
        0ull,
        1ull,
        (uint64_t)p,
        (uint64_t)p + 1ull,
        18ull,
        1ull << 40,
        UINT64_MAX,
        UINT64_MAX - 1,
    };
    auto out = run_u64(src, 16, p);

    bool ok = true;
    bool saw_strict_reduction = false;
    for (size_t i = 0; i < src.size(); i++) {
        const TestDataTypeUint want = (TestDataTypeUint)(src[i] % p);
        ok &= (out[i] == want);
        ok &= (out[i] < p);
        if (src[i] >= p)
            saw_strict_reduction |= (out[i] != (TestDataTypeUint)src[i]);
    }
    for (size_t i = src.size(); i < 16; i++) ok &= (out[i] == 0);
    ok &= saw_strict_reduction;
    check("REGRESSION u64 zero_pad reduces mod p", ok);
}

void test_u64_pipeline_prime_reduction() {
    const TestDataTypeUint p = kPipelineModulus;
    std::vector<uint64_t> src = {
        UINT64_MAX,
        (uint64_t)p,
        (uint64_t)p + 1ull,
        1ull << 63,
    };
    auto out = run_u64(src, 8, p);

    bool ok = true;
    for (size_t i = 0; i < src.size(); i++) {
        ok &= (out[i] == (TestDataTypeUint)(src[i] % p));
        ok &= (out[i] < p);
        ok &= (out[i] != (TestDataTypeUint)src[i]);  // all inputs >= p
    }
    for (size_t i = src.size(); i < 8; i++) ok &= (out[i] == 0);
    check("REGRESSION u64 reduces vs pipeline prime", ok);
}

#endif

int main() {
#if defined(NATIVE_HOST_LIMBS)
    printf("=== zero_pad tests (LIMB_BITS=%d, NATIVE_HOST_LIMBS) ===\n", LIMB_BITS);
    test_u64_mod_pad();
    test_u64_reduction_regression();
    test_u64_pipeline_prime_reduction();
#else
    printf("=== zero_pad tests (LIMB_BITS=%d) ===\n", LIMB_BITS);
    test_normal_pad();
    test_noop_pad();
    test_single_element();
    test_zeros_in_source_preserved();
    test_no_upper_bits_set();
    test_limbs_reduced_mod_p();
    test_u32_reduction_regression();
#if LIMB_BITS == 32
    test_u32_pipeline_prime_reduction();
#endif
    test_large_N();
#endif
    printf("\n%d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
