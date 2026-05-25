// test_zero_pad.cu
#include <cuda_runtime.h>
#include <cassert>
#include <vector>
#include <cstdio>
#include "zero_pad.h"
#include "config.h"

static int passed = 0, failed = 0;

void check(const char* name, bool ok) {
    if (ok) { printf("  PASS  %s\n", name); passed++; }
    else     { printf("  FAIL  %s\n", name); failed++; }
}

// Run zero_pad_gpu and return result on host
std::vector<TestDataTypeUint> run(
    const std::vector<TestDataTypeUint>& src, size_t N)
{
    size_t L = src.size();

    TestDataTypeUint *d_src, *d_dst;
    cudaMalloc(&d_src, L * sizeof(TestDataTypeUint));
    cudaMalloc(&d_dst, N * sizeof(TestDataTypeUint));

    cudaMemcpy(d_src, src.data(), L * sizeof(TestDataTypeUint),
               cudaMemcpyHostToDevice);

    zero_pad_gpu(d_src, d_dst, L, N, /*stream=*/0);
    cudaDeviceSynchronize();

    std::vector<TestDataTypeUint> out(N);
    cudaMemcpy(out.data(), d_dst, N * sizeof(TestDataTypeUint),
               cudaMemcpyDeviceToHost);

    cudaFree(d_src);
    cudaFree(d_dst);
    return out;
}

void test_normal_pad() {
    std::vector<TestDataTypeUint> src = {1, 2, 3, 4};
    auto out = run(src, 8);

    bool ok = true;
    for (size_t i = 0; i < 4; i++) ok &= (out[i] == src[i]);
    for (size_t i = 4; i < 8; i++) ok &= (out[i] == 0);
    check("normal pad (L=4, N=8)", ok);
}

void test_noop_pad() {
    std::vector<TestDataTypeUint> src = {10, 20, 30, 40};
    auto out = run(src, 4);  // L == N

    bool ok = true;
    for (size_t i = 0; i < 4; i++) ok &= (out[i] == src[i]);
    check("no-op pad (L==N)", ok);
}

void test_single_element() {
    std::vector<TestDataTypeUint> src = {42};
    auto out = run(src, 8);

    bool ok = (out[0] == 42);
    for (size_t i = 1; i < 8; i++) ok &= (out[i] == 0);
    check("single element (L=1, N=8)", ok);
}

void test_zeros_in_source_preserved() {
    // zeros inside [0,L) must not be confused with padding zeros
    std::vector<TestDataTypeUint> src = {5, 0, 7, 0};
    auto out = run(src, 8);

    bool ok = true;
    for (size_t i = 0; i < 4; i++) ok &= (out[i] == src[i]);
    for (size_t i = 4; i < 8; i++) ok &= (out[i] == 0);
    check("zeros in source preserved", ok);
}

void test_large_N() {
    size_t L = 1 << 20;
    size_t N = 1 << 23;
    std::vector<TestDataTypeUint> src(L);
    for (size_t i = 0; i < L; i++) src[i] = (TestDataTypeUint)(i + 1);

    auto out = run(src, N);

    bool ok = true;
    for (size_t i = 0; i < L && ok; i++) ok &= (out[i] == src[i]);
    for (size_t i = L; i < N && ok; i++) ok &= (out[i] == 0);
    check("large N (L=2^20, N=2^23)", ok);
}

int main() {
    printf("=== zero_pad tests ===\n");
    test_normal_pad();
    test_noop_pad();
    test_single_element();
    test_zeros_in_source_preserved();
    test_large_N();
    printf("\n%d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}