#include <vector>
#include <iostream>
#include <random>
#include <cassert>
#include <limits>
#include <chrono>
#include <gmp.h>
#include <algorithm>
#include <fstream>

#include "../include/multiply.h"
#include "../include/config.h"

using namespace std;

// ---------------- ANSI COLORS ----------------
#define GREEN_BOLD "\033[1;32m"
#define RED_BOLD   "\033[1;31m"
#define YELLOW     "\033[33m"
#define RESET      "\033[0m"

// ---------------- CPU REFERENCE MULTIPLY ----------------
// Bigint schoolbook multiply with full carry
vector<TestDataTypeUint> cpu_schoolbook_mul(
    const vector<TestDataTypeUint>& A,
    const vector<TestDataTypeUint>& B)
{
    size_t n = A.size(), m = B.size();
    vector<unsigned __int128> tmp(n + m, 0);

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            tmp[i + j] += (unsigned __int128)A[i] * B[j];
        }
    }

    vector<TestDataTypeUint> C(tmp.size());
    unsigned __int128 carry = 0;
    const unsigned SHIFT = 8 * sizeof(TestDataTypeUint);

    for (size_t i = 0; i < tmp.size(); i++) {
        tmp[i] += carry;
        C[i] = (TestDataTypeUint)(tmp[i] & (((unsigned __int128)1 << SHIFT) - 1));
        carry = tmp[i] >> SHIFT;
    }

    while (C.size() > 1 && C.back() == 0)
        C.pop_back();

    size_t linear_len = A.size() + B.size(); // 2*L-1
    if (C.size() < linear_len)
        C.resize(linear_len);

    return C;
}

// ---------------- RANDOM INPUT GENERATOR ----------------
vector<TestDataTypeUint> random_limbs(size_t n, uint64_t seed)
{
    mt19937_64 rng(seed);
    vector<TestDataTypeUint> v(n);

    for (size_t i = 0; i < n; i++) {
        if (i % 8 == 0) v[i] = 0; // edge case
        else if (i % 8 == 1) v[i] = 1;
        // else if (i % 8 == 2) v[i] = numeric_limits<TestDataTypeUint>::max();
        else v[i] = (TestDataTypeUint)rng() % (1ULL << 10);
    }
    return v;
}

// ---------------- COMPARE & REPORT ----------------
bool compare_vectors(const vector<TestDataTypeUint>& A,
                     const vector<TestDataTypeUint>& B)
{
    if (A.size() != B.size()) {
        cout << RED_BOLD << "[FAIL] Size mismatch: "
             << A.size() << " vs " << B.size() << RESET << "\n";
        return false;
    }

    bool ok = true;
    for (size_t i = 0; i < A.size(); i++) {
        if (A[i] != B[i]) {
            cout << RED_BOLD << "[MISMATCH] index " << i
                 << " GPU=" << A[i]
                 << " CPU=" << B[i] << RESET << "\n";
            ok = false;
        }
    }
    return ok;
}

// ---------------- GMP HELPERS ----------------
void to_mpz(mpz_t out, const vector<TestDataTypeUint>& v)
{
    mpz_set_ui(out, 0);
    const size_t limb_bits = 8 * sizeof(TestDataTypeUint);

    for (ssize_t i = (ssize_t)v.size() - 1; i >= 0; i--) {
        mpz_mul_2exp(out, out, limb_bits);
        mpz_add_ui(out, out, v[i]);
    }
}

vector<TestDataTypeUint> from_mpz(const mpz_t x, size_t expected_limbs)
{
    vector<TestDataTypeUint> out(expected_limbs, 0);

    mpz_t tmp;
    mpz_init_set(tmp, x);

    const unsigned limb_bits = 8 * sizeof(TestDataTypeUint);

    for (size_t i = 0; i < expected_limbs; i++) {
        out[i] = (TestDataTypeUint) mpz_get_ui(tmp);
        mpz_fdiv_q_2exp(tmp, tmp, limb_bits);
    }

    mpz_clear(tmp);
    return out;
}

void limbs_to_mpz(mpz_t result, uint32_t *limbs, size_t n) {
    mpz_import(result, n, -1, sizeof(uint32_t), 0, 0, limbs);
}

vector<TestDataTypeUint> fast_from_mpz(mpz_t value, size_t expected_limbs)
{
    vector<TestDataTypeUint> out(expected_limbs);

    size_t count = 0;
    mpz_export(
        out.data(),     // destination buffer
        &count,         // actual number of limbs written
        -1,             // least significant limb first
        sizeof(uint32_t),
        0,              // native endianness
        0,              // no nail bits
        value
    );

    // Zero-pad unused limbs if needed
    out.resize(expected_limbs, 0);

    return out;
}

vector<TestDataTypeUint> gmp_mul(
    const vector<TestDataTypeUint>& A,
    const vector<TestDataTypeUint>& B)
{
    using clock = std::chrono::high_resolution_clock;

    auto t_total_start = clock::now();

    mpz_t a, b, c;
    mpz_init(a);
    mpz_init(b);
    mpz_init(c);

    // ---- Import A ----
    auto t1 = clock::now();
    limbs_to_mpz(a, reinterpret_cast<uint32_t*>(const_cast<TestDataTypeUint*>(A.data())), A.size());
    auto t2 = clock::now();

    // ---- Import B ----
    auto t3 = clock::now();
    limbs_to_mpz(b, reinterpret_cast<uint32_t*>(const_cast<TestDataTypeUint*>(B.data())), B.size());
    auto t4 = clock::now();

    // ---- Multiply ----
    auto t5 = clock::now();
    mpz_mul(c, a, b);
    auto t6 = clock::now();

    // ---- Export ----
    auto t7 = clock::now();
    auto out = fast_from_mpz(c, A.size() + B.size());
    auto t8 = clock::now();

    mpz_clear(a);
    mpz_clear(b);
    mpz_clear(c);

    auto t_total_end = clock::now();

    // ---- Compute durations ----
    double importA_us = std::chrono::duration<double, std::micro>(t2 - t1).count();
    double importB_us = std::chrono::duration<double, std::micro>(t4 - t3).count();
    double multiply_us = std::chrono::duration<double, std::micro>(t6 - t5).count();
    double export_us = std::chrono::duration<double, std::micro>(t8 - t7).count();
    double total_us = std::chrono::duration<double, std::micro>(t_total_end - t_total_start).count();

    // ---- Append to CSV ----
    static bool wrote_header = false;
    std::ofstream csv("gmp_profile.csv", std::ios::app);

    if (!wrote_header) {
        csv << "sizeA,sizeB,importA_us,importB_us,multiply_us,export_us,total_us\n";
        wrote_header = true;
    }

    csv << A.size() << ","
        << B.size() << ","
        << importA_us << ","
        << importB_us << ","
        << multiply_us << ","
        << export_us << ","
        << total_us << "\n";

    return out;
}

// ---------------- PIPELINE TEST ----------------
void test_full_pipeline(size_t L)
{
    cout << YELLOW << "\n[Test] Full multiply pipeline, L = "
         << L << " limbs" << RESET << "\n";

    vector<TestDataTypeUint> A = random_limbs(L, 1234);
    vector<TestDataTypeUint> B = random_limbs(L, 5678);

    // vector<TestDataTypeUint> A = {1,2,3,4,5,6,7,8};
    // vector<TestDataTypeUint> B = {9,10,11,12,13,14,15,16};

    // vector<TestDataTypeUint> A = {1,2,3,4};
    // vector<TestDataTypeUint> B = {5,6,7,8};

    // GPU pipeline
    vector<TestDataTypeUint> C_gpu;
    host_multiply_merge(A, B, C_gpu);

    // CPU reference
    vector<TestDataTypeUint> C_cpu = cpu_schoolbook_mul(A, B);

    // Compare
    bool ok = compare_vectors(C_gpu, C_cpu);

    if (ok)
        cout << GREEN_BOLD << "[PASS] Full pipeline correct\n" << RESET;
    else
        cout << RED_BOLD << "[FAIL] Pipeline incorrect\n" << RESET;
}

void test_identities(size_t L) {
    vector<TestDataTypeUint> Z(L, 0);
    vector<TestDataTypeUint> O(L, 1);
    vector<TestDataTypeUint> R;

    host_multiply_merge(Z, Z, R);
    assert(all_of(R.begin(), R.end(), [](auto x){return x==0;}));

    host_multiply_merge(O, Z, R);
    assert(all_of(R.begin(), R.end(), [](auto x){return x==0;}));

    host_multiply_merge(O, O, R);
    assert(R[0] == 1);
}

// ---------------- BENCHMARK ----------------
void benchmark_vs_gmp(size_t L)
{
    cout << YELLOW << "\n[Benchmark] L = " << L << RESET << "\n";

    vector<TestDataTypeUint> A = random_limbs(L, 1234);
    vector<TestDataTypeUint> B = random_limbs(L, 5678);

    vector<TestDataTypeUint> C_gpu, C_gmp;

    // Warm-up
    vector<TestDataTypeUint> warm;
    host_multiply_merge(A, B, warm);
    C_gmp = gmp_mul(A, B);

    const int ITERS = 100;
    double gpu_time = 0.0, gmp_time = 0.0;

    for (int i = 0; i < ITERS; i++) {
        auto t1 = chrono::high_resolution_clock::now();
        host_multiply_merge(A, B, C_gpu);
        auto t2 = chrono::high_resolution_clock::now();
        gpu_time += chrono::duration<double, milli>(t2 - t1).count();
    }

    for (int i = 0; i < ITERS; i++) {
        auto t3 = chrono::high_resolution_clock::now();
        C_gmp = gmp_mul(A, B);
        auto t4 = chrono::high_resolution_clock::now();
        gmp_time += chrono::duration<double, milli>(t4 - t3).count();
    }

    gpu_time /= ITERS;
    gmp_time /= ITERS;

    cout << "GPU avg time: " << gpu_time << " ms\n";
    cout << "GMP avg time: " << gmp_time << " ms\n";

    bool ok = compare_vectors(C_gpu, C_gmp);
    cout << (ok ? GREEN_BOLD "[MATCH]\n" RESET : RED_BOLD "[MISMATCH]\n" RESET);
}

// ---------------- MAIN ----------------
int main()
{
    cout << YELLOW << "==== FULL MULTIPLICATION PIPELINE TEST ====\n" << RESET;

    test_identities(128);

    test_full_pipeline(4);
    test_full_pipeline(8);
    test_full_pipeline(16);

    test_full_pipeline(128);
    test_full_pipeline(2048);
    test_full_pipeline(10000);

    benchmark_vs_gmp(4);
    benchmark_vs_gmp(64);
    benchmark_vs_gmp(256);
    benchmark_vs_gmp(1ULL << 20);

    cout << YELLOW << "\n==== TEST COMPLETE ====\n" << RESET;
    return 0;
}
