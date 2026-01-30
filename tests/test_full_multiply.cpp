#include <vector>
#include <iostream>
#include <random>
#include <cassert>
#include <limits>

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
        else if (i % 8 == 2) v[i] = numeric_limits<TestDataTypeUint>::max();
        else v[i] = (TestDataTypeUint)rng();
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

// ---------------- FULL PIPELINE TEST ----------------
void test_full_pipeline(size_t L)
{
    cout << YELLOW << "\n[Test] Full multiply pipeline, L = "
         << L << " limbs" << RESET << "\n";

    // vector<TestDataTypeUint> A = random_limbs(L, 1234);
    // vector<TestDataTypeUint> B = random_limbs(L, 5678);

    vector<TestDataTypeUint> A = {1, 2, 3, 4};
    vector<TestDataTypeUint> B = {5, 6, 7, 8};

    // GPU pipeline
    vector<TestDataTypeUint> C_gpu;
    host_multiply_merge(A, B, C_gpu);

    // CPU reference
    vector<TestDataTypeUint> C_cpu = cpu_schoolbook_mul(A, B);

    // print cpu reference
    cout << "CPU result: ";
    for (auto limb : C_cpu) {
        cout << limb << " ";
    }
    cout << "\n";

    // print gpu result
    cout << "GPU result: ";
    for (auto limb : C_gpu) {
        cout << limb << " ";
    }
    cout << "\n";

    // Compare
    bool ok = compare_vectors(C_gpu, C_cpu);

    if (ok)
        cout << GREEN_BOLD << "[PASS] Full pipeline correct\n" << RESET;
    else
        cout << RED_BOLD << "[FAIL] Pipeline incorrect\n" << RESET;
}

// ---------------- MAIN TEST DRIVER ----------------
int main()
{
    cout << YELLOW << "==== FULL MULTIPLICATION PIPELINE TEST ====\n" << RESET;

    // Small sizes (debug friendly)
    test_full_pipeline(4);
    // test_full_pipeline(8);
    // test_full_pipeline(16);

    // // Medium
    // test_full_pipeline(64);
    // test_full_pipeline(128);

    // // Larger stress test
    // test_full_pipeline(512);

    cout << YELLOW << "\n==== TEST COMPLETE ====\n" << RESET;
    return 0;
}
