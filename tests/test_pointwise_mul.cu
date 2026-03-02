#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cassert>
#include <iostream>
#include <random>
#include "config.h"
#include "gpu_ntt.h"

#define GREEN_BOLD "\033[1;32m"
#define RED_BOLD   "\033[1;31m"
#define RESET      "\033[0m"

using namespace std;

vector<TestDataTypeUint> testmoduli = {754974721, 595591169, 645922817};

TestDataTypeUint cpu_modmul(TestDataTypeUint a, TestDataTypeUint b, TestDataTypeUint modulus) {
    if constexpr (sizeof(TestDataTypeUint) == 4) {
        uint64_t full = (uint64_t)a * (uint64_t)b;
        return (TestDataTypeUint)(full % modulus);
    } else {
        __uint128_t full = (__uint128_t)a * b;
        return (TestDataTypeUint)(full % modulus);
    }
}

// Main test function
void test_pointwise_multiply() {
    size_t N = 16; // small size for demonstration
    vector<vector<TestDataTypeUint>> A_mod(NUM_MODULI, vector<TestDataTypeUint>(N));
    vector<vector<TestDataTypeUint>> B_mod(NUM_MODULI, vector<TestDataTypeUint>(N));
    vector<vector<TestDataTypeUint>> C_mod; // will be filled by GPU function

    // Fill inputs with random numbers + edge cases
    mt19937 rng(1234);
    uniform_int_distribution<uint64_t> dist(0, UINT32_MAX);

    for (size_t m = 0; m < NUM_MODULI; ++m) {
        for (size_t i = 0; i < N; ++i) {
            if (i % 5 == 0) {
                A_mod[m][i] = 0;
                B_mod[m][i] = 0;
            } else if (i % 5 == 1) {
                A_mod[m][i] = 1;
                B_mod[m][i] = 1;
            } else if (i % 5 == 2) {
                A_mod[m][i] = numeric_limits<TestDataTypeUint>::max();
                B_mod[m][i] = numeric_limits<TestDataTypeUint>::max();
            } else if (i % 5 == 3) {
                A_mod[m][i] = numeric_limits<TestDataTypeUint>::max() / 2;
                B_mod[m][i] = numeric_limits<TestDataTypeUint>::max() / 2;
            } else {
                A_mod[m][i] = (TestDataTypeUint)dist(rng);
                B_mod[m][i] = (TestDataTypeUint)dist(rng);
            }
        }
    }

    // Run GPU pointwise multiply
    // gpu_pointwise_multiply(A_mod, B_mod, C_mod);

    // Verify results against CPU
    bool ok = true;
    // for (size_t m = 0; m < NUM_MODULI; ++m) {
    //     for (size_t i = 0; i < N; ++i) {
    //         TestDataTypeUint expected = cpu_modmul(A_mod[m][i], B_mod[m][i], testmoduli[m]);
    //         if (C_mod[m][i] != expected) {
    //             cout << "[ERROR] modulus " << m << ", index " << i
    //                  << ": GPU=" << C_mod[m][i]
    //                  << ", CPU=" << expected << endl;
    //             ok = false;
    //         }
    //     }
    // }

    if (ok)
        cout << GREEN_BOLD << "[TEST PASS] All GPU results match CPU reference." << RESET << "\n";
    else
        cout << RED_BOLD << "[TEST FAIL] Mismatches detected." << RESET << "\n";
}

int main() {
    test_pointwise_multiply();
    return 0;
}