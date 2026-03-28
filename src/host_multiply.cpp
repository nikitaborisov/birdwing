#include "../include/gpu_ntt.h"
#include "../include/multiply.h"
#include "../include/crt.h"
#include "../include/crt_utils.h"
#include "../include/crt_gpu.h"
#include "config.h"
#include <chrono>
#include <fstream>
#include <cstring>

#include <algorithm>

// ---------------- ANSI COLORS ----------------
#define GREEN_BOLD "\033[1;32m"
#define RED_BOLD   "\033[1;31m"
#define YELLOW     "\033[33m"
#define RESET      "\033[0m"

using namespace std;

// parameters
using limb_t = TestDataTypeUint; // each limb is stored in 32-bit containers

// host functions
void host_multiply_merge(const vector<TestDataTypeUint> &A, const vector<TestDataTypeUint> &B, vector<TestDataTypeUint> &C, chrono::duration<double, milli> &duration) {
    size_t L_A = A.size();
    size_t L_B = B.size();
    size_t L_C = L_A + L_B - 1;

    size_t N = 1;
    while (N < L_C)
        N <<= 1;

    TestDataTypeUint* a_pinned;
    TestDataTypeUint* b_pinned;
    cudaMallocHost(&a_pinned, L_A * sizeof(TestDataTypeUint));
    cudaMallocHost(&b_pinned, L_B * sizeof(TestDataTypeUint));
    memcpy(a_pinned, A.data(), L_A * sizeof(TestDataTypeUint));
    memcpy(b_pinned, B.data(), L_B * sizeof(TestDataTypeUint));

    vector<uint64_t> C_hi(N), C_lo(N);
    vector<__uint128_t> C_big(N);
    C.resize(L_C + 1, 0);

    unsigned __int128 M = 1;
    for (int j = 0; j < NUM_MODULI; j++) M *= moduli[j];

    NTTPrecomputed pre = precompute_ntt(N);

    auto t0 = chrono::high_resolution_clock::now();
    NTTContext ctx = allocate_ntt_context(pre, L_A, L_B);
    execute_ntt_multiply(ctx, a_pinned, b_pinned, C_hi, C_lo);
    auto t1 = chrono::high_resolution_clock::now();

    for (size_t i = 0; i < N; i++) {
        C_big[i] = ((unsigned __int128)C_hi[i] << 64) | C_lo[i];
        if (C_big[i] > M / 2) C_big[i] -= M;
    }

    const __int128 BASE = (__int128)1 << 32;
    __int128 carry = 0;

    for (size_t i = 0; i < L_C; ++i) {
        __int128 temp = C_big[i] + carry;

        __int128 limb = temp % BASE;
        if (limb < 0) {
            limb += BASE;
            temp -= BASE;
        }

        C[i] = (TestDataTypeUint)limb;
        carry = (temp - limb) / BASE;
    }

    if (carry != 0)
        C[L_C] = (TestDataTypeUint)carry;

    cleanup_ntt_context(ctx);
    auto t2 = chrono::high_resolution_clock::now();

    cleanup_ntt_precomputed(pre);
    cudaFreeHost(a_pinned);
    cudaFreeHost(b_pinned);

    // print t1-t0, t3-t2
    // cout << "[Host] NTT multiply time: " << chrono::duration<double, milli>(t1 - t0).count() << " ms\n";
    // cout << "[Host] CRT combine time: " << chrono::duration<double, milli>(t3 - t2).count() << " ms\n";
    // cout << "[Host] Carry Prop time: " << chrono::duration<double, milli>(t4 - t3).count() << " ms\n";

    std::ofstream csv("timings_new.csv", std::ios::app); // append mode

    // Write header only if file is empty/new
    if (csv.tellp() == 0) {
        csv << "N,ntt_multiply_ms,carry_prop_ms\n";
    }

    csv << N << "," 
        << chrono::duration<double, milli>(t1 - t0).count() << ","
        << chrono::duration<double, milli>(t2 - t1).count()
        << "\n";
    csv.close();

    duration = t2 - t0;
}
