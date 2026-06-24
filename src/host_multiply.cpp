#include "../include/gpu_ntt.h"
#include "../include/multiply.h"
#include "../include/ntt_limits.h"
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
void host_multiply_merge(const vector<uint32_t> &A, const vector<uint32_t> &B, vector<TestDataTypeUint> &C, chrono::duration<double, milli> &duration) {
    size_t L_A = A.size();
    size_t L_B = B.size();
    size_t L_C = L_A + L_B - 1;

    size_t N = 1;
    while (N < L_C)
        N <<= 1;

    ensure_ntt_size_supported(N);

    uint32_t* a_pinned;
    uint32_t* b_pinned;
    cudaMallocHost(&a_pinned, L_A * sizeof(uint32_t));
    cudaMallocHost(&b_pinned, L_B * sizeof(uint32_t));
    memcpy(a_pinned, A.data(), L_A * sizeof(uint32_t));
    memcpy(b_pinned, B.data(), L_B * sizeof(uint32_t));

    vector<TestDataTypeUint> C_out(N + 1, 0);

    C.resize(L_C + 1, 0);

    unsigned __int128 M = 1;
    for (int j = 0; j < NUM_MODULI; j++) M *= moduli[j];
    __int128 M_half = M >> 1;

    // also compute M/2
    NTTPrecomputed pre = precompute_ntt(N);

    auto t0 = chrono::high_resolution_clock::now();
    NTTContext ctx = allocate_ntt_context(pre, L_A, L_B);
    execute_ntt_multiply(ctx, a_pinned, b_pinned, C_out, M, M_half);
    auto t1 = chrono::high_resolution_clock::now();

    for (size_t i = 0; i <= L_C; i++)
        C[i] = C_out[i];

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
