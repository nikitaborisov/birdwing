#include "../include/gpu_cgbn.h"
#include "../include/gpu_ntt.h"
#include "../include/multiply.h"
#include "../include/crt.h"
#include "../include/crt_utils.h"
#include "config.h"
#include <chrono>
#include <fstream>

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

    vector<TestDataTypeUint> A_pad(N, 0), B_pad(N, 0);
    copy(A.begin(), A.end(), A_pad.begin());
    copy(B.begin(), B.end(), B_pad.begin());

    vector<vector<TestDataTypeUint>> C_recovered;

    NTTContext ctx = setup_ntt_context(N);

    auto t0 = chrono::high_resolution_clock::now();
    execute_ntt_multiply(ctx, A_pad, B_pad, C_recovered);
    auto t1 = chrono::high_resolution_clock::now();

    cleanup_ntt_context(ctx);

    #if DEBUG == 1
    for (size_t j = 0; j < NUM_MODULI; ++j) {
        cout << "[Host] Inverse NTT result mod " << moduli[j] << ": ";
        for (size_t i = 0; i < N; ++i) {
            cout << C_recovered[j][i] << " ";
        }
        cout << endl;
    }
    #endif
    
    auto t2 = chrono::high_resolution_clock::now();
    vector<__uint128_t> C_big(N, 0);

    for (size_t i = 0; i < N; ++i) {
        vector<TestDataTypeUint> residues(NUM_MODULI);
        for (size_t j = 0; j < NUM_MODULI; ++j)
            residues[j] = C_recovered[j][i];

        // print the residues for coefficient i
        #if DEBUG == 1
        cout << "[Host] Coefficient " << i << " residues: ";
        for (auto x : residues)
            cout << x << " ";
        cout << endl;
        #endif
        C_big[i] = crt_combine_many(moduli, residues); // reconstruct coefficient i
        __uint128_t M = 1;
        for (size_t j = 0; j < NUM_MODULI; ++j)
            M *= moduli[j];

        if (C_big[i] > M/2)
            C_big[i] -= M;
    }

    C.resize(L_C + 1, 0);
    auto t3 = chrono::high_resolution_clock::now();

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

    // trim
    while (C.size() > 1 && C.back() == 0)
        C.pop_back();

    // pad to length L_C if needed
    if (C.size() < (L_C + 1))
        C.resize(L_C + 1, 0);
    auto t4 = chrono::high_resolution_clock::now();

    // print t1-t0, t3-t2
    cout << "[Host] NTT multiply time: " << chrono::duration<double, milli>(t1 - t0).count() << " ms\n";
    cout << "[Host] CRT combine time: " << chrono::duration<double, milli>(t3 - t2).count() << " ms\n";
    cout << "[Host] Carry Prop time: " << chrono::duration<double, milli>(t4 - t3).count() << " ms\n";

    duration = t1 - t0 + t3 - t2;
}
