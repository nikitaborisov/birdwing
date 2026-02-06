// multiplication pipeline: split -> NTs modulo several primes -> pointwise mul
// -> CRT reconstruction -> carry propagation

// need to choose a limb / base b = 2^w, probably w = 32 to use CGBN
// A, B -> need to generate arrays of L limbs in base b. Convolution length <=
// L_A + L_B - 1 choose N - next power of 2 >= L_A + L_B - 1. Use GPU-NTT to
// compute transform of length N. We need several primes here. Single NTT module
// one prime p gives residues module p. We need to pick enough moduli p_i s. t.
// product p_i > (b-1)^2 * L. reconstruct each coefficient. CGBN can be used
// here. after crt reconstruction, do base b carry propagation.

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

using hires_clock = chrono::high_resolution_clock;
using ms = chrono::duration<double, std::milli>;

// parameters
// constexpr unsigned LIMB_BITS = 32;
using limb_t = TestDataTypeUint; // each limb is stored in 32-bit containers
// constexpr TestDataTypeUint BASE = (1ULL << LIMB_BITS); // base b = 2^w

void log_timing_csv(
    size_t L,
    double pad,
    double fwd_ntt,
    double pointwise,
    double inv_ntt,
    double crt,
    double carry,
    double total)
{
    static bool header_written = false;
    std::ofstream file("timing_results.csv", std::ios::app);

    if (!header_written) {
        file << "L,padding_ms,fwd_ntt_ms,pointwise_ms,inv_ntt_ms,crt_ms,carry_ms,total_ms\n";
        header_written = true;
    }

    file << L << ","
         << pad << ","
         << fwd_ntt << ","
         << pointwise << ","
         << inv_ntt << ","
         << crt << ","
         << carry << ","
         << total << "\n";
}

// host functions
void host_multiply_merge(const vector<limb_t> &A, const vector<limb_t> &B, vector<limb_t> &C) {
    auto t_start = hires_clock::now();
    size_t L_A = A.size();
    size_t L_B = B.size();
    size_t L_C = L_A + L_B - 1;  // not sure what the length of outputs will be? paper suggests L_A = L_B = L_C

    auto t0 = hires_clock::now();

    size_t N = 1;
    while (N < L_C)
        N <<= 1;

    vector<TestDataTypeUint> A_pad(N, 0), B_pad(N, 0);
    copy(A.begin(), A.end(), A_pad.begin());
    copy(B.begin(), B.end(), B_pad.begin());

    auto t1 = hires_clock::now();

    // ntt_merge_forward should return 4 versions of the NTT. don't do the for loop
    vector<vector<TestDataTypeUint>> A_mod(NUM_MODULI, vector<TestDataTypeUint>(N));
    vector<vector<TestDataTypeUint>> B_mod(NUM_MODULI, vector<TestDataTypeUint>(N));
    auto t2 = hires_clock::now();
    ntt_merge_forward(A_pad, A_mod);
    ntt_merge_forward(B_pad, B_mod);
    auto t3 = hires_clock::now();

    // pointwise multiplication for each of the elements of A_mod, B_mod
    vector<vector<TestDataTypeUint>> C_mod;
    auto t4 = hires_clock::now();
    gpu_pointwise_multiply(A_mod, B_mod, C_mod);
    auto t5 = hires_clock::now();
    // print pointwise multiplication results
    #if DEBUG == 1
    for (size_t j = 0; j < NUM_MODULI; ++j) {
        cout << "[Host] Pointwise multiplication mod " << moduli[j] << ": ";
        for (size_t i = 0; i < N; ++i) {
            cout << C_mod[j][i] << " ";
        }
        cout << endl;
    }
    #endif

    // gpu_ntt_inverse calls, should do the inverse ntt computation for all 4 C_mod vectors
    vector<vector<TestDataTypeUint>> C_recovered;
    auto t6 = hires_clock::now();
    gpu_ntt_inverse(C_mod, C_recovered);
    auto t7 = hires_clock::now();

    // ---------------- CRT Reconstruction ----------------
    // print recovered C_mod results
    #if DEBUG == 1
    for (size_t j = 0; j < NUM_MODULI; ++j) {
        cout << "[Host] Inverse NTT result mod " << moduli[j] << ": ";
        for (size_t i = 0; i < N; ++i) {
            cout << C_recovered[j][i] << " ";
        }
        cout << endl;
    }
    #endif

    auto t8 = hires_clock::now();
    // CRT recombination (CGBN)
    // vector<__uint128_t> C_big(N);
    // gpu_crt_reconstruct(C_mod, C_big, MODULI, NUM_MODULI);
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
            C_big[i] -= M;   // now signed
    }
    auto t9 = hires_clock::now();

    // ---------------- Carry Propagation ----------------
    auto t10 = hires_clock::now();

    C.resize(L_C + 1, 0);
    __int128 carry = 0;
    for (size_t i = 0; i < L_C; ++i) {
        __int128 temp = C_big[i] + carry;
        C[i] = (limb_t)(temp & 0xFFFFFFFF);
        carry = (limb_t)(temp >> 32);
    }
    if (carry != 0) C[L_C] = static_cast<limb_t>(carry);

    // trim
    while (C.size() > 1 && C.back() == 0)
        C.pop_back();

    auto t_end = hires_clock::now();

    // ---------------- Timing Report ----------------
    // cout << YELLOW << "\n[TIMING BREAKDOWN]\n" << RESET;

    // cout << "Padding:              " << ms(t1 - t0).count()  << " ms\n";
    // cout << "Forward NTT:          " << ms(t3 - t2).count()  << " ms\n";
    // cout << "Pointwise multiply:   " << ms(t5 - t4).count()  << " ms\n";
    // cout << "Inverse NTT:          " << ms(t7 - t6).count()  << " ms\n";
    // cout << "CRT reconstruction:   " << ms(t9 - t8).count()  << " ms\n";
    // cout << "Carry propagation:    " << ms(t_end - t10).count() << " ms\n";
    // cout << "TOTAL:                " << ms(t_end - t_start).count() << " ms\n";

    log_timing_csv(
        L_A,
        ms(t1 - t0).count(),
        ms(t3 - t2).count(),
        ms(t5 - t4).count(),
        ms(t7 - t6).count(),
        ms(t9 - t8).count(),
        ms(t_end - t10).count(),
        ms(t_end - t_start).count()
    );
}
