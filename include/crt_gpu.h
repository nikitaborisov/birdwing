#pragma once

#include "config.h"
#include "crt_utils.h"
#include <vector>

using namespace std;

// Precomputed Garner constants for GPU CRT (host-side, uploaded to constant memory)
// use cudaMemcpyToSymbol
struct CRTGarnerParams {
    uint64_t primes[NUM_MODULI];
    uint64_t inv[NUM_MODULI];      // inv[i] = inverse of (p0*...*p_{i-1}) mod p_i
    uint64_t prefix_M[NUM_MODULI]; // prefix_M[i] = (p0*...*p_{i-1}) mod p_i
};

// Precompute Garner params from a list of primes (runs on host)
CRTGarnerParams compute_garner_params(const vector<TestDataTypeUint> &primes);

void upload_garner_params(const CRTGarnerParams &params);

void upload_residue_ptrs(const vector<TestDataTypeUint*> &c_dev);

// GPU CRT: combines residues for N coefficients in parallel.
// residues layout: [NUM_MODULI][N] (row = modulus, col = coefficient index)
// Output C_hi, C_lo: each coefficient x = (C_hi[i] << 64) | C_lo[i]
void crt_combine_gpu(
    uint64_t *d_C_hi,   // pre-allocated device output
    uint64_t *d_C_lo,
    int N
);