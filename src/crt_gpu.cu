#include "crt_gpu.h"
#include "config.h"
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>

using namespace std;

// Constant memory: Garner params visible to all threads
__constant__ uint64_t d_primes[NUM_MODULI];
__constant__ uint64_t d_inv[NUM_MODULI];
__constant__ uint64_t d_M_mod_table[NUM_MODULI][NUM_MODULI];
__constant__ uint64_t d_barrett_m[NUM_MODULI];

__constant__ TestDataTypeUint* d_residue_ptrs[NUM_MODULI];

// (hi, lo) += (b_hi, b_lo)
__device__ __forceinline__
void add128_128(uint64_t &hi, uint64_t &lo, uint64_t b_hi, uint64_t b_lo) {
    uint64_t old_lo = lo;
    lo += b_lo;
    hi += b_hi + (lo < old_lo ? 1ULL : 0ULL);
}

// (hi, lo) * scalar -> (hi, lo), safe because hi < 2^26 and scalar < 2^30
__device__ __forceinline__
void mul128_scalar(uint64_t &hi, uint64_t &lo, uint64_t b)
{
    unsigned __int128 full_lo =
        (unsigned __int128)lo * b;

    unsigned __int128 full_hi =
        (unsigned __int128)hi * b;

    uint64_t new_lo = (uint64_t)full_lo;

    uint64_t carry =
        (uint64_t)(full_lo >> 64);

    uint64_t new_hi =
        (uint64_t)(full_hi + carry);

    lo = new_lo;
    hi = new_hi;
}

__device__ __forceinline__
uint64_t mulmod64(uint64_t a, uint64_t b, uint64_t p, uint64_t m) {
#if LIMB_BITS == 64
    // a, b < p < 2^62: product fits in 128 bits
    uint64_t prod_lo = a * b;
    uint64_t prod_hi = __umul64hi(a, b);
    // Barrett: q = floor(prod / p) ≈ (prod * m) >> 64, but m is 128-bit here
    // Use __uint128_t for correctness
    unsigned __int128 prod = ((unsigned __int128)prod_hi << 64) | prod_lo;
    // TODO fix this to use barrett reduction later
    unsigned __int128 q    = prod / p;   // exact division is fine in host; on device use Barrett
    uint64_t r = (uint64_t)(prod - q * p);
    return r >= p ? r - p : r;
#else
    // original: a,b < p < 2^32, product fits in uint64_t
    uint64_t prod = a * b;
    uint64_t q    = __umul64hi(prod, m);
    uint64_t r    = prod - q * p;
    return r >= p ? r - p : r;
#endif
}

// Host to precompute Garner params
CRTGarnerParams compute_garner_params(const vector<TestDataTypeUint> &primes) {
    assert(primes.size() == NUM_MODULI);
    CRTGarnerParams p;

    unsigned __int128 M = 1;
    for (int i = 0; i < NUM_MODULI; i++) {
        // check if primes are corret size for chosen limb size
        #if LIMB_BITS == 64
            assert((uint64_t)primes[i] < (1ULL << 62) &&
            "primes must be < 2^62 for 64-bit mode");
        #else
            assert((uint64_t)primes[i] < (1ULL << 32) &&
            "primes must be < 2^32 for 32-bit mode");
        #endif

        uint64_t pi = (uint64_t)primes[i];
        // assert((pi - 1) <= (uint64_t)UINT32_MAX &&
        //    "prime too large for 64-bit product");
        p.primes[i]   = pi;
        p.inv[i]      = (i == 0) ? 1ULL : modinv_u64((uint64_t)(M % pi), pi);

        unsigned __int128 Mk = 1;
        for (int k = 0; k < NUM_MODULI; k++) {
            p.M_mod_table[k][i] = (uint64_t)(Mk % pi);  // M_k mod p_j
            Mk *= (unsigned __int128)primes[k];
        }
        #if LIMB_BITS == 32
            p.barrett_m[i] = (uint64_t)(((unsigned __int128)1 << 64) / (uint64_t)primes[i]);
        #else
            p.barrett_m[i] = 0;  // unused in 64-bit mode, division handled differently
        #endif

        M *= pi;
    }
    return p;
}


void upload_garner_params(const CRTGarnerParams &params) {
    cudaMemcpyToSymbol(d_primes,   params.primes,   NUM_MODULI * sizeof(uint64_t));
    cudaMemcpyToSymbol(d_inv,      params.inv,       NUM_MODULI * sizeof(uint64_t));
    cudaMemcpyToSymbol(d_M_mod_table, params.M_mod_table, NUM_MODULI * NUM_MODULI * sizeof(uint64_t));
    cudaMemcpyToSymbol(d_barrett_m, params.barrett_m, NUM_MODULI * sizeof(uint64_t));
}

void upload_residue_ptrs(const vector<TestDataTypeUint*> &c_dev) {
    TestDataTypeUint* h_ptrs[NUM_MODULI];
    for (int j = 0; j < NUM_MODULI; j++) h_ptrs[j] = c_dev[j];
    cudaMemcpyToSymbol(d_residue_ptrs, h_ptrs, NUM_MODULI * sizeof(TestDataTypeUint*));
}

// __global__
// void crt_combine_kernel(uint64_t* C_hi, uint64_t* C_lo, int N) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i >= N) return;

//     unsigned __int128 x = 0;
//     unsigned __int128 M = 1;

//     uint64_t x_mod[NUM_MODULI] = {};

//     #pragma unroll
//     for (int j = 0; j < NUM_MODULI; j++) {
//         uint64_t p   = d_primes[j];
//         uint64_t r   = (uint64_t)d_residue_ptrs[j][i];
//         uint64_t inv = d_inv[j];

//         uint64_t x_mod_p = x_mod[j];
//         uint64_t t   = (r >= x_mod_p) ? (r - x_mod_p) : (r + p - x_mod_p);
//         uint64_t k_i = mulmod64(t, inv, p, d_barrett_m[j]);

//         #pragma unroll
//         for (int k = j + 1; k < NUM_MODULI; k++) {
//             uint64_t pk      = d_primes[k];
//             uint64_t contrib = mulmod64(d_M_mod_table[j][k], k_i, pk, d_barrett_m[k]);
//             x_mod[k] += contrib;
//             if (x_mod[k] >= pk) x_mod[k] -= pk;
//         }

//         x += (unsigned __int128)(M * k_i);
//         M *= p;
//     }

//     C_hi[i] = (uint64_t)(x >> 64);
//     C_lo[i] = (uint64_t)(x);
// }

__global__
void crt_combine_kernel(uint64_t* __restrict__ C_hi, uint64_t* __restrict__ C_lo, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    uint64_t x_hi = 0, x_lo = 0;   // running Garner solution
    uint64_t M_hi = 0, M_lo = 1;   // running prefix product, starts at 1

    uint64_t x_mod[NUM_MODULI] = {};

    #pragma unroll
    for (int j = 0; j < NUM_MODULI; j++) {
        uint64_t p    = d_primes[j];

        // this is coalesced because for a fixed j, threads access consecutive i's
        uint64_t r = (uint64_t)d_residue_ptrs[j][i];
        uint64_t inv  = d_inv[j];

        // x_mod_p = current x mod p
        uint64_t x_mod_p = x_mod[j];

        // t = (r - x_mod_p) mod p
        uint64_t t = (r >= x_mod_p) ? (r - x_mod_p) : (r + p - x_mod_p);

        // k_i = t * inv mod p
        uint64_t k_i = mulmod64(t, inv, p, d_barrett_m[j]);

        #pragma unroll
        for (int k = j + 1; k < NUM_MODULI; k++) {
            uint64_t pk      = d_primes[k];
            uint64_t contrib = mulmod64(d_M_mod_table[j][k], k_i, pk, d_barrett_m[k]);
            x_mod[k] += contrib;
            if (x_mod[k] >= pk) x_mod[k] -= pk;   // single conditional, no division
        }

        // x += M * k_i
        uint64_t tmp_hi = M_hi, tmp_lo = M_lo;
        mul128_scalar(tmp_hi, tmp_lo, k_i);
        add128_128(x_hi, x_lo, tmp_hi, tmp_lo);

        // M *= p
        mul128_scalar(M_hi, M_lo, p);
    }

    C_hi[i] = x_hi;
    C_lo[i] = x_lo;
}

// Host to launch CRT kernel
void crt_combine_gpu(
    uint64_t *d_C_hi,   // pre-allocated device output
    uint64_t *d_C_lo,
    int N
) {
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    crt_combine_kernel<<<blocks, threads>>>(d_C_hi, d_C_lo, N);
    cudaDeviceSynchronize();
}