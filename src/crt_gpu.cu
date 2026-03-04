#include "crt_gpu.h"
#include "config.h"
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>

// Constant memory: Garner params visible to all threads
__constant__ uint64_t d_primes[NUM_MODULI];
__constant__ uint64_t d_inv[NUM_MODULI];
__constant__ uint64_t d_prefix_M[NUM_MODULI];

// (hi, lo) += b,  where hi is small (~26 bits) so no hi overflow
__device__ __forceinline__
void add128(uint64_t &hi, uint64_t &lo, uint64_t b) {
    uint64_t old_lo = lo;
    lo += b;
    if (lo < old_lo) hi++;
}

// (hi, lo) += (b_hi, b_lo)
__device__ __forceinline__
void add128_128(uint64_t &hi, uint64_t &lo, uint64_t b_hi, uint64_t b_lo) {
    uint64_t old_lo = lo;
    lo += b_lo;
    hi += b_hi + (lo < old_lo ? 1ULL : 0ULL);
}

// (hi, lo) * scalar -> (hi, lo), safe because hi < 2^26 and scalar < 2^30
__device__ __forceinline__
void mul128_scalar(uint64_t &hi, uint64_t &lo, uint64_t b) {
    uint64_t lo_hi = __umul64hi(lo, b);
    uint64_t lo_lo = lo * b;
    hi = hi * b + lo_hi;   // hi*b can't overflow: hi<2^26, b<2^30 -> product<2^56
    lo = lo_lo;
}

// (hi << 64 | lo) % p,  using __uint128_t which is valid in device code
__device__ __forceinline__
uint64_t mod128(uint64_t hi, uint64_t lo, uint64_t p) {
    unsigned __int128 val = ((unsigned __int128)hi << 64) | lo;
    return (uint64_t)(val % p);
}

// (a * b) % p, safe via __uint128_t
__device__ __forceinline__
uint64_t mulmod64(uint64_t a, uint64_t b, uint64_t p) {
    return (uint64_t)((unsigned __int128)a * b % p);
}

// Kernel: one thread per coefficient
__global__
void crt_combine_kernel(
    const TestDataTypeUint* __restrict__ residues,  // [NUM_MODULI * N]
    uint64_t* __restrict__ C_hi,
    uint64_t* __restrict__ C_lo,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    uint64_t x_hi = 0, x_lo = 0;   // running Garner solution
    uint64_t M_hi = 0, M_lo = 1;   // running prefix product, starts at 1

    #pragma unroll
    for (int j = 0; j < NUM_MODULI; j++) {
        uint64_t p   = d_primes[j];
        uint64_t r   = (uint64_t)residues[j * N + i];   // residue for this coeff, this modulus
        uint64_t inv = d_inv[j];

        // x_mod_p = current x mod p
        uint64_t x_mod_p = mod128(x_hi, x_lo, p);

        // t = (r - x_mod_p) mod p
        uint64_t t = (r >= x_mod_p) ? (r - x_mod_p) : (r + p - x_mod_p);

        // k_i = t * inv mod p
        uint64_t k_i = mulmod64(t, inv, p);

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

// Host to precompute Garner params
CRTGarnerParams compute_garner_params(const vector<TestDataTypeUint> &primes) {
    assert(primes.size() == NUM_MODULI);
    CRTGarnerParams p;

    unsigned __int128 M = 1;
    for (int i = 0; i < NUM_MODULI; i++) {
        uint64_t pi = (uint64_t)primes[i];
        p.primes[i]   = pi;
        p.prefix_M[i] = (uint64_t)(M % pi);
        p.inv[i]      = (i == 0) ? 1ULL : modinv_u64(p.prefix_M[i], pi);
        M *= pi;
    }
    return p;
}

// Host to launch CRT kernel
void crt_combine_gpu(
    const vector<vector<TestDataTypeUint>> &residues,
    vector<uint64_t> &C_hi,
    vector<uint64_t> &C_lo,
    const CRTGarnerParams &params,
    int N
) {
    // Upload Garner params to constant memory (once per call -- cheap)
    cudaMemcpyToSymbol(d_primes,   params.primes,   NUM_MODULI * sizeof(uint64_t));
    cudaMemcpyToSymbol(d_inv,      params.inv,      NUM_MODULI * sizeof(uint64_t));
    cudaMemcpyToSymbol(d_prefix_M, params.prefix_M, NUM_MODULI * sizeof(uint64_t));

    // Flatten residues: [NUM_MODULI][N] -> single device array
    size_t total = (size_t)NUM_MODULI * N;
    std::vector<TestDataTypeUint> h_residues_flat(total);
    for (int j = 0; j < NUM_MODULI; j++)
        for (int i = 0; i < N; i++)
            h_residues_flat[j * N + i] = residues[j][i];

    // Allocate device memory
    TestDataTypeUint *d_residues;
    uint64_t *d_C_hi, *d_C_lo;
    cudaMalloc(&d_residues, total * sizeof(TestDataTypeUint));
    cudaMalloc(&d_C_hi,     N * sizeof(uint64_t));
    cudaMalloc(&d_C_lo,     N * sizeof(uint64_t));

    // Copy residues to device
    cudaMemcpy(d_residues, h_residues_flat.data(), total * sizeof(TestDataTypeUint), cudaMemcpyHostToDevice);

    // Launch
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    crt_combine_kernel<<<blocks, threads>>>(d_residues, d_C_hi, d_C_lo, N);
    cudaDeviceSynchronize();

    // Copy results back
    C_hi.resize(N);
    C_lo.resize(N);
    cudaMemcpy(C_hi.data(), d_C_hi, N * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(C_lo.data(), d_C_lo, N * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    cudaFree(d_residues);
    cudaFree(d_C_hi);
    cudaFree(d_C_lo);
}