#include "gpu_ntt.h"
#include "modular_arith.cuh"
#include "crt_gpu.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <ctime>

#include "ntt.cuh"
#include "config.h"

using namespace std;
using namespace gpuntt;

typedef Data32 TestDataType;

vector<TestDataTypeUint> moduli = {754974721, 595591169, 645922817};
vector<TestDataTypeUint> roots_of_unity_2_23 = {663, 721, 19};

static TestDataType mod_mul(long long a, long long b, long long mod) {
    return (a * b) % mod;
}

// helper to generate new factors table compatible with given N
static array<NTTFactors<TestDataType>, NUM_MODULI> generate_factors_for_N(int logN) {
    // we want the (2^logN - 1) and 2^logN th roots of unity from 2^23rd roots
    array<NTTFactors<TestDataType>, NUM_MODULI> new_factors;
    for (int i = 0; i < NUM_MODULI; i++) {
        TestDataType root_2_23 = roots_of_unity_2_23[i];
        // now need to square this root (23 - logN) times to get the 2^logN th root
        TestDataType root_2_logN = root_2_23;
        for (int j = 0; j < (23 - logN); j++) {
            root_2_logN = mod_mul(root_2_logN, root_2_logN, moduli[i]);
        }
        new_factors[i] = {moduli[i], root_2_logN, mod_mul(root_2_logN, root_2_logN, moduli[i])};
    }
    return new_factors;
}

template <typename T>
__device__ __forceinline__ void mul_wide(T a, T b, T &lo, T &hi)
{
    if constexpr (sizeof(T) == 4)
    {
        // 32-bit multiply -> 64-bit product
        uint64_t prod = (uint64_t)a * (uint64_t)b;
        lo = (T)prod;            // low 32 bits
        hi = (T)(prod >> 32);    // high 32 bits
    }
    else if constexpr (sizeof(T) == 8)
    {
        // 64-bit multiply -> 128-bit product
        lo = a * b;             // low 64 bits
        hi = __umul64hi(a, b);  // high 64 bits
    }
    else {
        static_assert(sizeof(T) == 4 || sizeof(T) == 8,
                      "Unsupported TestDataTypeUint size.");
    }
}

__global__ void pointwise_mul_kernel(TestDataTypeUint* A,
                                     TestDataTypeUint* B,
                                     TestDataTypeUint* C,
                                     TestDataTypeUint modulus,
                                     size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        TestDataTypeUint lo, hi;
        mul_wide(A[idx], B[idx], lo, hi);

        if constexpr (sizeof(TestDataTypeUint) == 4) {
            uint64_t full = ((uint64_t)hi << 32) | lo;
            C[idx] = (TestDataTypeUint)(full % modulus);
        } else {
            unsigned __int128 full =
                ((unsigned __int128)hi << 64) | lo;
            C[idx] = (TestDataTypeUint)(full % modulus);
        }
    }
}

NTTContext setup_ntt_context(size_t N) {
    NTTContext ctx;
    ctx.N = N;
    ctx.logN = log2((int)N);

    ctx.params.resize(NUM_MODULI);
    ctx.a_dev.resize(NUM_MODULI);
    ctx.b_dev.resize(NUM_MODULI);
    ctx.c_dev.resize(NUM_MODULI);
    ctx.forward_omega_dev.resize(NUM_MODULI);
    ctx.inverse_omega_dev.resize(NUM_MODULI);
    ctx.modulus_dev.resize(NUM_MODULI);
    ctx.ninv_dev.resize(NUM_MODULI);

    auto factors = generate_factors_for_N(ctx.logN);

    for (int i = 0; i < NUM_MODULI; i++) {

        ctx.params[i] =
        NTTParameters<TestDataType>(
            ctx.logN,
            factors[i],
            ReductionPolynomial::X_N_minus);

        auto &p = ctx.params[i];

        // allocate data buffers
        cudaMalloc(&ctx.a_dev[i], p.n * sizeof(TestDataType));
        cudaMalloc(&ctx.b_dev[i], p.n * sizeof(TestDataType));
        cudaMalloc(&ctx.c_dev[i], p.n * sizeof(TestDataType));

        // forward omega
        cudaMalloc(&ctx.forward_omega_dev[i],
                   p.root_of_unity_size * sizeof(Root<TestDataType>));

        auto forward_table =
            p.gpu_root_of_unity_table_generator(p.forward_root_of_unity_table);

        cudaMemcpy(ctx.forward_omega_dev[i],
                   forward_table.data(),
                   p.root_of_unity_size * sizeof(Root<TestDataType>),
                   cudaMemcpyHostToDevice);

        // inverse omega
        cudaMalloc(&ctx.inverse_omega_dev[i],
                   p.root_of_unity_size * sizeof(Root<TestDataType>));

        auto inverse_table =
            p.gpu_root_of_unity_table_generator(p.inverse_root_of_unity_table);

        cudaMemcpy(ctx.inverse_omega_dev[i],
                   inverse_table.data(),
                   p.root_of_unity_size * sizeof(Root<TestDataType>),
                   cudaMemcpyHostToDevice);

        // modulus
        cudaMalloc(&ctx.modulus_dev[i], sizeof(Modulus<TestDataType>));
        Modulus<TestDataType> mod_host[1] = {p.modulus};
        cudaMemcpy(ctx.modulus_dev[i], mod_host,
                   sizeof(Modulus<TestDataType>),
                   cudaMemcpyHostToDevice);

        // n inverse
        cudaMalloc(&ctx.ninv_dev[i], sizeof(Ninverse<TestDataType>));
        Ninverse<TestDataType> ninv_host[1] = {p.n_inv};
        cudaMemcpy(ctx.ninv_dev[i], ninv_host,
                   sizeof(Ninverse<TestDataType>),
                   cudaMemcpyHostToDevice);
    }

    cudaMalloc(&ctx.d_C_hi, N * sizeof(uint64_t));
    cudaMalloc(&ctx.d_C_lo, N * sizeof(uint64_t));

    cudaDeviceSynchronize();
    return ctx;
}

void execute_ntt_multiply(
    NTTContext &ctx,
    const vector<TestDataTypeUint> &a,
    const vector<TestDataTypeUint> &b,
    vector<uint64_t> &C_hi,
    vector<uint64_t> &C_lo)
{
    vector<TestDataType> a32(a.begin(), a.end());
    vector<TestDataType> b32(b.begin(), b.end());

    for (int i = 0; i < NUM_MODULI; i++) {

        auto &p = ctx.params[i];

        cudaMemcpy(ctx.a_dev[i], a32.data(),
                   p.n * sizeof(TestDataType),
                   cudaMemcpyHostToDevice);

        cudaMemcpy(ctx.b_dev[i], b32.data(),
                   p.n * sizeof(TestDataType),
                   cudaMemcpyHostToDevice);

        ntt_rns_configuration<TestDataType> cfg_fwd = {
            .n_power = ctx.logN,
            .ntt_type = FORWARD,
            .ntt_layout = PerPolynomial,
            .reduction_poly = ReductionPolynomial::X_N_minus,
            .zero_padding = false,
            .stream = 0
        };

        GPU_NTT_Inplace(ctx.a_dev[i],
                        ctx.forward_omega_dev[i],
                        ctx.modulus_dev[i],
                        cfg_fwd,
                        BATCH, 1);

        GPU_NTT_Inplace(ctx.b_dev[i],
                        ctx.forward_omega_dev[i],
                        ctx.modulus_dev[i],
                        cfg_fwd,
                        BATCH, 1);

        // pointwise
        int threads = 256;
        int blocks = (ctx.N + threads - 1) / threads;

        pointwise_mul_kernel<<<blocks, threads>>>(
            ctx.a_dev[i],
            ctx.b_dev[i],
            ctx.c_dev[i],
            moduli[i],
            ctx.N);
    }

    // inverse
    for (int i = 0; i < NUM_MODULI; i++) {

        ntt_rns_configuration<TestDataType> cfg_inv = {
            .n_power = ctx.logN,
            .ntt_type = INVERSE,
            .ntt_layout = PerPolynomial,
            .reduction_poly = ReductionPolynomial::X_N_minus,
            .zero_padding = false,
            .mod_inverse = ctx.ninv_dev[i],
            .stream = 0
        };

        GPU_INTT(ctx.c_dev[i],
                 ctx.c_dev[i],
                 ctx.inverse_omega_dev[i],
                 ctx.modulus_dev[i],
                 cfg_inv,
                 BATCH, 1);
    }

    cudaDeviceSynchronize();

    // ctx.c_dev[i] holds INTT results — pass directly to CRT, no host round-trip
    CRTGarnerParams garner = compute_garner_params(moduli);
    crt_combine_gpu(ctx.c_dev, ctx.d_C_hi, ctx.d_C_lo, garner, ctx.N);

    // only copy the final 128-bit result back, not the intermediate residues
    C_hi.resize(ctx.N);
    C_lo.resize(ctx.N);
    cudaMemcpy(C_hi.data(), ctx.d_C_hi, ctx.N * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(C_lo.data(), ctx.d_C_lo, ctx.N * sizeof(uint64_t), cudaMemcpyDeviceToHost);
}

void cleanup_ntt_context(NTTContext &ctx) {
    for (int i = 0; i < NUM_MODULI; i++) {
        cudaFree(ctx.a_dev[i]);
        cudaFree(ctx.b_dev[i]);
        cudaFree(ctx.c_dev[i]);
        cudaFree(ctx.forward_omega_dev[i]);
        cudaFree(ctx.inverse_omega_dev[i]);
        cudaFree(ctx.modulus_dev[i]);
        cudaFree(ctx.ninv_dev[i]);
    }
    cudaFree(ctx.d_C_hi);
    cudaFree(ctx.d_C_lo);
}