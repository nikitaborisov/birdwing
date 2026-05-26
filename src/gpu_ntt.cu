#include "gpu_ntt.h"
#include "modular_arith.cuh"
#include "zero_pad.h"
#include "carry_prop.h"
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

#if LIMB_BITS == 64
  typedef Data64 TestDataType;
#else
  typedef Data32 TestDataType;
#endif

#if LIMB_BITS == 64
  // 62-bit NTT-friendly primes: p = k * 2^M + 1, M >= 23
  vector<TestDataTypeUint> moduli = {0x6723cbb800001, 0x6723cb6800001};
  vector<TestDataTypeUint> roots_of_unity_2_23 = {11, 6};
#else
  vector<TestDataTypeUint> moduli = {0x23800001, 0x26800001, 0x2d000001};
  vector<TestDataTypeUint> roots_of_unity_2_23 = {663, 721, 19};
#endif

struct GPUTimer {
    cudaEvent_t start, stop;

    GPUTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GPUTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void tic(cudaStream_t stream = 0) {
        cudaEventRecord(start, stream);
    }

    float toc(cudaStream_t stream = 0) {
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

// helper for modular multiplication; promotes to 64 / 128 bits to prevent overflow and then reduces mod
static TestDataType mod_mul(TestDataTypeUint a, TestDataTypeUint b, TestDataTypeUint mod) {
    if constexpr (sizeof(TestDataType) == 4) {
        return (TestDataType)((uint64_t)a * b % mod);
    } else {
        return (TestDataType)((__uint128_t)a * b % mod);
    }
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

__global__ void pointwise_mul_kernel(TestDataType* A,
                                     TestDataType* B,
                                     TestDataType* C,
                                     TestDataTypeUint modulus,
                                     size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        TestDataTypeUint a = (TestDataTypeUint)A[idx];
        TestDataTypeUint b = (TestDataTypeUint)B[idx];
        TestDataTypeUint lo, hi;
        mul_wide(a, b, lo, hi);

        if constexpr (sizeof(TestDataTypeUint) == 4) {
            uint64_t full = ((uint64_t)hi << 32) | lo;
            C[idx] = (TestDataTypeUint)(full % modulus);
        } else {
            unsigned __int128 full =
                ((unsigned __int128)hi << 64) | lo;
            C[idx] = (TestDataType)(full % modulus);
        }
    }
}

NTTPrecomputed precompute_ntt(size_t N) {
    NTTPrecomputed pre;
    pre.N    = N;
    pre.logN = (int)log2((double)N);

    pre.params.resize(NUM_MODULI);
    pre.forward_omega_dev.resize(NUM_MODULI);
    pre.inverse_omega_dev.resize(NUM_MODULI);
    pre.modulus_dev.resize(NUM_MODULI);
    pre.ninv_dev.resize(NUM_MODULI);

    auto factors = generate_factors_for_N(pre.logN);

    for (int i = 0; i < NUM_MODULI; i++) {
        pre.params[i] = NTTParameters<TestDataType>(
            pre.logN, factors[i], ReductionPolynomial::X_N_minus);
        auto &p = pre.params[i];

        // forward omega
        auto fwd = p.gpu_root_of_unity_table_generator(p.forward_root_of_unity_table);
        cudaMalloc(&pre.forward_omega_dev[i],
                   p.root_of_unity_size * sizeof(Root<TestDataType>));
        cudaMemcpy(pre.forward_omega_dev[i], fwd.data(),
                   p.root_of_unity_size * sizeof(Root<TestDataType>),
                   cudaMemcpyHostToDevice);

        // inverse omega
        auto inv = p.gpu_root_of_unity_table_generator(p.inverse_root_of_unity_table);
        cudaMalloc(&pre.inverse_omega_dev[i],
                   p.root_of_unity_size * sizeof(Root<TestDataType>));
        cudaMemcpy(pre.inverse_omega_dev[i], inv.data(),
                   p.root_of_unity_size * sizeof(Root<TestDataType>),
                   cudaMemcpyHostToDevice);

        // modulus
        cudaMalloc(&pre.modulus_dev[i], sizeof(Modulus<TestDataType>));
        Modulus<TestDataType> mod_host[1] = {p.modulus};
        cudaMemcpy(pre.modulus_dev[i], mod_host,
                   sizeof(Modulus<TestDataType>), cudaMemcpyHostToDevice);

        // n inverse
        cudaMalloc(&pre.ninv_dev[i], sizeof(Ninverse<TestDataType>));
        Ninverse<TestDataType> ninv_host[1] = {p.n_inv};
        cudaMemcpy(pre.ninv_dev[i], ninv_host,
                   sizeof(Ninverse<TestDataType>), cudaMemcpyHostToDevice);
    }

    pre.garner = compute_garner_params(moduli);
    upload_garner_params(pre.garner);

    cudaDeviceSynchronize();
    return pre;
}

// now allocate only owns the mutable buffers
NTTContext allocate_ntt_context(const NTTPrecomputed &pre, size_t L_A, size_t L_B) {
    NTTContext ctx;
    ctx.N    = pre.N;
    ctx.logN = pre.logN;
    ctx.L_A  = L_A;
    ctx.L_B  = L_B;

    ctx.a_dev.resize(NUM_MODULI);
    ctx.b_dev.resize(NUM_MODULI);
    ctx.c_dev.resize(NUM_MODULI);

    // borrow read-only pointers — no copy, no new allocation
    ctx.forward_omega_dev = pre.forward_omega_dev;
    ctx.inverse_omega_dev = pre.inverse_omega_dev;
    ctx.modulus_dev       = pre.modulus_dev;
    ctx.ninv_dev          = pre.ninv_dev;

    for (int i = 0; i < NUM_MODULI; i++) {
        const auto &p = pre.params[i];

        cudaMalloc(&ctx.a_dev[i], p.n * sizeof(TestDataType));
        cudaMalloc(&ctx.b_dev[i], p.n * sizeof(TestDataType));
        cudaMalloc(&ctx.c_dev[i], p.n * sizeof(TestDataType));
    }

    cudaMalloc(&ctx.a_raw_dev, L_A * sizeof(uint32_t));
    cudaMalloc(&ctx.b_raw_dev, L_B * sizeof(uint32_t));
    cudaMalloc(&ctx.d_C_hi, pre.N * sizeof(uint64_t));
    cudaMalloc(&ctx.d_C_lo, pre.N * sizeof(uint64_t));
    cudaMalloc(&ctx.d_out,  (pre.N + 1) * sizeof(TestDataTypeUint));
    size_t num_segs = (pre.N + CARRY_SEG - 1) / CARRY_SEG;
    cudaMalloc(&ctx.d_seg_carry, num_segs * sizeof(int64_t));

    cudaStreamCreate(&ctx.stream_a);
    cudaStreamCreate(&ctx.stream_b);

    upload_residue_ptrs(ctx.c_dev);

    cudaDeviceSynchronize();
    return ctx;
}

void execute_ntt_multiply(
    NTTContext &ctx,
    const uint32_t* a_pinned,
    const uint32_t* b_pinned,
    vector<TestDataTypeUint> &C_out,
    __int128 M, __int128 M_half)
{

    GPUTimer timer;

    #ifdef TIMING
    float t_ntt   = 0.0f;
    float t_mul   = 0.0f;
    float t_intt  = 0.0f;
    float t_crt   = 0.0f;
    float t_carry = 0.0f;
    #endif

    timer.tic(ctx.stream_a);

    cudaMemcpyAsync(ctx.a_raw_dev, a_pinned,
                ctx.L_A * sizeof(uint32_t),
                cudaMemcpyHostToDevice, ctx.stream_a);
    cudaMemcpyAsync(ctx.b_raw_dev, b_pinned,
                ctx.L_B * sizeof(uint32_t),
                cudaMemcpyHostToDevice, ctx.stream_b);

    for (int i = 0; i < NUM_MODULI; i++) {
        ntt_rns_configuration<TestDataType> cfg_a = {
            .n_power = ctx.logN,
            .ntt_type = FORWARD,
            .ntt_layout = PerPolynomial,
            .reduction_poly = ReductionPolynomial::X_N_minus,
            .zero_padding = false,
            .stream = ctx.stream_a
        };
        ntt_rns_configuration<TestDataType> cfg_b = {
            .n_power = ctx.logN,
            .ntt_type = FORWARD,
            .ntt_layout = PerPolynomial,
            .reduction_poly = ReductionPolynomial::X_N_minus,
            .zero_padding = false,
            .stream = ctx.stream_b
        };
        zero_pad_gpu(ctx.a_raw_dev, ctx.a_dev[i], ctx.L_A, ctx.N, ctx.stream_a);
        zero_pad_gpu(ctx.b_raw_dev, ctx.b_dev[i], ctx.L_B, ctx.N, ctx.stream_b);
        GPU_NTT_Inplace(ctx.a_dev[i], ctx.forward_omega_dev[i],
                        ctx.modulus_dev[i], cfg_a, BATCH, 1);

        GPU_NTT_Inplace(ctx.b_dev[i], ctx.forward_omega_dev[i],
                        ctx.modulus_dev[i], cfg_b, BATCH, 1);
    }

    cudaStreamSynchronize(ctx.stream_a);
    cudaStreamSynchronize(ctx.stream_b);

    #ifdef TIMING
    t_ntt = timer.toc(ctx.stream_a);
    #endif

    timer.tic(ctx.stream_a);

    for (int i = 0; i < NUM_MODULI; i++) {
        int threads = 256;
        int blocks = (ctx.N + threads - 1) / threads;
        pointwise_mul_kernel<<<blocks, threads, 0, ctx.stream_a>>>(
            ctx.a_dev[i], ctx.b_dev[i], ctx.c_dev[i], moduli[i], ctx.N);
    }
    #ifdef TIMING
    t_mul = timer.toc(ctx.stream_a);
    #endif

    timer.tic(ctx.stream_a);

    // inverse
    for (int i = 0; i < NUM_MODULI; i++) {
        ntt_rns_configuration<TestDataType> cfg_inv = {
            .n_power = ctx.logN,
            .ntt_type = INVERSE,
            .ntt_layout = PerPolynomial,
            .reduction_poly = ReductionPolynomial::X_N_minus,
            .zero_padding = false,
            .mod_inverse = ctx.ninv_dev[i],
            .stream = ctx.stream_a
        };

        GPU_INTT(ctx.c_dev[i],
                 ctx.c_dev[i],
                 ctx.inverse_omega_dev[i],
                 ctx.modulus_dev[i],
                 cfg_inv,
                 BATCH, 1);
    }

    cudaStreamSynchronize(ctx.stream_a);

    #ifdef TIMING
    t_intt = timer.toc(ctx.stream_a);
    #endif

    timer.tic(ctx.stream_a);

    // ctx.c_dev[i] holds INTT results — pass directly to CRT, no host round-trip
    crt_combine_gpu(ctx.d_C_hi, ctx.d_C_lo, ctx.N);

    #ifdef TIMING
    t_crt = timer.toc(ctx.stream_a);
    #endif

    timer.tic(ctx.stream_a);

    size_t num_segs = (ctx.N + CARRY_SEG - 1) / CARRY_SEG;

    carry_intra_segment_kernel<<<num_segs, 1, 0, ctx.stream_a>>>(
        ctx.d_C_hi, ctx.d_C_lo, ctx.d_out, ctx.d_seg_carry, ctx.N, M, M_half);

    carry_inter_segment_kernel<<<1, 1, 0, ctx.stream_a>>>(
        ctx.d_seg_carry, num_segs);

    carry_fixup_kernel<<<num_segs, 1, 0, ctx.stream_a>>>(
        ctx.d_out, ctx.d_seg_carry, ctx.N);

    cudaStreamSynchronize(ctx.stream_a);

    #ifdef TIMING
    t_carry = timer.toc(ctx.stream_a);
    #endif

    cudaMemcpy(C_out.data(), ctx.d_out,
            (ctx.N + 1) * sizeof(TestDataTypeUint), cudaMemcpyDeviceToHost);

    #ifdef TIMING
    static bool header_written = false;

    ofstream file("ntt_timing.csv", ios::app);

    if (!header_written) {
        file << "N,L_A,L_B,NTT,MUL,INTT,CRT,CARRY,TOTAL\n";
        header_written = true;
    }

    file << ctx.N << ","
        << ctx.L_A << ","
        << ctx.L_B << ","
        << t_ntt << ","
        << t_mul << ","
        << t_intt << ","
        << t_crt << ","
        << t_carry << "\n";
    
    #endif
}

void cleanup_ntt_context(NTTContext &ctx) {
    for (int i = 0; i < NUM_MODULI; i++) {
        cudaFree(ctx.a_dev[i]);
        cudaFree(ctx.b_dev[i]);
        cudaFree(ctx.c_dev[i]);
    }
    cudaStreamDestroy(ctx.stream_a);
    cudaStreamDestroy(ctx.stream_b);
    cudaFree(ctx.a_raw_dev);
    cudaFree(ctx.b_raw_dev);
    cudaFree(ctx.d_C_hi);
    cudaFree(ctx.d_C_lo);
    cudaFree(ctx.d_out);
    cudaFree(ctx.d_seg_carry);
}

void cleanup_ntt_precomputed(NTTPrecomputed &pre) {
    for (int i = 0; i < NUM_MODULI; i++) {
        cudaFree(pre.forward_omega_dev[i]);
        cudaFree(pre.inverse_omega_dev[i]);
        cudaFree(pre.modulus_dev[i]);
        cudaFree(pre.ninv_dev[i]);
    }
}