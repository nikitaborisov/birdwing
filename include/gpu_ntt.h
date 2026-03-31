#pragma once

#include "ntt.cuh"
#include "config.h"
#include "crt_gpu.h"
#include "carry_prop_serial.h"
#include <vector>

using namespace std;
using namespace gpuntt;

typedef Data32 TestDataType;

struct NTTContext {
    size_t N;
    int logN;

    TestDataTypeUint* a_raw_dev = nullptr;  // size L_A
    TestDataTypeUint* b_raw_dev = nullptr;  // size L_B
    size_t L_A = 0, L_B = 0;

    vector<TestDataType*> a_dev;
    vector<TestDataType*> b_dev;
    vector<TestDataType*> c_dev;

    vector<Root<TestDataType>*> forward_omega_dev;
    vector<Root<TestDataType>*> inverse_omega_dev;

    vector<Modulus<TestDataType>*> modulus_dev;
    vector<Ninverse<TestDataType>*> ninv_dev;

    cudaStream_t stream_a, stream_b;

    uint64_t* d_C_hi;
    uint64_t* d_C_lo;
    uint32_t* d_out;
    int64_t*  d_seg_carry;
};

struct NTTPrecomputed {
    size_t N;
    int logN;
    vector<NTTParameters<TestDataType>> params;
    vector<Root<TestDataType>*>     forward_omega_dev;
    vector<Root<TestDataType>*>     inverse_omega_dev;
    vector<Modulus<TestDataType>*>  modulus_dev;
    vector<Ninverse<TestDataType>*> ninv_dev;
    CRTGarnerParams garner;
};

NTTPrecomputed precompute_ntt(size_t N);
NTTContext allocate_ntt_context(const NTTPrecomputed &pre, size_t L_A, size_t L_B);

void execute_ntt_multiply(
    NTTContext &ctx,
    const TestDataTypeUint* a_pinned,
    const TestDataTypeUint* b_pinned,
    vector<TestDataTypeUint> &C_out,
    __int128 M, __int128 M_half
);

void cleanup_ntt_context(NTTContext &ctx);
void cleanup_ntt_precomputed(NTTPrecomputed &pre);