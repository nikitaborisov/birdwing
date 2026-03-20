#pragma once

#include "ntt.cuh"
#include "config.h"
#include "crt_gpu.h"
#include <vector>

using namespace std;
using namespace gpuntt;

typedef Data32 TestDataType;

struct NTTContext {
    size_t N;
    int logN;

    vector<NTTParameters<TestDataType>> params;

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

    uint64_t* d_C_hi = nullptr;
    uint64_t* d_C_lo = nullptr;

    CRTGarnerParams garner;
};

NTTContext setup_ntt_context(size_t N, size_t L_A, size_t L_B);

void execute_ntt_multiply(
    NTTContext &ctx,
    const vector<TestDataTypeUint> &a,
    const vector<TestDataTypeUint> &b,
    vector<uint64_t> &C_hi,
    vector<uint64_t> &C_lo
);

void cleanup_ntt_context(NTTContext &ctx);