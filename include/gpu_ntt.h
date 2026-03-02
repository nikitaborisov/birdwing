#pragma once

#include "ntt.cuh"
#include "config.h"
#include <vector>

using namespace std;
using namespace gpuntt;

typedef Data32 TestDataType;

struct NTTContext {
    size_t N;
    int logN;

    vector<NTTParameters<TestDataType>> params;

    vector<TestDataType*> a_dev;
    vector<TestDataType*> b_dev;
    vector<TestDataType*> c_dev;

    vector<Root<TestDataType>*> forward_omega_dev;
    vector<Root<TestDataType>*> inverse_omega_dev;

    vector<Modulus<TestDataType>*> modulus_dev;
    vector<Ninverse<TestDataType>*> ninv_dev;
};

NTTContext setup_ntt_context(size_t N);

void execute_ntt_multiply(
    NTTContext &ctx,
    const vector<TestDataTypeUint> &a,
    const vector<TestDataTypeUint> &b,
    vector<vector<TestDataTypeUint>> &c_recovered
);

void cleanup_ntt_context(NTTContext &ctx);