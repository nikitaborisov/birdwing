// crt.h
#pragma once
#include <cstdint>
#include <vector>
#include "crt_utils.h"

using namespace std;

// struct holding parameters for CRT with 2 primes
struct CRT2Params {
    TestDataTypeUint p1;
    TestDataTypeUint p2;
    TestDataTypeUint p1_inv_mod_p2;
    unsigned __int128 modulus;

    CRT2Params(TestDataTypeUint _p1, TestDataTypeUint _p2);
};

// combine residues modulo two primes
unsigned __int128 crt_combine_2(
    const CRT2Params &params,
    TestDataTypeUint a_mod_p1,
    TestDataTypeUint b_mod_p2
);

// general CRT for many primes
unsigned __int128 crt_combine_many(
    const vector<TestDataTypeUint> &primes,
    const vector<TestDataTypeUint> &residues
);
