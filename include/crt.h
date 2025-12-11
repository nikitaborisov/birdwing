// crt.h
#pragma once
#include <cstdint>
#include <vector>
#include "crt_utils.h"

using namespace std;

// struct holding parameters for CRT with 2 primes
struct CRT2Params {
    uint64_t p1;
    uint64_t p2;
    uint64_t p1_inv_mod_p2;
    unsigned __int128 modulus;

    CRT2Params(uint64_t _p1, uint64_t _p2);
};

// combine residues modulo two primes
unsigned __int128 crt_combine_2(
    const CRT2Params &params,
    uint64_t a_mod_p1,
    uint64_t b_mod_p2
);

// general CRT for many primes
unsigned __int128 crt_combine_many(
    const vector<uint64_t> &primes,
    const vector<uint64_t> &residues
);
