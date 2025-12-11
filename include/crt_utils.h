#pragma once
#include <cstdint>
#include "config.h"

using namespace std;

// helper for printing __int128
void print_u128(unsigned __int128 x);

// Modular inverse (64-bit), can be removed later if we decide to hardcode inverses (and primes)
uint64_t modinv_u64(uint64_t a, uint64_t m);