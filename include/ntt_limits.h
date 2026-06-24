#pragma once

#include "config.h"
#include <cstddef>
#include <cstdint>
#include <string>

// Largest power-of-two NTT size supported by the current moduli / roots tables.
int max_logN_for_prime(uint64_t p);
int max_root_logN(TestDataTypeUint root, TestDataTypeUint p);
int max_supported_logN();
size_t max_supported_N();

// Max operand limb count L (square inputs) s.t. padded NTT size stays valid.
size_t max_supported_limb_count();

size_t padded_ntt_size(size_t L_A, size_t L_B);

// N must be a power of two and within per-modulus root-of-unity order.
bool ntt_size_supported(size_t N, std::string* why = nullptr);

// Print to stderr and std::exit(1) if unsupported.
void ensure_ntt_size_supported(size_t N);
