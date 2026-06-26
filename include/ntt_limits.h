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

// Max operand limb count L (square inputs) s.t. both NTT and CRT bounds hold.
size_t max_supported_limb_count();

// Max L for which worst-case int64 segment carries in carry_fixup stay in range.
// Native (u160) path returns SIZE_MAX. Hybrid caps at 2^31 (below NTT/CRT L=2^32).
size_t max_limb_count_for_int64_segment_carry();

size_t padded_ntt_size(size_t L_A, size_t L_B);

// N must be a power of two and within per-modulus root-of-unity order.
bool ntt_size_supported(size_t N, std::string* why = nullptr);

// min(L_A, L_B) * (2^OUTPUT_LIMB_BITS - 1)^2 must be < product(moduli).
bool crt_coefficient_bound_satisfied(size_t L_A, size_t L_B,
                                     std::string* why = nullptr);

// Padded NTT size valid for moduli/roots and CRT coefficient bound holds.
bool multiply_size_supported(size_t L_A, size_t L_B, std::string* why = nullptr);

// Print to stderr and std::exit(1) if unsupported.
void ensure_ntt_size_supported(size_t N);
void ensure_multiply_size_supported(size_t L_A, size_t L_B);
