#pragma once
#include <cstdint>
#include <vector>
#include <chrono>
#include "config.h"

using limb_t = TestDataTypeUint;

// Multiply two large integers represented as base-2^30 limbs.
void host_multiply_merge(const std::vector<limb_t> &A, const std::vector<limb_t> &B,
                   std::vector<limb_t> &C, std::chrono::duration<double, std::milli> &duration);