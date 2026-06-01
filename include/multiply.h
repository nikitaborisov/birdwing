#pragma once
#include <cstdint>
#include <vector>
#include <chrono>
#include "config.h"

using limb_t = TestDataTypeUint;

// Multiply two large integers represented as base-2^30 limbs.
void host_multiply_merge(const vector<uint32_t> &A, // changed from limb_t to uint32_t
                         const vector<uint32_t> &B, // changed from limb_t to uint32_t
                         vector<TestDataTypeUint> &C, // changed from limb_t to TestDataTypeUint
                         chrono::duration<double, milli> &duration);