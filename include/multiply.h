#pragma once
#include <cstdint>
#include <vector>
#include <chrono>
#include "config.h"

using limb_t = TestDataTypeUint;

// Multiply two large integers represented as base-2^30 limbs.
void host_multiply_merge(const vector<uint32_t> &A, 
                         const vector<uint32_t> &B,
                         vector<TestDataTypeUint> &C,
                         chrono::duration<double, milli> &duration);