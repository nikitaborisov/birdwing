#pragma once
#include <cstdint>
#include <vector>
#include <chrono>
#include "config.h"

using limb_t = OutputLimbType;

// Multiply two large integers represented as 32-bit limbs.
void host_multiply_merge(const vector<uint32_t> &A,
                         const vector<uint32_t> &B,
                         vector<OutputLimbType> &C,
                         chrono::duration<double, milli> &duration);