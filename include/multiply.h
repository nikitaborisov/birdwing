#pragma once
#include <cstdint>
#include <vector>
#include <chrono>
#include "config.h"

using limb_t = OutputLimbType;

#if !defined(NATIVE_HOST_LIMBS)
// 32-bit and hybrid pipelines (uint32_t host limbs)
void host_multiply_merge(const vector<uint32_t> &A,
                         const vector<uint32_t> &B,
                         vector<OutputLimbType> &C,
                         chrono::duration<double, milli> &duration);
#endif

#if defined(NATIVE_HOST_LIMBS)
// 64-bit pipeline (uint64_t host limbs)
void host_multiply_merge_native(const vector<uint64_t> &A,
                                const vector<uint64_t> &B,
                                vector<uint64_t> &C,
                                chrono::duration<double, milli> &duration);
#endif
