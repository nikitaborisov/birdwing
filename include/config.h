#pragma once
#ifndef CONFIG_H
#define CONFIG_H

#include <vector>
#include <cstdint>
#include <type_traits>

#ifndef LIMB_BITS
#define LIMB_BITS 32   // default
#endif

#ifndef DEBUG
#define DEBUG 0
#endif

// Multiply pipelines (separate compile-time binaries):
//   -DLIMB_BITS=32                          → 32-bit: uint32 in/out, Data32, 3 small primes
//   -DLIMB_BITS=64                          → 64/2-mod: uint32 in/out (widened), Data64, 2×59-bit
//   -DLIMB_BITS=64 -DNATIVE_HOST_LIMBS      → 64-native: uint64 in/out, Data64, 3×59-bit, U160 CRT
#if LIMB_BITS == 64 && defined(NATIVE_HOST_LIMBS)
	#define NUM_MODULI 3
	#define INPUT_LIMB_BITS 64
	#define OUTPUT_LIMB_BITS 64
	#define CRT_COEFF_BITS 160
#elif LIMB_BITS == 64
	#define NUM_MODULI 2
	#define INPUT_LIMB_BITS 32
	#define OUTPUT_LIMB_BITS 32
	#define CRT_COEFF_BITS 128
#else
	#define NUM_MODULI 3
	#define INPUT_LIMB_BITS 32
	#define OUTPUT_LIMB_BITS 32
	#define CRT_COEFF_BITS 128
#endif

#if LIMB_BITS == 64
	using LimbType = uint64_t;
	using TestDataTypeUint = uint64_t;
	using TestDataTypeInt = int64_t;
	#define LIMB_MASK 0xFFFFFFFFFFFFFFFFULL
#else
	using LimbType = uint32_t;
	using TestDataTypeUint = uint32_t;
	using TestDataTypeInt = int32_t;
	#define LIMB_MASK 0xFFFFFFFFULL
#endif

constexpr int BIT_WIDTH = sizeof(TestDataTypeUint) * 8;

using InputLimbType = std::conditional_t<INPUT_LIMB_BITS == 64, uint64_t, uint32_t>;
using OutputLimbType = std::conditional_t<OUTPUT_LIMB_BITS == 64, uint64_t, uint32_t>;

#if OUTPUT_LIMB_BITS == 64
	#define OUTPUT_LIMB_MASK 0xFFFFFFFFFFFFFFFFULL
#else
	#define OUTPUT_LIMB_MASK 0xFFFFFFFFULL
#endif

using namespace std;

#define BATCH 1
#define CARRY_SEG 1024

extern vector<TestDataTypeUint> moduli;
extern vector<TestDataTypeUint> roots_of_unity_max;

#endif // CONFIG_H
