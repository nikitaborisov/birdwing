#pragma once
#ifndef CONFIG_H
#define CONFIG_H

#include <vector>
#include <cstdint>

#ifndef LIMB_BITS
#define LIMB_BITS 32   // default
#endif

#if LIMB_BITS == 64
	using LimbType = uint64_t;
	using TestDataTypeUint = uint64_t;
	#define LIMB_MASK 0xFFFFFFFFFFFFFFFFULL
	// Number of moduli used in computations
	#define NUM_MODULI 2
#else
	using LimbType = uint32_t;
	using TestDataTypeUint = uint32_t;
	#define LIMB_MASK 0xFFFFFFFFULL
	#define NUM_MODULI 3
#endif

constexpr int BIT_WIDTH = sizeof(TestDataTypeUint) * 8;

using namespace std;

// Batch size for processing
#define BATCH 1

// Carry segment size
#define CARRY_SEG 1024

// Collection of moduli used across the application
extern vector<TestDataTypeUint> moduli;

#endif // CONFIG_H