#pragma once
#ifndef CONFIG_H
#define CONFIG_H

#define NUM_MODULI 3
#define BATCH 1

#define CARRY_SEG 1024

#include <vector>
#include <cstdint>

typedef uint32_t TestDataTypeUint;
typedef uint64_t TestDataTypeTwice;
typedef __uint128_t TestDataTypeUint128;

extern std::vector<TestDataTypeUint> moduli;

#endif // CONFIG_H