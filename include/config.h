#pragma once
#ifndef CONFIG_H
#define CONFIG_H

#define NUM_MODULI 2
#define BATCH 1

typedef uint64_t TestDataTypeUint;
typedef __uint128_t TestDataTypeUint128;

extern std::vector<TestDataTypeUint> moduli;

#endif // CONFIG_H