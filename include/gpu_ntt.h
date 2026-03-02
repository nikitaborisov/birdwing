#pragma once
#include <cstdint>
#include <vector>
#include <iostream>
#include "config.h"

using namespace std;

void ntt_multiply(vector<TestDataTypeUint> &a, vector<TestDataTypeUint> &b, vector<vector<TestDataTypeUint>> &c_recovered);
void ntt_merge_forward(vector<TestDataTypeUint> &a, vector<vector<TestDataTypeUint>> &a_mod);
void gpu_pointwise_multiply(const vector<vector<TestDataTypeUint>>& A_mod, const vector<vector<TestDataTypeUint>>& B_mod, vector<vector<TestDataTypeUint>>& C_mod);
void gpu_ntt_inverse(vector<vector<TestDataTypeUint>> &c_mod, vector<vector<TestDataTypeUint>> &c_recovered);
