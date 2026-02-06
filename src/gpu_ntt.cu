#include "modular_arith.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#include "ntt.cuh"
#include "config.h"

using namespace std;
using namespace gpuntt;

typedef Data32 TestDataType;

NTTFactors<TestDataType> factors[4] = {
    {Modulus<TestDataType>(754974721), 205430076, 49823357},
    {Modulus<TestDataType>(595591169), 68081344, 443510249},
    {Modulus<TestDataType>(645922817), 146614077, 280465537}
    // {Modulus<TestDataType>(10753), 4305, 4894}
};

vector<TestDataTypeUint> moduli = {754974721, 595591169, 645922817};

__host__ void ntt_merge_forward(vector<TestDataTypeUint> &a, vector<vector<TestDataTypeUint>> &a_mod) {
    cout << "Entering host side ntt_merge_forward function" << endl;

    // need to convert to compatible data type
    vector<TestDataType> a32(a.begin(), a.end());

    size_t N = a.size();
    if (N == 0) return;
    int logN = log2(static_cast<int>(N));

    for (int i = 0; i < NUM_MODULI; i ++ ) {
        NTTParameters parameters(logN, factors[i], ReductionPolynomial::X_N_minus); // N is the length of the array you are sending in

        // CPU NTT
        NTTCPU<TestDataType> generator(parameters);
        vector<TestDataType> cpu_ntt_result = generator.ntt(a32);
        cout << "[CPU] Forward NTT result: [ ";
        for (const auto& x : cpu_ntt_result)
            cout << x << " ";
        cout << "]" << endl;

        vector<TestDataType> cpu_intt_result = generator.intt(cpu_ntt_result);
        cout << "[CPU] Inverse NTT result: [ ";
        for (const auto& x : cpu_intt_result)
            cout << x << " ";
        cout << "]" << endl;

        // input copying to the device
        TestDataType* InOut_Datas;
        GPUNTT_CUDA_CHECK(cudaMalloc(&InOut_Datas, parameters.n * sizeof(TestDataType)));
        GPUNTT_CUDA_CHECK(cudaMemcpy(InOut_Datas, a32.data(), parameters.n * sizeof(TestDataType), cudaMemcpyHostToDevice));

        // Forward omega table allocation + generation + copying to device
        Root<TestDataType>* Forward_Omega_Table_Device;
        GPUNTT_CUDA_CHECK(cudaMalloc(&Forward_Omega_Table_Device, parameters.root_of_unity_size * sizeof(Root<TestDataType>)));
        vector<Root<TestDataType>> forward_omega_table = parameters.gpu_root_of_unity_table_generator(parameters.forward_root_of_unity_table);

        cout << "[GPU] Forward omega table values:" << endl;
        for (size_t j = 0; j < forward_omega_table.size(); j++) {
            cout << "  Omega[" << j << "] = " << static_cast<unsigned long long>(forward_omega_table[j]) << endl;
        }

        GPUNTT_CUDA_CHECK(cudaMemcpy(Forward_Omega_Table_Device, forward_omega_table.data(),
                parameters.root_of_unity_size * sizeof(Root<TestDataType>), cudaMemcpyHostToDevice));

        // GPU NTT inplace call
        Modulus<TestDataType>* test_modulus;
        GPUNTT_CUDA_CHECK(cudaMalloc(&test_modulus, sizeof(Modulus<TestDataType>)));
        Modulus<TestDataType> test_modulus_[1] = {parameters.modulus};
        GPUNTT_CUDA_CHECK(cudaMemcpy(test_modulus, test_modulus_, sizeof(Modulus<TestDataType>), cudaMemcpyHostToDevice));

        ntt_rns_configuration<TestDataType> cfg_ntt = {
            .n_power = logN,
            .ntt_type = FORWARD,
            .ntt_layout = PerPolynomial,
            .reduction_poly = ReductionPolynomial::X_N_minus,
            .zero_padding = false,
            .stream = 0};
    
        // launching kernel for gpu ntt in place
        GPU_NTT_Inplace(InOut_Datas, Forward_Omega_Table_Device, test_modulus, cfg_ntt, BATCH, 1);

        // copying output to host
        TestDataType* Output_Host;
        Output_Host = (TestDataType*) malloc(parameters.n * sizeof(TestDataType));
        GPUNTT_CUDA_CHECK(cudaMemcpy(Output_Host, InOut_Datas,
                    parameters.n * sizeof(TestDataType),
                    cudaMemcpyDeviceToHost));

        cout << "[GPU] NTT output (device -> host): [ ";
        for (long unsigned int j = 0; j < parameters.n; j++) {
            cout << static_cast<unsigned long long>(Output_Host[j]) << " ";
        }
        cout << "]" << endl;

        // Comparing GPU NTT results and CPU NTT results
        bool check = true;
        for (int j = 0; j < BATCH; j++)
        {
            check = check_result(Output_Host + (j * parameters.n),
                                    cpu_ntt_result.data(), parameters.n);

            if (!check)
            {
                cout << "(in " << j << ". Poly.)" << endl;
                break;
            }

            if ((j == (BATCH - 1)) && check)
            {
                cout << "All Correct for PerPolynomial NTT." << endl;
            }
        }

        // copy Output_Host to a_mod[i]
        if (a_mod.empty()) a_mod.push_back(vector<TestDataTypeUint>());
        
        a_mod[i].clear();
        a_mod[i].reserve(parameters.n * 2);  // each data32 has two TestDataTypeUints? i'm not sure how the datatypes will work
        
        for (size_t j = 0; j < parameters.n; j++) {
            TestDataTypeUint low  = static_cast<TestDataTypeUint>(Output_Host[j] & 0xFFFFFFFF);
            a_mod[i].push_back(low);
        }

        cout << "[HOST] a_mod[" << i << "] = [ ";
        for (size_t k = 0; k < a_mod[i].size(); k++)
            cout << a_mod[i][k] << " ";
        cout << "]" << endl;

        GPUNTT_CUDA_CHECK(cudaFree(InOut_Datas));
        GPUNTT_CUDA_CHECK(cudaFree(Forward_Omega_Table_Device));
        free(Output_Host);
    }
}

template <typename T>
__device__ __forceinline__ void mul_wide(T a, T b, T &lo, T &hi)
{
    if constexpr (sizeof(T) == 4)
    {
        // 32-bit multiply -> 64-bit product
        uint64_t prod = (uint64_t)a * (uint64_t)b;
        lo = (T)prod;            // low 32 bits
        hi = (T)(prod >> 32);    // high 32 bits
    }
    else if constexpr (sizeof(T) == 8)
    {
        // 64-bit multiply -> 128-bit product
        lo = a * b;             // low 64 bits
        hi = __umul64hi(a, b);  // high 64 bits
    }
    else {
        static_assert(sizeof(T) == 4 || sizeof(T) == 8,
                      "Unsupported TestDataTypeUint size.");
    }
}

__global__ void pointwise_mul_kernel(TestDataTypeUint* A,
                                     TestDataTypeUint* B,
                                     TestDataTypeUint* C,
                                     TestDataTypeUint modulus,
                                     size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        TestDataTypeUint lo, hi;
        mul_wide(A[idx], B[idx], lo, hi);

        if constexpr (sizeof(TestDataTypeUint) == 4) {
            uint64_t full = ((uint64_t)hi << 32) | lo;
            C[idx] = (TestDataTypeUint)(full % modulus);
        } else {
            unsigned __int128 full =
                ((unsigned __int128)hi << 64) | lo;
            C[idx] = (TestDataTypeUint)(full % modulus);
        }
    }
}

__host__ void gpu_pointwise_multiply(const vector<vector<TestDataTypeUint>>& A_mod, const vector<vector<TestDataTypeUint>>& B_mod, vector<vector<TestDataTypeUint>>& C_mod) {
    size_t N = A_mod[0].size();

    cout << "[HOST] Starting GPU pointwise multiplication" << endl;

    for (size_t m = 0; m < NUM_MODULI; ++m) {
        cout << "[HOST] A_mod for modulus " << m << " (mod = " << moduli[m] << "): [ ";
        for (size_t i = 0; i < N; ++i)
            cout << A_mod[m][i] << " ";
        cout << "]" << endl;

        cout << "[HOST] B_mod for modulus " << m << " (mod = " << moduli[m] << "): [ ";
        for (size_t i = 0; i < N; ++i)
            cout << B_mod[m][i] << " ";
        cout << "]" << endl;
    }

    C_mod.resize(NUM_MODULI, vector<TestDataTypeUint>(N));

    for (size_t m = 0; m < NUM_MODULI; ++m) {
        cout << "[HOST] Processing modulus " << m << " (mod = " << moduli[m] << ")" << endl;
        // construct device arrays
        const TestDataTypeUint* A_host = A_mod[m].data();
        const TestDataTypeUint* B_host = B_mod[m].data();

        TestDataTypeUint* C_host = C_mod[m].data();
        TestDataTypeUint modulus = moduli[m];

        TestDataTypeUint *A_dev, *B_dev, *C_dev;
        GPUNTT_CUDA_CHECK(cudaMalloc(&A_dev, N * sizeof(TestDataTypeUint)));
        GPUNTT_CUDA_CHECK(cudaMalloc(&B_dev, N * sizeof(TestDataTypeUint)));
        GPUNTT_CUDA_CHECK(cudaMalloc(&C_dev, N * sizeof(TestDataTypeUint)));

        GPUNTT_CUDA_CHECK(cudaMemcpy(A_dev, A_host, N * sizeof(TestDataTypeUint), cudaMemcpyHostToDevice));
        GPUNTT_CUDA_CHECK(cudaMemcpy(B_dev, B_host, N * sizeof(TestDataTypeUint), cudaMemcpyHostToDevice));

        int threads = 256;
        int blocks = (N + threads - 1) / threads;

        pointwise_mul_kernel<<<blocks, threads>>>(A_dev, B_dev, C_dev, modulus, N);
        GPUNTT_CUDA_CHECK(cudaDeviceSynchronize());

        GPUNTT_CUDA_CHECK(cudaMemcpy(C_host, C_dev, N * sizeof(TestDataTypeUint), cudaMemcpyDeviceToHost));

        cout << "[HOST] Result for modulus " << m << ": [ ";
        for (size_t i = 0; i < N; ++i)
            cout << C_host[i] << " ";
        cout << "]" << endl;

        GPUNTT_CUDA_CHECK(cudaFree(A_dev));
        GPUNTT_CUDA_CHECK(cudaFree(B_dev));
        GPUNTT_CUDA_CHECK(cudaFree(C_dev));
    }
}

__host__ void gpu_ntt_inverse(vector<vector<TestDataTypeUint>> &c_mod, vector<vector<TestDataTypeUint>> &c_recovered) {
    cout << "Entering host side gpu_ntt_inverse function" << endl;

    for (int i = 0; i < NUM_MODULI; i ++ ) {
        // get the size of c_mod for logN in parameters
        size_t N = c_mod[i].size();
        if (N == 0) continue;
        int logN = log2(static_cast<int>(N));
        
        NTTParameters parameters(logN, factors[i], ReductionPolynomial::X_N_minus);

        vector<TestDataType> c32(c_mod[i].begin(), c_mod[i].end());
        
        // CPU INTT
        NTTCPU<TestDataType> generator(parameters);
        vector<TestDataType> cpu_intt_result = generator.intt(c32);
        cout << "[CPU] Inverse NTT result: [ ";
        for (const auto& x : cpu_intt_result)
            cout << x << " ";
        cout << "]" << endl;

        // input copying to the device
        TestDataType* InOut_Datas;
        GPUNTT_CUDA_CHECK(cudaMalloc(&InOut_Datas, parameters.n * sizeof(TestDataType)));
        GPUNTT_CUDA_CHECK(cudaMemcpy(InOut_Datas, c32.data(), parameters.n * sizeof(TestDataType), cudaMemcpyHostToDevice));

        // Inverse omega table allocation + generation + copying to device
        Root<TestDataType>* Inverse_Omega_Table_Device;
        GPUNTT_CUDA_CHECK(cudaMalloc(&Inverse_Omega_Table_Device, parameters.root_of_unity_size * sizeof(Root<TestDataType>)));
        vector<Root<TestDataType>> inverse_omega_table = parameters.gpu_root_of_unity_table_generator(parameters.inverse_root_of_unity_table);

        cout << "[GPU] Inverse omega table values:" << endl;
        for (size_t j = 0; j < inverse_omega_table.size(); j++) {
            cout << "  Omega[" << j << "] = " << static_cast<unsigned long long>(inverse_omega_table[j]) << endl;
        }

        GPUNTT_CUDA_CHECK(cudaMemcpy(Inverse_Omega_Table_Device, inverse_omega_table.data(),
            parameters.root_of_unity_size * sizeof(Root<TestDataType>), cudaMemcpyHostToDevice));

        // setting up modulus / mod_n_inverse to pass into ntt config
        Modulus<TestDataType>* test_modulus;
        GPUNTT_CUDA_CHECK(cudaMalloc(&test_modulus, sizeof(Modulus<TestDataType>)));
        Modulus<TestDataType> test_modulus_[1] = {parameters.modulus};
        GPUNTT_CUDA_CHECK(cudaMemcpy(test_modulus, test_modulus_, sizeof(Modulus<TestDataType>), cudaMemcpyHostToDevice));

        // not sure what this does?
        Ninverse<TestDataType>* test_ninverse;
        GPUNTT_CUDA_CHECK(cudaMalloc(&test_ninverse, sizeof(Ninverse<TestDataType>)));
        Ninverse<TestDataType> test_ninverse_[1] = {parameters.n_inv};
        GPUNTT_CUDA_CHECK(cudaMemcpy(test_ninverse, test_ninverse_, sizeof(Ninverse<TestDataType>), cudaMemcpyHostToDevice));

        ntt_rns_configuration<TestDataType> cfg_intt = {
            .n_power = logN,
            .ntt_type = INVERSE,
            .ntt_layout = PerPolynomial,
            .reduction_poly = ReductionPolynomial::X_N_minus,
            .zero_padding = false,
            .mod_inverse = test_ninverse,
            .stream = 0};

        // output alloc + GPU INTT call
        TestDataType* Out_Datas;
        GPUNTT_CUDA_CHECK(cudaMalloc(&Out_Datas, parameters.n * sizeof(TestDataType)));
        GPUNTT_CUDA_CHECK(cudaMemcpy(Out_Datas, c32.data(), parameters.n * sizeof(TestDataType), cudaMemcpyHostToDevice));
        GPU_INTT(InOut_Datas, Out_Datas, Inverse_Omega_Table_Device, test_modulus, cfg_intt, BATCH, 1);

        // copying output to host
        TestDataType* Output_Host;

        Output_Host = (TestDataType*) malloc(parameters.n * sizeof(TestDataType));
        GPUNTT_CUDA_CHECK(cudaMemcpy(Output_Host, Out_Datas, parameters.n * sizeof(TestDataType), cudaMemcpyDeviceToHost));

        cout << "[GPU] INTT output (device -> host): [ ";
        for (long unsigned int j = 0; j < parameters.n; j++) {
            cout << static_cast<unsigned long long>(Output_Host[j]) << " ";
        }
        cout << "]" << endl;

        // Comparing GPU NTT results and CPU NTT results
        bool check = true;
        for (int j = 0; j < BATCH; j++)
        {
            check = check_result(Output_Host + (j * parameters.n),
                                    cpu_intt_result.data(), parameters.n);

            if (!check)
            {
                cout << "(in " << j << ". Poly.)" << endl;
                break;
            }

            if ((j == (BATCH - 1)) && check)
            {
                cout << "All Correct for PerPolynomial INTT." << endl;
            }
        }

        // copy Output_Host to c_recovered[i]
        if (c_recovered.empty()) c_recovered.push_back(vector<TestDataTypeUint>());
        
        c_recovered.resize(NUM_MODULI);
        c_recovered[i].clear();
        c_recovered[i].reserve(parameters.n * 2);  // each data32 has two TestDataTypeUints? i'm not sure how the datatypes will work
        
        for (size_t j = 0; j < parameters.n; j++) {
            TestDataTypeUint low  = static_cast<TestDataTypeUint>(Output_Host[j] & 0xFFFFFFFF);
            c_recovered[i].push_back(low);
        }

        cout << "[HOST] c_recovered[" << i << "] = [ ";
        for (size_t k = 0; k < c_recovered[i].size(); k++)
            cout << c_recovered[i][k] << " ";
        cout << "]" << endl;

        GPUNTT_CUDA_CHECK(cudaFree(InOut_Datas));
        GPUNTT_CUDA_CHECK(cudaFree(Inverse_Omega_Table_Device));
        free(Output_Host);
    }
}