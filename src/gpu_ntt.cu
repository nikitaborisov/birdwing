#define DEBUG 0

#include "modular_arith.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <ctime>

#include "ntt.cuh"
#include "config.h"

using namespace std;
using namespace gpuntt;

typedef Data32 TestDataType;

vector<TestDataTypeUint> moduli = {754974721, 595591169, 645922817};
vector<TestDataTypeUint> roots_of_unity_2_23 = {663, 721, 19};

static void log_durations_to_csv(
    const string &filename,
    const vector<double> &durations,
    const vector<string> &labels,
    size_t L
) {
    ofstream file;
    bool file_exists = ifstream(filename).good();

    file.open(filename, ios::app);

    // Write CSV header only once
    if (!file_exists) {
        file << "L,step_index,step_name,duration_ms\n";
    }

    // Write rows
    for (size_t i = 0; i < durations.size(); i++) {
        file << L << ","
             << i << ","
             << (i < labels.size() ? labels[i] : "unknown") << ","
             << fixed << setprecision(6) << durations[i]
             << "\n";
    }

    file.close();
}

static TestDataType mod_mul(long long a, long long b, long long mod) {
    return (a * b) % mod;
}

// helper to generate new factors table compatible with given N
static array<NTTFactors<TestDataType>, NUM_MODULI> generate_factors_for_N(int logN) {
    // we want the (2^logN - 1) and 2^logN th roots of unity from 2^23rd roots
    array<NTTFactors<TestDataType>, NUM_MODULI> new_factors;
    for (int i = 0; i < NUM_MODULI; i++) {
        TestDataType root_2_23 = roots_of_unity_2_23[i];
        // now need to square this root (23 - logN) times to get the 2^logN th root
        TestDataType root_2_logN = root_2_23;
        for (int j = 0; j < (23 - logN); j++) {
            root_2_logN = mod_mul(root_2_logN, root_2_logN, moduli[i]);
        }
        new_factors[i] = {moduli[i], root_2_logN, mod_mul(root_2_logN, root_2_logN, moduli[i])};
    }
    return new_factors;
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

__host__ void ntt_multiply(vector<TestDataTypeUint> &a, vector<TestDataTypeUint> &b, vector<vector<TestDataTypeUint>> &c_recovered) {
    #if DEBUG == 1
    cout << "Entering host side ntt_merge_forward function" << endl;
    #endif

    // array of times
    vector<double> durations;
    vector<string> labels;
    vector<vector<TestDataTypeUint>> a_mod;
    vector<vector<TestDataTypeUint>> b_mod;
    vector<TestDataType*> a_InOut_Datas(NUM_MODULI);
    vector<TestDataType*> b_InOut_Datas(NUM_MODULI);
    vector<TestDataType*> c_pointwise_mul(NUM_MODULI);

    auto t0 = chrono::high_resolution_clock::now();

    // need to convert to compatible data type
    vector<TestDataType> a32(a.begin(), a.end());
    vector<TestDataType> b32(b.begin(), b.end());

    size_t N = a.size();
    if (N == 0) return;
    int logN = log2(static_cast<int>(N));

    auto factors_for_N = generate_factors_for_N(logN);
    auto t1 = chrono::high_resolution_clock::now();
    durations.push_back(chrono::duration<double, milli>(t1 - t0).count());
    labels.push_back("preprocessing");

    for (int i = 0; i < NUM_MODULI; i ++ ) {
        // CPU NTT
        #if DEBUG == 1
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
        #endif

        auto t2 = chrono::high_resolution_clock::now();
        NTTParameters parameters(logN, factors_for_N[i], ReductionPolynomial::X_N_minus); // N is the length of the array you are sending in
        auto t3 = chrono::high_resolution_clock::now();
        durations.push_back(chrono::duration<double, milli>(t3 - t2).count());
        labels.push_back("NTTParameters initialization");

        // input copying to the device
        auto t4 = chrono::high_resolution_clock::now();
        GPUNTT_CUDA_CHECK(cudaMalloc(&a_InOut_Datas[i], parameters.n * sizeof(TestDataType)));
        GPUNTT_CUDA_CHECK(cudaMemcpy(a_InOut_Datas[i], a32.data(), parameters.n * sizeof(TestDataType), cudaMemcpyHostToDevice));
        GPUNTT_CUDA_CHECK(cudaMalloc(&b_InOut_Datas[i], parameters.n * sizeof(TestDataType)));
        GPUNTT_CUDA_CHECK(cudaMemcpy(b_InOut_Datas[i], b32.data(), parameters.n * sizeof(TestDataType), cudaMemcpyHostToDevice));
        auto t5 = chrono::high_resolution_clock::now();
        durations.push_back(chrono::duration<double, milli>(t5 - t4).count());
        labels.push_back("input copying to device");

        // Forward omega table allocation + generation + copying to device
        auto t6 = chrono::high_resolution_clock::now();
        Root<TestDataType>* Forward_Omega_Table_Device;
        GPUNTT_CUDA_CHECK(cudaMalloc(&Forward_Omega_Table_Device, parameters.root_of_unity_size * sizeof(Root<TestDataType>)));
        vector<Root<TestDataType>> forward_omega_table = parameters.gpu_root_of_unity_table_generator(parameters.forward_root_of_unity_table);
        auto t7 = chrono::high_resolution_clock::now();
        durations.push_back(chrono::duration<double, milli>(t7 - t6).count());
        labels.push_back("forward omega table alloc and generation");

        #if DEBUG == 1
        cout << "[GPU] Forward omega table values:" << endl;
        for (size_t j = 0; j < forward_omega_table.size(); j++) {
            cout << "  Omega[" << j << "] = " << static_cast<unsigned long long>(forward_omega_table[j]) << endl;
        }
        #endif

        // copying forward omega table to device
        GPUNTT_CUDA_CHECK(cudaDeviceSynchronize());
        auto t8 = chrono::high_resolution_clock::now();
        GPUNTT_CUDA_CHECK(cudaMemcpy(Forward_Omega_Table_Device, forward_omega_table.data(),
                parameters.root_of_unity_size * sizeof(Root<TestDataType>), cudaMemcpyHostToDevice));
        GPUNTT_CUDA_CHECK(cudaDeviceSynchronize());
        auto t9 = chrono::high_resolution_clock::now();
        durations.push_back(chrono::duration<double, milli>(t9 - t8).count());
        labels.push_back("copying forward omega table to device");

        // modulus copying to device
        auto t10 = chrono::high_resolution_clock::now();
        Modulus<TestDataType>* test_modulus;
        GPUNTT_CUDA_CHECK(cudaMalloc(&test_modulus, sizeof(Modulus<TestDataType>)));
        Modulus<TestDataType> test_modulus_[1] = {parameters.modulus};
        GPUNTT_CUDA_CHECK(cudaMemcpy(test_modulus, test_modulus_, sizeof(Modulus<TestDataType>), cudaMemcpyHostToDevice));
        auto t11 = chrono::high_resolution_clock::now();
        durations.push_back(chrono::duration<double, milli>(t11 - t10).count());
        labels.push_back("modulus copying to device");

        GPUNTT_CUDA_CHECK(cudaDeviceSynchronize());
        auto t12 = chrono::high_resolution_clock::now();
        ntt_rns_configuration<TestDataType> cfg_ntt = {
            .n_power = logN,
            .ntt_type = FORWARD,
            .ntt_layout = PerPolynomial,
            .reduction_poly = ReductionPolynomial::X_N_minus,
            .zero_padding = false,
            .stream = 0};
    
        // launching kernel for gpu ntt in place
        GPU_NTT_Inplace(a_InOut_Datas[i], Forward_Omega_Table_Device, test_modulus, cfg_ntt, BATCH, 1);
        GPUNTT_CUDA_CHECK(cudaDeviceSynchronize());
        GPU_NTT_Inplace(b_InOut_Datas[i], Forward_Omega_Table_Device, test_modulus, cfg_ntt, BATCH, 1);
        GPUNTT_CUDA_CHECK(cudaDeviceSynchronize());
        auto t13 = chrono::high_resolution_clock::now();
        durations.push_back(chrono::duration<double, milli>(t13 - t12).count());
        labels.push_back("GPU NTT inplace execution");

        // copying output to host
        #if DEBUG == 1
        auto t14 = chrono::high_resolution_clock::now();
        TestDataType* Output_Host;
        Output_Host = (TestDataType*) malloc(parameters.n * sizeof(TestDataType));
        GPUNTT_CUDA_CHECK(cudaMemcpy(Output_Host, a_InOut_Datas[i],
                    parameters.n * sizeof(TestDataType),
                    cudaMemcpyDeviceToHost));
        auto t15 = chrono::high_resolution_clock::now();
        durations.push_back(chrono::duration<double, milli>(t15 - t14).count());
        labels.push_back("copying output to host");

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
        #endif

        cudaFree(Forward_Omega_Table_Device);
        cudaFree(test_modulus); 
    }

    // pointwise multiplication on the gpu
    #if DEBUG == 1
    cout << "[HOST] Starting GPU pointwise multiplication" << endl;
    #endif

    for (size_t i = 0; i < NUM_MODULI; ++i) {
        #if DEBUG == 1
        cout << "[HOST] Processing modulus " << m << " (mod = " << moduli[i] << ")" << endl;
        #endif

        TestDataTypeUint modulus = moduli[i];

        int threads = 256;
        int blocks = (N + threads - 1) / threads;

        cudaMalloc(&c_pointwise_mul[i], N * sizeof(TestDataType));

        pointwise_mul_kernel<<<blocks, threads>>>(a_InOut_Datas[i], b_InOut_Datas[i], c_pointwise_mul[i], modulus, N);
        GPUNTT_CUDA_CHECK(cudaDeviceSynchronize());
    }

    c_recovered.resize(NUM_MODULI);

    // intt
    #if DEBUG == 1
    cout << "Entering host side gpu_ntt_inverse function" << endl;
    #endif

    for (int i = 0; i < NUM_MODULI; i ++ ) {
        NTTParameters parameters(logN, factors_for_N[i], ReductionPolynomial::X_N_minus);

        // Inverse omega table allocation + generation + copying to device
        Root<TestDataType>* Inverse_Omega_Table_Device;
        GPUNTT_CUDA_CHECK(cudaMalloc(&Inverse_Omega_Table_Device, parameters.root_of_unity_size * sizeof(Root<TestDataType>)));
        vector<Root<TestDataType>> inverse_omega_table = parameters.gpu_root_of_unity_table_generator(parameters.inverse_root_of_unity_table);

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

        TestDataType* Out_Datas;
        GPUNTT_CUDA_CHECK(cudaMalloc(&Out_Datas, parameters.n * sizeof(TestDataType)));
        GPU_INTT(c_pointwise_mul[i], Out_Datas, Inverse_Omega_Table_Device, test_modulus, cfg_intt, BATCH, 1);

        GPUNTT_CUDA_CHECK(cudaDeviceSynchronize());

        vector<TestDataType> host_result(parameters.n);

        GPUNTT_CUDA_CHECK(cudaMemcpy(host_result.data(), Out_Datas, parameters.n * sizeof(TestDataType), cudaMemcpyDeviceToHost));

        c_recovered[i].resize(parameters.n);

        for (size_t j = 0; j < parameters.n; j++)
            c_recovered[i][j] = static_cast<TestDataTypeUint>(host_result[j]);

        cudaFree(Inverse_Omega_Table_Device);
        cudaFree(test_modulus);
        cudaFree(test_ninverse);
        cudaFree(Out_Datas);
    }

    for (int i = 0; i < NUM_MODULI; i++) {
        cudaFree(a_InOut_Datas[i]);
        cudaFree(b_InOut_Datas[i]);
        cudaFree(c_pointwise_mul[i]);
    }
}