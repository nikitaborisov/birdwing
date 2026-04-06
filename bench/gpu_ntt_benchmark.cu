// bench_ntt_32_vs_64.cu
//
// Benchmarks GPU NTT + INTT as a round-trip (forward then inverse on the same
// buffer), comparing 32-bit (Data32) vs 64-bit (Data64) performance.
//
// Timing: average latency over BENCH_ITERS iterations (CUDA events).
//         BENCH_WARMUP iterations are run first and excluded from stats.
//
// Fill in moduli_64 and roots_64 before building.

#include "gpu_ntt.h"
#include "modular_arith.cuh"
#include "ntt.cuh"
#include "config.h"

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>

using namespace std;
using namespace gpuntt;

// ---------------------------------------------------------------------------
// Benchmark knobs
// ---------------------------------------------------------------------------

static constexpr int BENCH_LOG_N_32  = 20;   // polynomial degree = 2^BENCH_LOG_N
static constexpr int BENCH_LOG_N_64 = 19;
static constexpr int BENCH_ITERS  = 100;  // counted iterations
static constexpr int BENCH_WARMUP = 1;   // warm-up iterations (excluded)

// ---------------------------------------------------------------------------
// 32-bit configuration
// ---------------------------------------------------------------------------

using D32 = Data32;
using U32 = uint32_t;

static vector<U32> moduli_32 = {754974721};
static vector<U32> roots_32  = {663};  // primitive 2^23-rd roots

// ---------------------------------------------------------------------------
// 64-bit configuration  —  FILL THESE IN
// ---------------------------------------------------------------------------

using D64 = Data64;
using U64 = uint64_t;

static vector<U64> moduli_64 = {288547035500511233};
static vector<U64> roots_64 = {240676858840140095};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

template <typename T>
static T safe_mod_mul(T a, T b, T mod) {
    if constexpr (sizeof(T) == 4)
        return (T)(((uint64_t)a * b) % mod);
    else
        return (T)(((unsigned __int128)a * b) % mod);
}

// Square root_2_23 down to a 2^logN-th root.
template <typename T>
static T derive_root(T root_2_23, T mod, int logN) {
    T r = root_2_23;
    for (int j = 0; j < (23 - logN); ++j)
        r = safe_mod_mul(r, r, mod);
    return r;
}

// ---------------------------------------------------------------------------
// CUDA event timer
// ---------------------------------------------------------------------------

struct CudaTimer {
    cudaEvent_t start, stop;
    CudaTimer()  { cudaEventCreate(&start); cudaEventCreate(&stop); }
    ~CudaTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }

    void begin(cudaStream_t s = 0) { cudaEventRecord(start, s); }
    float end(cudaStream_t s = 0) {
        cudaEventRecord(stop, s);
        cudaEventSynchronize(stop);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

template <typename DataT>
vector<DataT> cpu_roundtrip_reference(
    const NTTParameters<DataT>& params,
    vector<DataT>& input)
{
    // CPU NTT engine
    NTTCPU<DataT> cpu(params);

    // Forward transform
    vector<DataT> freq = cpu.ntt(input);

    // Inverse transform (if your library provides it)
    vector<DataT> time = cpu.intt(freq);

    return time;
}

// ---------------------------------------------------------------------------
// Benchmark driver (templated on data width)
// ---------------------------------------------------------------------------

template <typename DataT, typename UintT>
struct NTTBench {

    int    logN;
    size_t N;

    vector<DataT*>               d_poly;
    vector<Root<DataT>*>         d_fwd_table;
    vector<Root<DataT>*>         d_inv_table;
    vector<Modulus<DataT>*>      d_mod;
    vector<Ninverse<DataT>*>     d_ninv;
    vector<NTTParameters<DataT>> params;
    vector<vector<DataT>>        h_input;

    // ------------------------------------------------------------------
    void setup(int _logN,
               const vector<UintT>& moduli,
               const vector<UintT>& roots_2_23)
    {
        logN      = _logN;
        N         = 1ULL << logN;
        size_t nm = moduli.size();

        d_poly.resize(nm);
        d_fwd_table.resize(nm);
        d_inv_table.resize(nm);
        d_mod.resize(nm);
        d_ninv.resize(nm);
        params.resize(nm);
        h_input.resize(nm);

        for (size_t i = 0; i < nm; ++i) {
            UintT root_logN = derive_root(roots_2_23[i], moduli[i], logN);
            UintT root_2n   = safe_mod_mul(root_logN, root_logN, moduli[i]);

            NTTFactors<DataT> fac{
                (DataT)moduli[i],
                (DataT)root_logN,
                (DataT)root_2n
            };

            params[i] = NTTParameters<DataT>(
                logN, fac, ReductionPolynomial::X_N_minus);
            auto& p = params[i];

            // Forward table
            auto fwd_host = p.gpu_root_of_unity_table_generator(p.forward_root_of_unity_table);
            cudaMalloc(&d_fwd_table[i], p.root_of_unity_size * sizeof(Root<DataT>));
            cudaMemcpy(d_fwd_table[i], fwd_host.data(),
                       p.root_of_unity_size * sizeof(Root<DataT>),
                       cudaMemcpyHostToDevice);

            // Inverse table
            auto inv_host = p.gpu_root_of_unity_table_generator(p.inverse_root_of_unity_table);
            cudaMalloc(&d_inv_table[i], p.root_of_unity_size * sizeof(Root<DataT>));
            cudaMemcpy(d_inv_table[i], inv_host.data(),
                       p.root_of_unity_size * sizeof(Root<DataT>),
                       cudaMemcpyHostToDevice);

            // Modulus
            cudaMalloc(&d_mod[i], sizeof(Modulus<DataT>));
            Modulus<DataT> mod_h[1] = { p.modulus };
            cudaMemcpy(d_mod[i], mod_h, sizeof(Modulus<DataT>), cudaMemcpyHostToDevice);

            // N-inverse
            cudaMalloc(&d_ninv[i], sizeof(Ninverse<DataT>));
            Ninverse<DataT> ninv_h[1] = { p.n_inv };
            cudaMemcpy(d_ninv[i], ninv_h, sizeof(Ninverse<DataT>), cudaMemcpyHostToDevice);

            // Polynomial buffer (constant 1 — stable, avoids zero-input shortcuts)
            cudaMalloc(&d_poly[i], N * sizeof(DataT));
            vector<DataT> hp(N, DataT(1));
            h_input[i] = hp;
            cudaMemcpy(d_poly[i], hp.data(), N * sizeof(DataT), cudaMemcpyHostToDevice);
        }

        cudaDeviceSynchronize();
    }

    // ------------------------------------------------------------------
    void teardown() {
        for (size_t i = 0; i < d_poly.size(); ++i) {
            cudaFree(d_poly[i]);
            cudaFree(d_fwd_table[i]);
            cudaFree(d_inv_table[i]);
            cudaFree(d_mod[i]);
            cudaFree(d_ninv[i]);
        }
    }

    // ------------------------------------------------------------------
    // Round-trip: NTT then INTT on the same buffer, across all moduli.
    // The host→device copy is intentionally outside the timed window —
    // we are benchmarking kernel throughput only.
    // Returns average milliseconds over `iters` counted iterations.
    // ------------------------------------------------------------------
    float bench_roundtrip(int warmup, int iters) {
        CudaTimer timer;
        float sum_ms = 0.f;

        for (int it = 0; it < warmup + iters; ++it) {
            // Reset each poly to a known state so every iteration is identical.
            for (size_t i = 0; i < d_poly.size(); ++i) {
                vector<DataT> hp(params[i].n, DataT(1));
                cudaMemcpy(d_poly[i], hp.data(),
                           params[i].n * sizeof(DataT),
                           cudaMemcpyHostToDevice);
            }
            cudaDeviceSynchronize();

            // ---- timed window ----
            timer.begin();

            for (size_t i = 0; i < d_poly.size(); ++i) {
                ntt_rns_configuration<DataT> cfg_fwd = {
                    .n_power        = logN,
                    .ntt_type       = FORWARD,
                    .ntt_layout     = PerPolynomial,
                    .reduction_poly = ReductionPolynomial::X_N_minus,
                    .zero_padding   = false,
                    .stream         = 0
                };
                GPU_NTT_Inplace(d_poly[i], d_fwd_table[i],
                                d_mod[i], cfg_fwd, 1, 1);

                ntt_rns_configuration<DataT> cfg_inv = {
                    .n_power        = logN,
                    .ntt_type       = INVERSE,
                    .ntt_layout     = PerPolynomial,
                    .reduction_poly = ReductionPolynomial::X_N_minus,
                    .zero_padding   = false,
                    .mod_inverse    = d_ninv[i],
                    .stream         = 0
                };
                GPU_INTT(d_poly[i], d_poly[i],
                         d_inv_table[i], d_mod[i],
                         cfg_inv, 1, 1);
            }

            float ms = timer.end();
            // ---- end timed window ----

            if (it >= warmup)
                sum_ms += ms;
        }

        return sum_ms / iters;
    }

    bool verify_roundtrip()
    {
        for (size_t i = 0; i < d_poly.size(); ++i)
        {
            vector<DataT> host(N);

            cudaMemcpy(host.data(), d_poly[i],
                    N * sizeof(DataT),
                    cudaMemcpyDeviceToHost);

            vector<DataT> ref = cpu_roundtrip_reference(params[i], h_input[i]);

            if (!check_result(host.data(), ref.data(), N))
            {
                std::cout << "Mismatch at modulus " << i << std::endl;
                return false;
            }
        }
        return true;
    }
};

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

int main() {

    // Guard against unfilled 64-bit entries.
    for (size_t i = 0; i < moduli_64.size(); ++i) {
        if (moduli_64[i] == 0 || roots_64[i] == 0) {
            cerr << "ERROR: moduli_64[" << i << "] or roots_64[" << i << "] "
                 << "is still 0 — fill them in before running.\n";
            return 1;
        }
    }

    cout << "\n"
         << "=============================================================\n"
         << "  GPU NTT Round-Trip Benchmark: 32-bit vs 64-bit\n"
         << "  Measure  : NTT + INTT (same buffer)\n"
         << "  Warm-up  : " << BENCH_WARMUP << " iterations (discarded)\n"
         << "  Counted  : " << BENCH_ITERS  << " iterations\n"
         << "=============================================================\n\n";

    // 32-bit
    nvtxRangePush("32-bit NTT");
    NTTBench<D32, U32> bench32;
    bench32.setup(BENCH_LOG_N_32, moduli_32, roots_32);
    float avg32 = bench32.bench_roundtrip(BENCH_WARMUP, BENCH_ITERS);
    if (!bench32.verify_roundtrip())
        std::cerr << "32-bit correctness FAILED\n";
    else
        std::cout << "32-bit correctness OK\n";
    bench32.teardown();
    nvtxRangePop();

    // 64-bit
    nvtxRangePush("64-bit NTT");
    NTTBench<D64, U64> bench64;
    bench64.setup(BENCH_LOG_N_64, moduli_64, roots_64);
    float avg64 = bench64.bench_roundtrip(BENCH_WARMUP, BENCH_ITERS);
    if (!bench64.verify_roundtrip())
        std::cerr << "64-bit correctness FAILED\n";
    else
        std::cout << "64-bit correctness OK\n";
    bench64.teardown();
    nvtxRangePop();

    // Results
    float speedup = avg64 / avg32;

    cout << fixed << setprecision(4)
         << left
         << setw(10) << "Width"
         << setw(26) << "Avg round-trip (ms)"
         << "\n"
         << string(36, '-') << "\n"
         << setw(10) << "32-bit" << avg32 << "\n"
         << setw(10) << "64-bit" << avg64 << "\n"
         << string(36, '-') << "\n\n"
         << "32-bit is " << fixed << setprecision(2)
         << speedup << "x faster than 64-bit (avg)\n\n";

    return 0;
}
