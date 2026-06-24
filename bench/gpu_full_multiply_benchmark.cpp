// bench/gpu_full_multiply_benchmark.cpp
//
// Benchmark the full GPU multiply pipeline with setup/teardown and per-stage
// CUDA event timings from execute_ntt_multiply().
//
// Build:  make bench_full_32 | bench_full_64 | bench_full
// Run:    ./build/bench_full_multiply_32 [--warmup N] [--iters N] [--csv FILE] [L ...]
//         ./build/bench_full_multiply_64 ...
//
// Or use the runner:  python scripts/run_gpu_bench.py --limb-bits 64 16-24
//
// L spec: values < 64 mean 1<<L limbs; values >= 64 are literal limb counts.
//         ranges are inclusive, e.g. 16-24 -> 16,17,...,24.

#include "gpu_ntt.h"
#include "config.h"
#include "ntt_limits.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <cstring>
#include <vector>

using namespace std;

static bool file_nonempty(const string& path)
{
    ifstream f(path);
    return f.good() && f.peek() != ifstream::traits_type::eof();
}

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

static constexpr int DEFAULT_WARMUP = 2;
static constexpr int DEFAULT_ITERS  = 20;

static const vector<size_t> DEFAULT_L_VALUES = {
    6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static vector<uint32_t> random_limbs(size_t n, uint64_t seed)
{
    mt19937_64 rng(seed);
    vector<uint32_t> v(n);

    for (size_t i = 0; i < n; i++) {
        if (i % 8 == 0)
            v[i] = 0;
        else if (i % 8 == 1)
            v[i] = 1;
        else
            v[i] = (uint32_t)(rng() % (1ULL << 30));
    }
    return v;
}

static size_t resolve_limb_count(size_t L_arg)
{
    if (L_arg < 64)
        return size_t(1) << L_arg;
    return L_arg;
}

static double host_elapsed_ms(
    const chrono::high_resolution_clock::time_point& t0,
    const chrono::high_resolution_clock::time_point& t1)
{
    return chrono::duration<double, milli>(t1 - t0).count();
}

template <typename Fn>
static double time_host_ms(Fn&& fn)
{
    const auto t0 = chrono::high_resolution_clock::now();
    fn();
    const auto t1 = chrono::high_resolution_clock::now();
    return host_elapsed_ms(t0, t1);
}

// First cudaMallocHost in a process pays driver init; warm up before timing setup.
static void warmup_cuda_runtime()
{
    uint32_t* pinned = nullptr;
    void* dev = nullptr;
    cudaMallocHost(&pinned, 4096);
    cudaMalloc(&dev, 4096);
    cudaFree(dev);
    cudaFreeHost(pinned);
}

struct TimingStats {
    double mean_ms = 0.0;
    double stddev_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
};

static TimingStats compute_stats(const vector<double>& samples)
{
    TimingStats s{};
    if (samples.empty())
        return s;

    double sum = 0.0;
    double sum_sq = 0.0;
    s.min_ms = samples[0];
    s.max_ms = samples[0];

    for (double t : samples) {
        sum += t;
        sum_sq += t * t;
        s.min_ms = min(s.min_ms, t);
        s.max_ms = max(s.max_ms, t);
    }

    const double n = static_cast<double>(samples.size());
    s.mean_ms = sum / n;
    const double var = (sum_sq / n) - (s.mean_ms * s.mean_ms);
    s.stddev_ms = sqrt(max(0.0, var));
    return s;
}

static TimingStats compute_stats_field(
    const vector<NTTTiming>& samples,
    float NTTTiming::*field)
{
    vector<double> values;
    values.reserve(samples.size());
    for (const NTTTiming& t : samples)
        values.push_back(static_cast<double>(t.*field));
    return compute_stats(values);
}

struct SetupTiming {
    double pinned_ms = 0.0;
    double precompute_ms = 0.0;
    double upload_ms = 0.0;
    double alloc_ctx_ms = 0.0;
    PrecomputeTiming precompute{};
    SetupUploadTiming upload{};
};

struct TeardownTiming {
    double free_ctx_ms = 0.0;
    double free_pre_ms = 0.0;
    double free_pinned_ms = 0.0;
};

struct BenchRow {
    size_t L_arg = 0;
    size_t L = 0;
    size_t N = 0;
    int logN = 0;
    int warmup = 0;
    int iters = 0;

    SetupTiming setup{};
    TeardownTiming teardown{};

    TimingStats execute_total{};
    TimingStats ingress_fwd{};
    TimingStats h2d{};
    TimingStats fwd_pad_ntt{};
    TimingStats fwd_pad_ntt_a{};
    TimingStats fwd_pad_ntt_b{};
    TimingStats pointwise_mul{};
    TimingStats intt{};
    TimingStats crt{};
    TimingStats carry{};
    TimingStats d2h{};
};

static void usage(const char* prog)
{
    cerr << "Usage: " << prog
         << " [--warmup N] [--iters N] [--csv FILE] [--append] [L ...]\n"
         << "\n"
         << "  --warmup N   warmup iterations per L (default "
         << DEFAULT_WARMUP << ")\n"
         << "  --iters N    timed iterations per L (default "
         << DEFAULT_ITERS << ")\n"
         << "  --csv FILE   output CSV path (default gpu_multiply_bench.csv)\n"
         << "  --append     append rows to CSV instead of overwriting\n"
         << "  L ...        limb spec: if L < 64, use 1<<L limbs; else L limbs\n"
         << "               ranges inclusive: 16-24 -> 16,17,...,24\n"
         << "               (default sweep: log2 sizes 6..22)\n";
}

static bool parse_size_t(const string& s, size_t& out)
{
    char* end = nullptr;
    unsigned long long v = strtoull(s.c_str(), &end, 10);
    if (end == s.c_str() || *end != '\0' || v == 0)
        return false;
    out = static_cast<size_t>(v);
    return true;
}

static bool append_l_spec(const string& spec, vector<size_t>& out)
{
    const auto dash = spec.find('-');
    if (dash == string::npos) {
        size_t L_arg = 0;
        if (!parse_size_t(spec, L_arg))
            return false;
        out.push_back(L_arg);
        return true;
    }

    if (dash == 0 || dash == spec.size() - 1)
        return false;

    size_t lo = 0, hi = 0;
    if (!parse_size_t(spec.substr(0, dash), lo) ||
        !parse_size_t(spec.substr(dash + 1), hi))
        return false;

    if (lo > hi)
        return false;

    for (size_t v = lo; v <= hi; ++v)
        out.push_back(v);
    return true;
}

static BenchRow benchmark_L(size_t L_arg, int warmup, int iters, uint64_t seed)
{
    const size_t L = resolve_limb_count(L_arg);
    const size_t L_A = L;
    const size_t L_B = L;
    const size_t N = padded_ntt_size(L_A, L_B);
    const int logN = static_cast<int>(lround(log2(static_cast<double>(N))));

    vector<uint32_t> A = random_limbs(L_A, seed);
    vector<uint32_t> B = random_limbs(L_B, seed + 1);

    uint32_t* a_pinned = nullptr;
    uint32_t* b_pinned = nullptr;

    SetupTiming setup{};
    setup.pinned_ms = time_host_ms([&] {
        cudaMallocHost(&a_pinned, L_A * sizeof(uint32_t));
        cudaMallocHost(&b_pinned, L_B * sizeof(uint32_t));
        memcpy(a_pinned, A.data(), L_A * sizeof(uint32_t));
        memcpy(b_pinned, B.data(), L_B * sizeof(uint32_t));
    });

    NTTPrecomputed pre{};
    PrecomputeTiming pre_timing{};
    pre = precompute_ntt(N, &pre_timing);
    setup.precompute = pre_timing;
    setup.precompute_ms = pre_timing.total_ms;

    SetupUploadTiming upload_timing{};
    upload_ntt_precomputed(pre, &upload_timing);
    setup.upload = upload_timing;
    setup.upload_ms = upload_timing.total_ms;

    NTTContext ctx{};
    setup.alloc_ctx_ms = time_host_ms([&] {
        ctx = allocate_ntt_context(pre, L_A, L_B);
    });

    vector<OutputLimbType> C_out(N + 1, 0);

    for (int i = 0; i < warmup; i++)
        execute_ntt_multiply(ctx, a_pinned, b_pinned, C_out);

    vector<double> execute_samples;
    vector<NTTTiming> stage_samples;
    execute_samples.reserve(static_cast<size_t>(iters));
    stage_samples.reserve(static_cast<size_t>(iters));

    for (int i = 0; i < iters; i++) {
        NTTTiming timing{};
        execute_ntt_multiply(
            ctx, a_pinned, b_pinned, C_out, &timing);
        execute_samples.push_back(static_cast<double>(timing.total_ms));
        stage_samples.push_back(timing);
    }

    TeardownTiming teardown{};
    teardown.free_ctx_ms = time_host_ms([&] {
        cleanup_ntt_context(ctx);
    });
    teardown.free_pre_ms = time_host_ms([&] {
        cleanup_ntt_precomputed(pre);
    });
    teardown.free_pinned_ms = time_host_ms([&] {
        cudaFreeHost(a_pinned);
        cudaFreeHost(b_pinned);
    });

    BenchRow row{};
    row.L_arg = L_arg;
    row.L = L;
    row.N = N;
    row.logN = logN;
    row.warmup = warmup;
    row.iters = iters;
    row.setup = setup;
    row.teardown = teardown;
    row.execute_total = compute_stats(execute_samples);
    row.ingress_fwd = compute_stats_field(stage_samples, &NTTTiming::ingress_fwd_ms);
    row.h2d = compute_stats_field(stage_samples, &NTTTiming::h2d_ms);
    row.fwd_pad_ntt = compute_stats_field(stage_samples, &NTTTiming::fwd_pad_ntt_ms);
    row.fwd_pad_ntt_a = compute_stats_field(stage_samples, &NTTTiming::fwd_pad_ntt_a_ms);
    row.fwd_pad_ntt_b = compute_stats_field(stage_samples, &NTTTiming::fwd_pad_ntt_b_ms);
    row.pointwise_mul = compute_stats_field(stage_samples, &NTTTiming::pointwise_mul_ms);
    row.intt = compute_stats_field(stage_samples, &NTTTiming::intt_ms);
    row.crt = compute_stats_field(stage_samples, &NTTTiming::crt_ms);
    row.carry = compute_stats_field(stage_samples, &NTTTiming::carry_ms);
    row.d2h = compute_stats_field(stage_samples, &NTTTiming::d2h_ms);
    return row;
}

static void write_csv(const string& path, const vector<BenchRow>& rows, bool append)
{
    const bool write_header = !append || !file_nonempty(path);
    ofstream csv(path, append ? ios::app : ios::trunc);
    if (!csv) {
        cerr << "Failed to open CSV: " << path << "\n";
        exit(1);
    }

    if (write_header) {
        csv << "limb_bits,L_arg,L,N,logN,warmup,iters,"
            << "mean_ms,stddev_ms,min_ms,max_ms,"
            << "setup_pinned_ms,setup_precompute_ms,setup_upload_ms,setup_alloc_ms,"
            << "pre_factors_ms,pre_params_ms,pre_twiddle_host_ms,pre_garner_host_ms,"
            << "upload_twiddle_ms,upload_mod_constants_ms,upload_garner_ms,"
            << "teardown_free_ctx_ms,teardown_free_pre_ms,teardown_free_pinned_ms,"
            << "ingress_fwd_mean_ms,h2d_mean_ms,fwd_pad_ntt_mean_ms,fwd_pad_ntt_a_mean_ms,fwd_pad_ntt_b_mean_ms,"
            << "mul_mean_ms,intt_mean_ms,crt_mean_ms,carry_mean_ms,d2h_mean_ms\n";
    }
    csv << fixed << setprecision(6);

    for (const BenchRow& row : rows) {
        csv << LIMB_BITS << ","
            << row.L_arg << ","
            << row.L << ","
            << row.N << ","
            << row.logN << ","
            << row.warmup << ","
            << row.iters << ","
            << row.execute_total.mean_ms << ","
            << row.execute_total.stddev_ms << ","
            << row.execute_total.min_ms << ","
            << row.execute_total.max_ms << ","
            << row.setup.pinned_ms << ","
            << row.setup.precompute_ms << ","
            << row.setup.upload_ms << ","
            << row.setup.alloc_ctx_ms << ","
            << row.setup.precompute.factors_ms << ","
            << row.setup.precompute.params_ms << ","
            << row.setup.precompute.twiddle_host_ms << ","
            << row.setup.precompute.garner_host_ms << ","
            << row.setup.upload.twiddle_upload_ms << ","
            << row.setup.upload.mod_constants_ms << ","
            << row.setup.upload.garner_upload_ms << ","
            << row.teardown.free_ctx_ms << ","
            << row.teardown.free_pre_ms << ","
            << row.teardown.free_pinned_ms << ","
            << row.ingress_fwd.mean_ms << ","
            << row.h2d.mean_ms << ","
            << row.fwd_pad_ntt.mean_ms << ","
            << row.fwd_pad_ntt_a.mean_ms << ","
            << row.fwd_pad_ntt_b.mean_ms << ","
            << row.pointwise_mul.mean_ms << ","
            << row.intt.mean_ms << ","
            << row.crt.mean_ms << ","
            << row.carry.mean_ms << ","
            << row.d2h.mean_ms << "\n";
    }
}

static void print_row(const BenchRow& row)
{
    const double setup_total = row.setup.pinned_ms + row.setup.precompute_ms
                             + row.setup.upload_ms + row.setup.alloc_ctx_ms;
    const double teardown_total = row.teardown.free_ctx_ms + row.teardown.free_pre_ms
                                + row.teardown.free_pinned_ms;

    cout << fixed << setprecision(3);
    cout << "L_arg=" << setw(3) << row.L_arg
         << "  L=" << setw(10) << row.L
         << "  N=" << setw(10) << row.N
         << "  logN=" << setw(2) << row.logN << "\n";
    cout << "  setup:   pinned=" << setw(8) << row.setup.pinned_ms
         << "  precompute=" << setw(8) << row.setup.precompute_ms
         << "  upload=" << setw(8) << row.setup.upload_ms
         << "  alloc=" << setw(8) << row.setup.alloc_ctx_ms
         << "  (total " << setup_total << " ms)\n";
    cout << "    pre: factors=" << row.setup.precompute.factors_ms
         << "  params=" << row.setup.precompute.params_ms
         << "  twiddle_host=" << row.setup.precompute.twiddle_host_ms
         << "  garner_host=" << row.setup.precompute.garner_host_ms << " ms\n";
    cout << "    upload: twiddle=" << row.setup.upload.twiddle_upload_ms
         << "  mod_constants=" << row.setup.upload.mod_constants_ms
         << "  garner=" << row.setup.upload.garner_upload_ms << " ms\n";
    cout << "  execute: mean=" << setw(8) << row.execute_total.mean_ms
         << "  stddev=" << setw(7) << row.execute_total.stddev_ms
         << "  min=" << setw(8) << row.execute_total.min_ms
         << "  max=" << setw(8) << row.execute_total.max_ms << " ms\n";
    cout << "    ingress_fwd=" << row.ingress_fwd.mean_ms
         << " (h2d=" << row.h2d.mean_ms
         << " fwd=" << row.fwd_pad_ntt.mean_ms
         << " a=" << row.fwd_pad_ntt_a.mean_ms
         << " b=" << row.fwd_pad_ntt_b.mean_ms << " — streams overlap, use ingress_fwd)"
         << "  mul=" << row.pointwise_mul.mean_ms
         << "  intt=" << row.intt.mean_ms
         << "  crt=" << row.crt.mean_ms
         << "  carry=" << row.carry.mean_ms
         << "  d2h=" << row.d2h.mean_ms << " ms\n";
    cout << "  teardown: free_ctx=" << setw(8) << row.teardown.free_ctx_ms
         << "  free_pre=" << setw(8) << row.teardown.free_pre_ms
         << "  free_pinned=" << setw(8) << row.teardown.free_pinned_ms
         << "  (total " << teardown_total << " ms)\n";
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    int warmup = DEFAULT_WARMUP;
    int iters = DEFAULT_ITERS;
    string csv_path = "gpu_multiply_bench.csv";
    bool csv_append = false;
    vector<size_t> L_args;

    for (int i = 1; i < argc; i++) {
        const string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            return 0;
        }
        if (arg == "--warmup") {
            if (i + 1 >= argc) {
                cerr << "Missing value for --warmup\n";
                return 1;
            }
            warmup = atoi(argv[++i]);
            continue;
        }
        if (arg == "--iters") {
            if (i + 1 >= argc) {
                cerr << "Missing value for --iters\n";
                return 1;
            }
            iters = atoi(argv[++i]);
            continue;
        }
        if (arg == "--csv") {
            if (i + 1 >= argc) {
                cerr << "Missing value for --csv\n";
                return 1;
            }
            csv_path = argv[++i];
            continue;
        }
        if (arg == "--append") {
            csv_append = true;
            continue;
        }

        if (!append_l_spec(arg, L_args)) {
            cerr << "Invalid limb spec: " << arg << "\n";
            usage(argv[0]);
            return 1;
        }
    }

    if (L_args.empty())
        L_args = DEFAULT_L_VALUES;

    if (warmup < 0 || iters <= 0) {
        cerr << "warmup must be >= 0 and iters must be > 0\n";
        return 1;
    }

    cout << "GPU full multiply benchmark"
         << " (LIMB_BITS=" << LIMB_BITS
         << ", warmup=" << warmup
         << ", iters=" << iters
         << ", max L_arg=" << (max_supported_logN() - 1)
         << " / max L=" << max_supported_limb_count() << ")\n";
    cout << string(72, '-') << "\n";

    warmup_cuda_runtime();

    vector<BenchRow> rows;
    rows.reserve(L_args.size());

    uint64_t seed = 1234;
    for (size_t L_arg : L_args) {
        const size_t L = resolve_limb_count(L_arg);
        const size_t N = padded_ntt_size(L, L);
        string why;
        if (!ntt_size_supported(N, &why)) {
            cerr << "Skipping L_arg=" << L_arg << " (L=" << L
                 << ", N=" << N << "): " << why << "\n";
            continue;
        }

        cout << "Benchmarking L_arg=" << L_arg << " (L=" << L << ") ... " << flush;
        BenchRow row = benchmark_L(L_arg, warmup, iters, seed);
        seed += 17;
        rows.push_back(row);
        cout << "done\n";
        print_row(row);
    }

    write_csv(csv_path, rows, csv_append);
    cout << string(72, '-') << "\n";
    cout << "Wrote " << rows.size() << " rows to " << csv_path << "\n";
    return 0;
}
