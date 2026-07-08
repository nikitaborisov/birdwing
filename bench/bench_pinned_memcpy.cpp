// bench/bench_pinned_memcpy.cpp
//
// Self-contained benchmark for cudaMallocHost, cudaFreeHost, and memcpy at
// sizes from a few KiB up to 1 GiB. Times each operation separately.
//
// Build:  make bench_pinned_memcpy
// Run:    ./build/bench_pinned_memcpy [--warmup N] [--iters N] [--csv FILE] [sizes ...]
//
// Size spec: bare integers are byte counts; suffixes k/m/g (or K/M/G) scale by
// 1024. Default sweep is powers of two from 4 KiB through 1 GiB.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

static constexpr int DEFAULT_WARMUP = 2;
static constexpr int DEFAULT_ITERS  = 20;

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _err = (expr);                                             \
        if (_err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(_err));                                 \
            abort();                                                           \
        }                                                                      \
    } while (0)

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

    s.mean_ms = sum / static_cast<double>(samples.size());
    if (samples.size() > 1) {
        const double variance =
            (sum_sq - sum * sum / static_cast<double>(samples.size()))
            / static_cast<double>(samples.size() - 1);
        s.stddev_ms = sqrt(max(0.0, variance));
    }
    return s;
}

static bool file_nonempty(const string& path)
{
    ifstream f(path);
    return f.good() && f.peek() != ifstream::traits_type::eof();
}

static void warmup_cuda_runtime()
{
    void* pinned = nullptr;
    void* dev = nullptr;
    CUDA_CHECK(cudaMallocHost(&pinned, 4096));
    CUDA_CHECK(cudaMalloc(&dev, 4096));
    CUDA_CHECK(cudaFree(dev));
    CUDA_CHECK(cudaFreeHost(pinned));
}

static const vector<size_t> default_sizes()
{
    vector<size_t> sizes;
    for (int exp = 12; exp <= 30; ++exp)
        sizes.push_back(size_t(1) << exp);
    return sizes;
}

static bool parse_size(const string& token, size_t& out)
{
    if (token.empty())
        return false;

    char* end = nullptr;
    const unsigned long long base = strtoull(token.c_str(), &end, 10);
    if (end == token.c_str())
        return false;

    size_t scale = 1;
    if (*end != '\0') {
        switch (*end) {
        case 'k':
        case 'K':
            scale = size_t(1) << 10;
            ++end;
            break;
        case 'm':
        case 'M':
            scale = size_t(1) << 20;
            ++end;
            break;
        case 'g':
        case 'G':
            scale = size_t(1) << 30;
            ++end;
            break;
        default:
            return false;
        }
        if (*end != '\0')
            return false;
    }

    out = static_cast<size_t>(base) * scale;
    return out > 0;
}

static string format_bytes(size_t nbytes)
{
    ostringstream oss;
    oss << fixed << setprecision(2);
    if (nbytes >= (size_t(1) << 30)) {
        oss << static_cast<double>(nbytes) / (size_t(1) << 30) << " GiB";
    } else if (nbytes >= (size_t(1) << 20)) {
        oss << static_cast<double>(nbytes) / (size_t(1) << 20) << " MiB";
    } else if (nbytes >= (size_t(1) << 10)) {
        oss << static_cast<double>(nbytes) / (size_t(1) << 10) << " KiB";
    } else {
        oss << nbytes << " B";
    }
    return oss.str();
}

static double bandwidth_gib_s(size_t nbytes, double ms)
{
    if (ms <= 0.0)
        return 0.0;
    return (static_cast<double>(nbytes) / (1024.0 * 1024.0 * 1024.0))
           / (ms / 1000.0);
}

struct BenchRow {
    size_t nbytes = 0;
    TimingStats malloc_host;
    TimingStats free_host;
    TimingStats memcpy_to_pinned;
    TimingStats memcpy_from_pinned;
};

static bool try_alloc_pinned(void** out, size_t nbytes, string& err)
{
    cudaError_t status = cudaMallocHost(out, nbytes);
    if (status == cudaSuccess)
        return true;
    err = cudaGetErrorString(status);
    *out = nullptr;
    return false;
}

static BenchRow bench_size(size_t nbytes, int warmup, int iters)
{
    BenchRow row{};
    row.nbytes = nbytes;

    for (int i = 0; i < warmup; ++i) {
        void* tmp = nullptr;
        CUDA_CHECK(cudaMallocHost(&tmp, nbytes));
        CUDA_CHECK(cudaFreeHost(tmp));
    }

    vector<double> malloc_samples;
    malloc_samples.reserve(static_cast<size_t>(iters));
    for (int i = 0; i < iters; ++i) {
        void* pinned = nullptr;
        const double ms = time_host_ms([&] {
            CUDA_CHECK(cudaMallocHost(&pinned, nbytes));
        });
        malloc_samples.push_back(ms);
        CUDA_CHECK(cudaFreeHost(pinned));
    }
    row.malloc_host = compute_stats(malloc_samples);

    vector<double> free_samples;
    free_samples.reserve(static_cast<size_t>(iters));
    for (int i = 0; i < iters; ++i) {
        void* pinned = nullptr;
        CUDA_CHECK(cudaMallocHost(&pinned, nbytes));
        const double ms = time_host_ms([&] {
            CUDA_CHECK(cudaFreeHost(pinned));
        });
        free_samples.push_back(ms);
        pinned = nullptr;
    }
    row.free_host = compute_stats(free_samples);

    vector<uint8_t> paged(nbytes);
    void* pinned = nullptr;
    CUDA_CHECK(cudaMallocHost(&pinned, nbytes));
    memset(paged.data(), 0xA5, nbytes);
    memset(pinned, 0, nbytes);

    for (int i = 0; i < warmup; ++i) {
        memcpy(pinned, paged.data(), nbytes);
        memcpy(paged.data(), pinned, nbytes);
    }

    vector<double> to_pinned_samples;
    to_pinned_samples.reserve(static_cast<size_t>(iters));
    for (int i = 0; i < iters; ++i) {
        const double ms = time_host_ms([&] {
            memcpy(pinned, paged.data(), nbytes);
        });
        to_pinned_samples.push_back(ms);
    }
    row.memcpy_to_pinned = compute_stats(to_pinned_samples);

    vector<double> from_pinned_samples;
    from_pinned_samples.reserve(static_cast<size_t>(iters));
    for (int i = 0; i < iters; ++i) {
        const double ms = time_host_ms([&] {
            memcpy(paged.data(), pinned, nbytes);
        });
        from_pinned_samples.push_back(ms);
    }
    row.memcpy_from_pinned = compute_stats(from_pinned_samples);

    CUDA_CHECK(cudaFreeHost(pinned));
    return row;
}

static void print_row(const BenchRow& row)
{
    cout << setw(12) << row.nbytes
         << "  (" << setw(9) << format_bytes(row.nbytes) << ")"
         << "  malloc=" << setw(9) << row.malloc_host.mean_ms
         << "  free=" << setw(9) << row.free_host.mean_ms
         << "  to_pin=" << setw(9) << row.memcpy_to_pinned.mean_ms
         << " (" << setw(6) << fixed << setprecision(2)
         << bandwidth_gib_s(row.nbytes, row.memcpy_to_pinned.mean_ms) << " GiB/s)"
         << "  from_pin=" << setw(9) << row.memcpy_from_pinned.mean_ms
         << " (" << setw(6) << bandwidth_gib_s(row.nbytes, row.memcpy_from_pinned.mean_ms)
         << " GiB/s)"
         << defaultfloat << setprecision(6)
         << '\n';
}

static void write_csv_header(ostream& out)
{
    out << "size_bytes,"
        << "malloc_mean_ms,malloc_stddev_ms,"
        << "free_mean_ms,free_stddev_ms,"
        << "memcpy_to_pinned_mean_ms,memcpy_to_pinned_stddev_ms,memcpy_to_pinned_gib_s,"
        << "memcpy_from_pinned_mean_ms,memcpy_from_pinned_stddev_ms,memcpy_from_pinned_gib_s\n";
}

static void write_csv_row(ostream& out, const BenchRow& row)
{
    out << row.nbytes << ","
        << row.malloc_host.mean_ms << "," << row.malloc_host.stddev_ms << ","
        << row.free_host.mean_ms << "," << row.free_host.stddev_ms << ","
        << row.memcpy_to_pinned.mean_ms << "," << row.memcpy_to_pinned.stddev_ms << ","
        << bandwidth_gib_s(row.nbytes, row.memcpy_to_pinned.mean_ms) << ","
        << row.memcpy_from_pinned.mean_ms << "," << row.memcpy_from_pinned.stddev_ms << ","
        << bandwidth_gib_s(row.nbytes, row.memcpy_from_pinned.mean_ms) << "\n";
}

static void usage(const char* prog)
{
    cerr << "Usage: " << prog
         << " [--warmup N] [--iters N] [--csv FILE] [size ...]\n"
         << "\n"
         << "Times cudaMallocHost, cudaFreeHost, and memcpy (paged<->pinned)\n"
         << "for each size. Sizes default to 4 KiB .. 1 GiB (powers of two).\n"
         << "Size tokens accept optional k/m/g suffixes, e.g. 64m, 1g.\n";
}

int main(int argc, char** argv)
{
    int warmup = DEFAULT_WARMUP;
    int iters = DEFAULT_ITERS;
    string csv_path;
    vector<size_t> sizes;

    for (int i = 1; i < argc; ++i) {
        const string arg = argv[i];
        if (arg == "--warmup" && i + 1 < argc) {
            warmup = atoi(argv[++i]);
        } else if (arg == "--iters" && i + 1 < argc) {
            iters = atoi(argv[++i]);
        } else if (arg == "--csv" && i + 1 < argc) {
            csv_path = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            return 0;
        } else {
            size_t nbytes = 0;
            if (!parse_size(arg, nbytes)) {
                cerr << "Unknown argument: " << arg << '\n';
                usage(argv[0]);
                return 1;
            }
            sizes.push_back(nbytes);
        }
    }

    if (sizes.empty())
        sizes = default_sizes();

    sort(sizes.begin(), sizes.end());
    sizes.erase(unique(sizes.begin(), sizes.end()), sizes.end());

    if (warmup < 0 || iters <= 0) {
        cerr << "warmup must be >= 0 and iters must be > 0\n";
        return 1;
    }

    warmup_cuda_runtime();

    ofstream csv;
    if (!csv_path.empty()) {
        const bool need_header = !file_nonempty(csv_path);
        csv.open(csv_path, ios::app);
        if (!csv) {
            cerr << "Failed to open CSV: " << csv_path << '\n';
            return 1;
        }
        if (need_header)
            write_csv_header(csv);
    }

    cout << "pinned host benchmark"
         << "  warmup=" << warmup
         << "  iters=" << iters
         << '\n';
    cout << setw(12) << "bytes"
         << "  size"
         << "  malloc_ms  free_ms    to_pinned_ms (GiB/s)     from_pinned_ms (GiB/s)\n";

    vector<BenchRow> rows;
    rows.reserve(sizes.size());

    for (size_t nbytes : sizes) {
        void* probe = nullptr;
        string err;
        if (!try_alloc_pinned(&probe, nbytes, err)) {
            cerr << "Skipping " << nbytes << " bytes (" << format_bytes(nbytes)
                 << "): cudaMallocHost failed: " << err << '\n';
            continue;
        }
        CUDA_CHECK(cudaFreeHost(probe));

        const BenchRow row = bench_size(nbytes, warmup, iters);
        rows.push_back(row);
        print_row(row);
        if (csv)
            write_csv_row(csv, row);
    }

    if (rows.empty()) {
        cerr << "No sizes completed successfully.\n";
        return 1;
    }

    if (!csv_path.empty())
        cout << "Wrote CSV: " << csv_path << '\n';

    return 0;
}
