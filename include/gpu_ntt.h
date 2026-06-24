#pragma once

#include "ntt.cuh"
#include "config.h"
#include "crt_gpu.h"
#include "carry_prop.h"
#include <vector>

using namespace std;
using namespace gpuntt;

#if LIMB_BITS == 64
    typedef Data64 TestDataType;
#else
    typedef Data32 TestDataType;
#endif

struct NTTContext {
    size_t N;
    int logN;

    // inputs are always 32-bits
    // changed from TestDataTypeUint to uint32_t
    uint32_t* a_raw_dev = nullptr;    // size L_A
    uint32_t* b_raw_dev = nullptr;    // size L_B
    size_t L_A = 0, L_B = 0;

    vector<TestDataType*> a_dev;
    vector<TestDataType*> b_dev;
    vector<TestDataType*> c_dev;

    vector<Root<TestDataType>*> forward_omega_dev;
    vector<Root<TestDataType>*> inverse_omega_dev;

    vector<Modulus<TestDataType>*> modulus_dev;
    vector<Ninverse<TestDataType>*> ninv_dev;

    cudaStream_t stream_a, stream_b;

    uint64_t* d_C_hi;
    uint64_t* d_C_lo;

    // changed from uint32_t to TestDataTypeUint
    TestDataTypeUint* d_out;
    int64_t*    d_seg_carry;
};

// Host precompute for transform size N. GPU pointers are null until
// upload_ntt_precomputed() is called (setup bucket).
struct NTTPrecomputed {
	size_t N;
	int logN;
	vector<NTTParameters<TestDataType>> params;
	vector<vector<Root<TestDataType>>> forward_omega_host;
	vector<vector<Root<TestDataType>>> inverse_omega_host;
	CRTGarnerParams garner;
	bool gpu_uploaded = false;
	vector<Root<TestDataType>*>         forward_omega_dev;
	vector<Root<TestDataType>*>         inverse_omega_dev;
	vector<Modulus<TestDataType>*>    modulus_dev;
	vector<Ninverse<TestDataType>*> ninv_dev;
};

// Host-only precompute_ntt breakdown (ms).
struct PrecomputeTiming {
	float factors_ms = 0.0f;
	float params_ms = 0.0f;
	float twiddle_host_ms = 0.0f;
	float garner_host_ms = 0.0f;
	float total_ms = 0.0f;
};

// GPU upload of precomputed tables (setup bucket).
struct SetupUploadTiming {
	float twiddle_upload_ms = 0.0f;
	float mod_constants_ms = 0.0f;
	float garner_upload_ms = 0.0f;
	float total_ms = 0.0f;
};

NTTPrecomputed precompute_ntt(size_t N, PrecomputeTiming* timing_out = nullptr);
void upload_ntt_precomputed(NTTPrecomputed& pre, SetupUploadTiming* timing_out = nullptr);
NTTContext allocate_ntt_context(const NTTPrecomputed &pre, size_t L_A, size_t L_B);

// Per-invocation GPU stage timings (CUDA events, ms).
//
// ingress_fwd_ms — wall-clock critical path for the parallel ingress phase
//   (H2D + zero-pad + forward NTT on streams a/b). Use this when summing stages.
//
// h2d_ms, fwd_pad_ntt_* — per-stream diagnostics; the two streams overlap in
//   wall time, so do not add h2d_ms + fwd_pad_ntt_ms.
struct NTTTiming {
	float ingress_fwd_ms = 0.0f;   // max wall time stream_a vs stream_b
	float h2d_ms = 0.0f;           // max(copy_a, copy_b) — diagnostic
	float fwd_pad_ntt_ms = 0.0f;   // max(fwd_a, fwd_b) after copies — diagnostic
	float fwd_pad_ntt_a_ms = 0.0f;
	float fwd_pad_ntt_b_ms = 0.0f;
	float pointwise_mul_ms = 0.0f;
	float intt_ms = 0.0f;
	float crt_ms = 0.0f;
	float carry_ms = 0.0f;
	float d2h_ms = 0.0f;
	float total_ms = 0.0f;         // wall clock, execute start -> D2H complete
};

void execute_ntt_multiply(
	NTTContext &ctx,
    // changed from TestDataTypeUint to uint32_t
	const uint32_t* a_pinned,
    // changed from TestDataTypeUint to uint32_t
	const uint32_t* b_pinned,
	vector<TestDataTypeUint> &C_out,
	__int128 M, __int128 M_half,
	NTTTiming* timing_out = nullptr
);

void cleanup_ntt_context(NTTContext &ctx);
void cleanup_ntt_precomputed(NTTPrecomputed &pre);
