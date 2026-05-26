// test_gpu_ntt_smoke.cu
#include "gpu_ntt.h"
#include "config.h"
#include <cstdio>

int main() {
    printf("=== gpu_ntt smoke test (LIMB_BITS=%d) ===\n", LIMB_BITS);

    size_t N = 1 << 10;

    printf("precompute_ntt...\n");
    NTTPrecomputed pre = precompute_ntt(N);
    printf("  OK: N=%zu logN=%d\n", pre.N, pre.logN);

    printf("allocate_ntt_context...\n");
    NTTContext ctx = allocate_ntt_context(pre, N/2, N/2);
    printf("  OK\n");

    cleanup_ntt_context(ctx);
    cleanup_ntt_precomputed(pre);

    printf("PASS\n");
    return 0;
}