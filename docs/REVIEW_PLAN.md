# Implementation Plan — Correctness Verification & Performance

Derived from the code review of `ntt_cgbn`. Two goals, in priority order:

1. **Correctness verification** — make every "PASS" trustworthy and close coverage gaps.
2. **Performance** — remove the device-side 128-bit division hot spots and fix kernel launch geometry.

This document is the working plan for a GPU box. Each phase is an independent, mergeable
unit with its own verification step. Do the phases in order: the verification
infrastructure (Phase 1) must land first so the later changes are provably safe.

---

## 0. Environment setup & baseline (do once on the GPU box)

Before changing anything, establish a known-good baseline to compare against.

```bash
# Build all three pipelines and run the full suite
make CMAKE_CUDA_ARCHITECTURES=<arch>          # e.g. 86 for RTX 3080, 89 for Ada
make test                                     # 32-bit shared-object suite
make test_zero_pad test_crt_gpu test_carry_prop test_ntt_limits   # all 3 widths
make main_32 main_hybrid main_64bit && \
  build/test_full_multiply_32 && \
  build/test_full_multiply_hybrid && \
  build/test_full_multiply_64bit
```

**Capture baseline numbers** (commit these CSVs or save outside the tree):

```bash
make bench_full
python scripts/run_gpu_bench.py --limb-bits all 12-22 --iters 20 --csv baseline_bench.csv
# optional: nsys profile to see where time actually goes per stage
make TIMING=1 main_32 && build/test_full_multiply_32   # writes ntt_timing.csv
```

Record: per-stage `ntt_timing.csv` (MUL / INTT / CRT / CARRY / D2H) and `baseline_bench.csv`.
These are the before/after reference for Phases 2–3.

> Note on object cache: the shared `build/*.o` cache is always `LIMB_BITS=32`. The
> dual-width targets (`main_*`, `bench_full_*`, component tests) recompile sources in a
> single `nvcc` line, so they're safe. If anything looks stale after editing a `.cu`,
> `make clean` first.

---

## Phase 1 — Correctness verification infrastructure (land first)

### 1a. CUDA error checking (review A1)
**Problem:** no `cudaGetLastError`/return-code checks anywhere. A kernel that fails to launch
yields stale-memory "results" that pass comparison — PASS is currently not trustworthy.

**Changes:**
- Add `include/cuda_check.h` with:
  - `CUDA_CHECK(expr)` — checks `cudaError_t` from runtime calls, prints file:line + `cudaGetErrorString`, aborts.
  - `CUDA_CHECK_KERNEL()` — `cudaGetLastError()` after every launch; under `-DDEBUG=1` also `cudaDeviceSynchronize()` + check.
- Wrap in `src/gpu_ntt.cu`, `src/crt_gpu.cu`, `src/zero_pad.cu`, `src/carry_prop.cu`,
  `src/host_multiply.cpp`: all `cudaMalloc`, `cudaMemcpy*`, `cudaMemcpyToSymbol`,
  `cudaStreamCreate`, `cudaEvent*`, and a `CUDA_CHECK_KERNEL()` after each `<<<>>>`.

**Verify:** full suite still passes; temporarily break one launch (e.g. absurd block count)
and confirm it now aborts with a clear message instead of silently passing.

**Risk:** low. Pure instrumentation. Keep the sync-heavy variant behind `DEBUG` so release perf is unaffected.

### 1b. Full-width + edge-value test inputs (review A2, A6)
**Problem:** `tests/test_full_multiply.cpp:69-70` masks 32-bit inputs to 30 bits and disables
the MAX case, so the default pipeline is never tested at full width or near the CRT bound.
`test_full_pipeline` also only prints `[FAIL]` and continues — `make test` can pass with a wrong multiply.

**Changes (`tests/test_full_multiply.cpp`):**
- `random_limbs`: use full 32-bit values; restore the `(0, 1, UINT32_MAX)` edge triple
  (mirror `random_limbs_u64` at line 431-441).
- Make GMP the asserting oracle for **all** correctness tests; `exit(1)` on mismatch like
  `test_native_pipeline` (line 503), so failures fail the build.
- Add a regression at `L = max_supported_limb_count()` and at `L-1`/`L+1` around it
  (use `include/ntt_limits.h`) to exercise the CRT coefficient bound exactly.

**Verify:** `build/test_full_multiply_{32,hybrid,64bit}` all green with full-width inputs.
This is also the gate that protects Phases 2–3.

**Risk:** low, but may surface a *latent* bug at full width — that's the point. If it does,
fix before proceeding.

### 1c. Integer `logN`, stale comment, bound guard (review A4, A5, A3)
- `src/gpu_ntt.cu:257` and `:534`: replace `(int)log2(...)` with `__builtin_ctzll(N)` (exact).
- `src/crt_gpu.cu:25-26`: rewrite the `mul128_scalar` comment to state the real invariant
  (running prefix product needs ≤128 bits before use; final `M*=p` at last `j` is dead).
- `src/carry_prop.cu` / `include/gpu_ntt.h:53`: the non-native `int64_t d_seg_carry` can
  overflow for hybrid at very large `L` (carry → ~2^64 > INT64_MAX). **Decision: guard + assert**
  — keep `int64_t` (the overflow `L` is unrealistically large) and add a runtime check (and
  `static_assert` where the bound is compile-time derivable) tying hybrid
  `max_supported_limb_count()` to the int64 carry range, so it fails loudly instead of
  silently corrupting. Document the assumed bound at the declaration.

**Verify:** suite green all three widths; the new bound test from 1b still passes.

**Risk:** low.

---

## Phase 2 — Kernel launch geometry & stream cleanup (cheap, high-value perf)

Land after Phase 1 so correctness is gated.

### 2a. Carry kernels: one thread per segment, packed blocks (review B2)
**Problem:** all carry kernels launch `<<<num_segs, 1>>>` (one thread/block; 31/32 of each
warp idle). `src/gpu_ntt.cu:892,896,901,904,911,915`.

**Change:** keep the per-segment serial scan, but map one thread per segment:
- Launch `<<<ceil(num_segs/256), 256>>>`; inside, `seg = blockIdx.x*blockDim.x + threadIdx.x;
  if (seg >= num_segs) return;` (replaces `seg = blockIdx.x` and the `threadIdx.x != 0` guard).
- Apply to intra / fixup (both `_u160` and non-native variants) in `src/carry_prop.cu`.
- The inter-segment kernel is a single serial scan — leave as `<<<1,1>>>`.

**Verify:** `test_carry_prop` all 3 widths (this test directly targets these kernels);
then full-multiply all 3 widths; compare `CARRY` column in `ntt_timing.csv` vs baseline.

**Risk:** low — same algorithm, only the segment→thread mapping changes. The `test_native_pipeline`
already has a cross-segment regression at `L=1<<12` (line 512).

### 2b. CRT on `stream_a`, drop mid-pipeline device sync (review B3)
**Problem:** `crt_combine_gpu*` launch on the default stream and call `cudaDeviceSynchronize()`
(`src/crt_gpu.cu:286-287,300-301`), forcing a global barrier between CRT and carry.

**Change:**
- Add a `cudaStream_t stream` parameter to `crt_combine_gpu` / `crt_combine_gpu_u160`; launch on it.
- Pass `ctx.stream_a` from `execute_ntt_multiply` (`gpu_ntt.cu:813,815`).
- Remove the `cudaDeviceSynchronize()` inside the CRT helpers; ordering is preserved because
  carry kernels run on the same stream. The only required host sync stays where it is
  (fixup escape-flag readback loop, final D2H copy).
- Note the early-return CRT-dump paths (`gpu_ntt.cu:843-884`) must keep their own
  `cudaStreamSynchronize(ctx.stream_a)` before the `cudaMemcpy` D2H.

**Verify:** `test_crt_gpu` + `test_pipeline_crt_{hybrid,64bit}`; full-multiply all 3 widths;
CRT column timing vs baseline.

**Risk:** low–medium (stream ordering). Run under `compute-sanitizer` once (see Phase 5).

### 2c. u160 fixup early-out (review B4)
`src/carry_prop.cu:194`: `carry_fixup_kernel_u160` loops to `seg_end` unconditionally.
Add `&& (c_lo || c_mid || c_hi)` to the loop condition (mirror the non-native kernel at line 69).

**Verify:** `test_carry_prop_64bit`, `test_full_multiply_64bit`.

**Risk:** trivial.

---

## Phase 3 — Barrett reduction on the multiply hot path (highest perf payoff)

**Problem:** device-side software 128-bit division in the two throughput kernels:
- `pointwise_mul_kernel` — `full % modulus` (`src/gpu_ntt.cu:230,235`).
- `mulmod64` 64-bit branch — `prod / p` (`src/crt_gpu.cu:57`).

Both flagged TODO. The 32-bit branch already shows the target pattern
(`__umul64hi`-based Barrett, `crt_gpu.cu:62-65`); the `barrett_m` table already exists.

**Changes:**
- Implement a shared 64-bit Barrett reduce for the ≤60-bit primes (a `__device__` helper,
  e.g. in a new `include/barrett.cuh`): precompute `m = floor(2^k / p)` for suitable `k`
  (the 60-bit primes need a 128-bit intermediate; use `__umul64hi` on the two 64-bit halves
  of the product). Provide `barrett_reduce_128(hi, lo, p, m) -> r`.
- Replace the `%`/`/` in `pointwise_mul_kernel` (feed `mul_wide`'s hi/lo directly — no
  `((u128)hi<<64)|lo` reassembly) and in `mulmod64`'s 64-bit branch.
- Populate the 64-bit `barrett_m` entries in `compute_garner_params`
  (`src/crt_gpu.cu:96-100`, currently `barrett_m[i] = 0` for `LIMB_BITS==64`).

**Verify (correctness is paramount here):**
- Unit-test the Barrett helper against `__uint128_t` reference exhaustively at the boundaries:
  `prod ∈ {0, 1, p-1, (p-1)^2, max}` for each of the 3 primes (add to `test_crt_gpu` or a new
  `test_barrett`). The CRT correctness path (`test_crt_gpu` all widths) already cross-checks
  against the host Garner reference — it must stay green.
- Full-multiply all 3 widths with the **full-width** inputs from Phase 1b.
- Compare `MUL` and `CRT` columns vs baseline; re-run `run_gpu_bench.py` and `plot_bench.py`.

**Decision: replace the `%`/`/` outright** (no `#ifdef` fallback) — rely on the exhaustive
boundary unit test plus the existing host-reference CRT comparison for the safety net.

**Risk:** medium — this is modular-arithmetic correctness. Mitigated by: Phase 1 gating, the
exhaustive boundary unit test against `__uint128_t`, the host-reference CRT comparison
(`test_crt_gpu` all widths), and a `compute-sanitizer` pass. Because there is no compile-time
fallback, do not merge Phase 3 until all three pipelines are green with full-width inputs.

---

## Phase 4 — Lower-priority cleanups (optional, after the above)

- **B5 — pointwise fusion / parallel moduli:** with Barrett in place, drop the 128-bit
  reassembly; optionally run the `NUM_MODULI` pointwise kernels on separate streams or as a
  single grid-strided kernel over `NUM_MODULI*N`.
- **B6 — setup allocations:** replace the many small `cudaMalloc`s in
  `upload_ntt_precomputed`/`allocate_ntt_context` with one arena sliced per modulus. Outside
  the timed region, so only matters for one-shot use. Low priority.

---

## Phase 5 — Validation gate (run before merging each phase)

```bash
# Correctness, all three pipelines
make test
make test_zero_pad test_crt_gpu test_carry_prop test_ntt_limits
make main_32 main_hybrid main_64bit
build/test_full_multiply_32 && build/test_full_multiply_hybrid && build/test_full_multiply_64bit

# Memory/race correctness (run at least for Phase 2 & 3)
compute-sanitizer --tool memcheck  build/test_full_multiply_64bit
compute-sanitizer --tool racecheck build/test_full_multiply_64bit

# Performance delta vs baseline
python scripts/run_gpu_bench.py --limb-bits all 12-22 --iters 20 --csv afterPHASE_bench.csv
python scripts/plot_bench.py afterPHASE_bench.csv -o afterPHASE_bench.png
```

**Merge criteria per phase:** all three pipelines green with full-width inputs (Phase 1b),
`compute-sanitizer` clean, and (Phases 2–3) a measured improvement in the targeted stage
with no regression elsewhere.

---

## Suggested PR sequence

1. **PR1 (Phase 1):** error-checking + full-width tests + hygiene. No perf change; makes the
   rest safe. *Must merge first.*
2. **PR2 (Phase 2):** carry launch geometry + CRT stream + u160 early-out. Cheap perf wins.
3. **PR3 (Phase 3):** Barrett reduction. Biggest perf win; gated by PR1's tests.
4. **PR4 (Phase 4):** optional cleanups.

## File-touch quick reference

| Phase | Primary files |
|-------|---------------|
| 1a | new `include/cuda_check.h`; `src/{gpu_ntt,crt_gpu,zero_pad,carry_prop}.cu`, `src/host_multiply.cpp` |
| 1b | `tests/test_full_multiply.cpp` |
| 1c | `src/gpu_ntt.cu`, `src/crt_gpu.cu`, `src/carry_prop.cu`, `include/gpu_ntt.h` |
| 2a | `src/carry_prop.cu`, `src/gpu_ntt.cu` (launch sites) |
| 2b | `src/crt_gpu.cu`, `include/crt_gpu.h`, `src/gpu_ntt.cu` |
| 2c | `src/carry_prop.cu` |
| 3 | new `include/barrett.cuh`; `src/gpu_ntt.cu` (pointwise), `src/crt_gpu.cu` (mulmod64, barrett_m) |
