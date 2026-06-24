# ntt_cgbn — Codebase Roadmap

GPU-accelerated large-integer multiplication via the **Number Theoretic Transform (NTT)** in a **Residue Number System (RNS)**, with **Chinese Remainder Theorem (CRT)** reconstruction and **segmented carry propagation**.

**Host inputs are always 32-bit limbs** (`uint32_t`), regardless of build width. The compile-time `LIMB_BITS` flag selects NTT arithmetic width, RNS prime count, and **output** limb type — not the host input format.

---

## High-level pipeline

Large integers `A` and `B` (as limb arrays) are multiplied by treating them as polynomials, transforming to the frequency domain, multiplying pointwise, transforming back, and reconstructing the integer result.

```
  Host input (32-bit limbs)
        │
        ▼
  ┌─────────────┐
  │  zero_pad   │  zero-extend each uint32_t limb to TestDataTypeUint,
  │             │  then pad to length N (next power of 2 ≥ L_A + L_B − 1)
  └──────┬──────┘
         │  (per modulus, on separate CUDA streams)
         ▼
  ┌─────────────┐
  │  Forward    │  GPU NTT (external GPU-NTT library)
  │  NTT        │
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │  Pointwise  │  coefficient-wise product mod p_i
  │  multiply   │
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │  Inverse    │  GPU INTT
  │  NTT        │
  └──────┬──────┘
         │  residues c_dev[i][k] for each modulus i, coefficient k
         ▼
  ┌─────────────┐
  │  CRT (GPU)  │  Garner's algorithm → 128-bit coefficients (hi, lo)
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │  Carry      │  3-pass segmented propagation → final limbs
  │  propagation│
  └──────┬──────┘
         ▼
  Host output (TestDataTypeUint limbs)
```

The orchestration lives in `execute_ntt_multiply()` (`src/gpu_ntt.cu`). The host entry point is `host_multiply_merge()` (`src/host_multiply.cpp`).

---

## Directory layout

```
ntt_cgbn/
├── ROADMAP.md          ← this file
├── Makefile            ← top-level build entry (includes make/*.mk)
├── main.cpp            ← CLI driver (merge / 4step / benchmark; mostly stubbed)
│
├── include/            ← public headers
├── src/                ← implementation (.cpp host, .cu GPU)
├── tests/              ← unit and integration tests
├── bench/              ← performance benchmarks
├── make/               ← Makefile fragments
└── scripts/            ← offline Python helpers (primes, CRT, verification)
```

### `include/` — API surface

| File | Role |
|------|------|
| `config.h` | Compile-time knobs: `LIMB_BITS` (32 or 64), `NUM_MODULI`, `CARRY_SEG`, type aliases |
| `gpu_ntt.h` | Main GPU multiply API: `NTTPrecomputed`, `NTTContext`, `precompute_ntt`, `execute_ntt_multiply` |
| `multiply.h` | Host multiply entry: `host_multiply_merge` |
| `zero_pad.h` | GPU zero-padding kernel wrapper |
| `carry_prop.h` | Three carry-propagation `__global__` kernels |
| `crt.h` | Host-side CRT (2-prime and general Garner) |
| `crt_gpu.h` | GPU CRT: `CRTGarnerParams`, `crt_combine_gpu` |
| `crt_utils.h` | Shared helpers: `modinv_u64`, `print_u128` |
| `ntt_limits.h` | Max supported NTT size and operand limb count for current moduli |

Headers pull in types from the external **GPU-NTT** library (`ntt.cuh`, `modular_arith.cuh`) via include paths in `make/config.mk`.

### `src/` — implementations

| File | Role |
|------|------|
| `gpu_ntt.cu` | Core pipeline: NTT precomputation, context allocation, `execute_ntt_multiply`, `pointwise_mul_kernel`, moduli/roots tables |
| `host_multiply.cpp` | Wraps GPU pipeline for host callers; writes timing CSV |
| `zero_pad.cu` | `zero_pad_kernel` — zero-extends `uint32_t` inputs to `TestDataTypeUint`, pads to length `N` |
| `crt_gpu.cu` | Garner CRT on GPU; constant-memory params; 128-bit coefficient output |
| `crt.cpp` | Host CRT reference (`crt_combine_2`, `crt_combine_many`) |
| `crt_utils.cpp` | `modinv_u64` implementation |
| `carry_prop.cu` | Intra-segment, inter-segment, and fixup carry kernels |

### `tests/` — verification

| Test | What it checks |
|------|----------------|
| `test_zero_pad.cu` | Zero-padding correctness (32- and 64-bit builds) |
| `test_crt_gpu.cu` | GPU CRT vs host reference (dual-width) |
| `test_carry_prop.cu` | Carry propagation edge cases and random inputs |
| `test_pointwise_mul.cu` | Pointwise multiply kernel |
| `test_crt.cpp` | Host CRT |
| `test_gpu_ntt_smoke.cu` | Minimal GPU NTT smoke test |
| `test_full_multiply.cpp` | End-to-end multiply vs CPU schoolbook and GMP |

Run all tests: `make test`  
Run one: `make test TEST=test_crt`  
Component targets: `make test_zero_pad`, `make test_crt_gpu`, `make test_carry_prop`  
Dual-width E2E: `make main_32`, `make main_64`

### `bench/`

| File | Role |
|------|------|
| `gpu_ntt_benchmark.cu` | NTT/INTT round-trip benchmark comparing 32- vs 64-bit NTT configs (GPU-NTT `Data32` vs `Data64`; separate from the E2E multiply path) |
| `gpu_full_multiply_benchmark.cpp` | End-to-end multiply benchmark: times `execute_ntt_multiply` only (precompute and allocation outside the timed region); sweeps operand limb counts; writes CSV |

Build targets (`make/bench.mk`):

| Target | Binary | Notes |
|--------|--------|-------|
| `make bench` | `build/bench_ntt` | NTT round-trip only |
| `make bench_full_32` | `build/bench_full_multiply_32` | Full pipeline, `LIMB_BITS=32`, 3-prime RNS |
| `make bench_full_64` | `build/bench_full_multiply_64` | Full pipeline, `LIMB_BITS=64`, 2-prime RNS |
| `make bench_full` | both of the above | |

The full-multiply benchmark accepts `--warmup N`, `--iters N`, `--csv FILE`, `--append`, and limb specs (`L` values `< 64` mean `1<<L` limbs; ranges like `16-24` are inclusive). Output CSV columns: `limb_bits,L_arg,L,N,logN,warmup,iters,mean_ms,stddev_ms,min_ms,max_ms`.

### `make/` — build system

| File | Role |
|------|------|
| `config.mk` | Toolchain, flags, include/lib paths, external deps |
| `sources.mk` | Source/object file lists |
| `rules.mk` | Compile rules, `main` target, `clean` |
| `tests.mk` | Test binaries, dual-width targets, `test` phony rules |
| `bench.mk` | Benchmark target |

### `scripts/` — offline tooling

| Script | Purpose |
|--------|---------|
| `prime_finder.py`, `find_small_prime.py` | Search for NTT-suitable primes |
| `find_prim_roots.py`, `auto_prim_roots.py` | Primitive roots and roots of unity |
| `crt.py` | CRT reference and residue debugging |
| `carry.py` | Carry propagation reference |
| `full_mul.py`, `gpu_ntt_verify.py` | End-to-end verification against Python reference |
| `rng.py` | Generate random limb pairs → binary input for `main` |
| `run_gpu_bench.py` | Build and run `bench_full_multiply_{32,64}`; `--limb-bits 32\|64\|both` |
| `plot_bench.py` | Plot `gpu_multiply_bench.csv` (log-log, error bars) → PNG |

---

## External dependencies

Configured in `make/config.mk`:

| Dependency | Path / lib | Used for |
|------------|------------|----------|
| **GPU-NTT** | `$(HOME)/GPU-NTT/src/include`, `-lntt-1.0` | `GPU_NTT_Inplace`, `GPU_INTT`, `NTTParameters`, root tables |
| **GMP** | `$(HOME)/gmp-local/lib`, `-lgmp` | Reference arithmetic in `test_full_multiply.cpp` |
| **CUDA** | `/usr/local/cuda` | All `.cu` compilation and runtime |

Moduli and 2²³-th roots of unity are hardcoded in `gpu_ntt.cu` and derived at runtime for each transform size `N = 2^logN`.

**CGBN is not used.** The repo name reflects an earlier plan, but no source file includes CGBN headers or calls `cgbn_*` APIs. Arithmetic is implemented locally (`pointwise_mul_kernel`, Garner CRT, carry kernels).

---

## Configuration

Set at compile time via `-DLIMB_BITS=32|64` (default 32 in `config.h`):

| `LIMB_BITS` | NTT type (`gpu_ntt.cu`) | Output limb type | `NUM_MODULI` | RNS primes (`gpu_ntt.cu`) |
|-------------|-------------------------|------------------|--------------|---------------------------|
| 32 | `Data32` | `uint32_t` | 3 | `0x2d000001`, `0x23800001`, `0x26800001` |
| 64 | `Data64` | `uint64_t` | 2 | `0x6723cbb800001`, `0x6723cb6800001` (62-bit NTT-friendly) |

### Input vs output limb width

| Stage | Width |
|-------|-------|
| Host input (`main`, benchmarks, `execute_ntt_multiply`) | Always `uint32_t` limbs |
| After `zero_pad_kernel` (`src/zero_pad.cu`) | `TestDataTypeUint` — zero-extended cast: upper 32 bits are 0 in 64-bit builds |
| After carry propagation | `TestDataTypeUint` output limbs (`uint32_t` or `uint64_t` per `LIMB_BITS`) |

The **64-bit build** therefore uses a **2-prime RNS** with **32-bit input limbs widened to 64 bits** before NTT. This is intentional: operand size is expressed in 32-bit host limbs while NTT/CRT arithmetic runs at 64-bit width with fewer, larger primes. Verified by `tests/test_zero_pad.cu` (`test_no_upper_bits_set`).

`NTTContext` stores raw inputs as `uint32_t*` (`a_raw_dev` / `b_raw_dev`); per-modulus padded buffers `a_dev[i]` / `b_dev[i]` are `TestDataType*`.

Other defines:

- `DEBUG=1` — verbose GPU/CPU cross-checks inside `execute_ntt_multiply`
- `TIMING=1` — per-stage CUDA event timing → `ntt_timing.csv`
- `PROFILE=1` — debug symbols and profiling flags
- `CARRY_SEG` — segment size for carry propagation (default 1024)

---

## Data structures

### `NTTPrecomputed` (immutable, reusable)

Precomputed per transform size `N`: NTT parameters, device root tables, modulus/n⁻¹ constants, and Garner CRT params. Created once via `precompute_ntt(N)`.

### `NTTContext` (per multiply)

Mutable GPU buffers: padded operands `a_dev`/`b_dev`, frequency-domain products `c_dev`, CRT output `d_C_hi`/`d_C_lo`, final limbs `d_out`, segment carries `d_seg_carry`. Allocated via `allocate_ntt_context(pre, L_A, L_B)`.

### Residue layout

After INTT, `c_dev[i]` holds residues modulo `moduli[i]` for all `N` coefficients. CRT reads these in place (no host round-trip).

---

## Build and run

```bash
make                  # build main
make clean
make test             # all tests
make bench            # NTT round-trip benchmark (build/bench_ntt)
make bench_full       # end-to-end multiply benchmarks (32- and 64-bit)
make bench_full_32    # build/bench_full_multiply_32 only
make bench_full_64    # build/bench_full_multiply_64 only
make DEBUG=1 test     # tests with debug checks
make TIMING=1         # timing instrumentation in gpu_ntt.cu
```

**Full-multiply benchmark** (times `execute_ntt_multiply` only):

```bash
make bench_full_64
./build/bench_full_multiply_64 --warmup 2 --iters 20 --csv gpu_multiply_bench.csv 16-22

# or via runner (builds if needed; use --append for the second width)
python scripts/run_gpu_bench.py --limb-bits both 16-22 --iters 20
python scripts/plot_bench.py gpu_multiply_bench.csv -o gpu_multiply_bench.png
```

Operand sizes are capped by `ntt_limits.h` (`max_supported_limb_count()`, `max_supported_logN()`); unsupported `L` values are skipped with a message.

`main` usage (host driver; multiply calls currently commented out):

```
./main [merge|4step|benchmark] [optional_binary_input]
```

Binary input format: `A` limbs then `B` limbs, each limb `uint32_t`, back-to-back (`scripts/rng.py` can generate this).

---

## Component dependency graph

```
main.cpp / host_multiply.cpp
        │
        └── gpu_ntt.h
                ├── config.h
                ├── crt_gpu.h ──► crt_utils.h
                ├── carry_prop.h
                └── GPU-NTT (ntt.cuh)   [external]

crt_gpu.cu ──► crt_gpu.h, crt_utils.cpp
carry_prop.cu ──► carry_prop.h
zero_pad.cu ──► zero_pad.h
crt.cpp ──► crt.h ──► crt_utils.h
```

---

## Development status and roadmap

Based on recent commit history and in-code TODOs:

### Done

- [x] 32-bit limb pipeline end-to-end (NTT → CRT → carry)
- [x] Unit tests for zero-pad, CRT, carry propagation
- [x] 64-bit elementary tests and working 64-bit path
- [x] Segmented carry propagation (intra / inter / fixup)
- [x] GPU CRT with Garner precomputation in constant memory
- [x] Dual-stream NTT for operands A and B
- [x] End-to-end multiply benchmarks (`bench_full_multiply_{32,64}`) with CSV output and plotting (`run_gpu_bench.py`, `plot_bench.py`)

### In progress / known gaps

- [ ] **Pointwise multiply**: `pointwise_mul_kernel` uses direct `% modulus`; comments note Barrett reduction should replace it (`gpu_ntt.cu`)
- [ ] **`main.cpp`**: `host_multiply_merge` / `host_multiply_4step` calls are commented out; CLI is a stub
- [ ] **4-step multiply**: `multiply.h` references a 4-step method; not wired in `main.cpp`
- [ ] **Hardcoded inverses**: `crt_utils.h` notes modular inverses could be precomputed at build time

### Suggested next steps

1. **Productionize pointwise mul** — Barrett or library modular multiply for 32/64-bit limbs
2. **Reconnect `main.cpp`** — enable `merge` path and optional file I/O for manual testing
3. **Parameter tooling** — promote `scripts/find_prim_roots.py` output into generated headers for new prime sets
4. **64-bit scaling** — use `bench_full_64` to profile larger `N` and limb counts; compare CRT/carry vs NTT (`TIMING=1`)
5. **CI / reproducibility** — document exact GPU-NTT and GMP install steps; pin moduli in a single config source

---

## Quick reference: where to change what

| Goal | Start here |
|------|------------|
| Change primes or roots | `src/gpu_ntt.cu` (`moduli`, `roots_of_unity_2_23`) |
| Adjust multiply pipeline stages | `src/gpu_ntt.cu` → `execute_ntt_multiply` |
| Fix final limb assembly | `src/carry_prop.cu`, `CARRY_SEG` in `config.h` |
| Host-facing API | `include/multiply.h`, `src/host_multiply.cpp` |
| CRT algorithm (GPU) | `src/crt_gpu.cu`, `include/crt_gpu.h` |
| CRT reference (CPU) | `src/crt.cpp`, `scripts/crt.py` |
| Add a test | `tests/`, register via wildcard in `make/tests.mk` |
| Run full-multiply benchmarks | `make bench_full`, `scripts/run_gpu_bench.py` |
| Plot benchmark results | `scripts/plot_bench.py` |
| Check max operand size | `include/ntt_limits.h`, `src/ntt_limits.cpp` |
| Build flags / deps | `make/config.mk` |
