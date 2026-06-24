# ntt_cgbn — Codebase Roadmap

GPU-accelerated large-integer multiplication via the **Number Theoretic Transform (NTT)** in a **Residue Number System (RNS)**, with **Chinese Remainder Theorem (CRT)** reconstruction and **segmented carry propagation**.

The repo ships **three compile-time multiply pipelines** (separate binaries). Host input/output limb width depends on the pipeline — see [Multiply pipelines](#multiply-pipelines) below.

---

## Multiply pipelines

Three mutually exclusive builds are selected at compile time via `LIMB_BITS` and optional `NATIVE_HOST_LIMBS` (see `include/config.h`).

| Pipeline | Build flags | Host in | Host out | NTT | `NUM_MODULI` | CRT width | Max square `L` | Binary |
|---|---|---|---|---|---|---|---|---|
| **32-bit** | `-DLIMB_BITS=32` | `uint32_t` | `uint32_t` | `Data32` | 3 | ~93-bit (128-bit storage) | 2²² | `bench_full_multiply_32` |
| **64-bit / 2-mod** | `-DLIMB_BITS=64` | `uint32_t` (widened) | `uint32_t` | `Data64` | 2 | ~118-bit (128-bit storage) | 2³² | `bench_full_multiply_64` |
| **64-bit native** | `-DLIMB_BITS=64 -DNATIVE_HOST_LIMBS` | `uint64_t` | `uint64_t` | `Data64` | 3 | **160-bit** (`U160`: 64+64+32) | 2³¹ | `bench_full_multiply_64native` |

**When to use which:**

- **32-bit:** default; smaller operands; three ~29-bit primes; most mature path.
- **64/2-mod:** operands counted in **32-bit host limbs** but NTT at 64-bit with 2×59-bit primes; output remains 32-bit limbs.
- **64-native:** operands already in **64-bit limbs** (full `uint64_t` range); 3×59-bit RNS; U160 CRT → 64-bit output limbs.

**Coefficient bound:** `max coeff = min(L_A, L_B) × (2^INPUT_LIMB_BITS − 1)²`. At native max square `L = 2³¹`, this is ≈ 2¹⁵⁹ — stored as **U160** (`lo`, `mid`, `hi32`).

**Approved 59-bit primes** (`src/gpu_ntt.cu`):

| Hex | v₂(p−1) | Used in |
|---|---|---|
| `0x400002600000001` | 33 | 64/2-mod p₀, 64native p₀ |
| `0x400004200000001` | 33 | 64/2-mod p₁, 64native p₁ |
| `0x400001100000001` | 32 | **64native p₂ only** (caps `max_logN` at 32) |

**API entry points:**

| Pipeline | Host multiply API |
|---|---|
| 32-bit, 64/2-mod | `host_multiply_merge(vector<uint32_t> …)` |
| 64-native | `host_multiply_merge_native(vector<uint64_t> …)` |

Orchestration: `execute_ntt_multiply()` in `src/gpu_ntt.cu`.

---

## High-level pipeline

Large integers `A` and `B` (as limb arrays) are multiplied by treating them as polynomials, transforming to the frequency domain, multiplying pointwise, transforming back, and reconstructing the integer result.

```
  Host input (limb array — width per pipeline)
        │
        ▼
  ┌─────────────┐
  │  zero_pad   │  32-bit: zero-extend to TestDataTypeUint, pad to N
  │             │  64/2-mod: same (uint32 → uint64 widen)
  │             │  64-native: reduce each limb mod p_i per stream, pad
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
  │  CRT (GPU)  │  Garner → 128-bit (hi, lo) or U160 (lo, mid, hi32)
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │  Carry      │  3-pass segmented propagation → final limbs
  │  propagation│
  └──────┬──────┘
         ▼
  Host output (OutputLimbType — uint32_t or uint64_t per pipeline)
```

The orchestration lives in `execute_ntt_multiply()` (`src/gpu_ntt.cu`). Host entry points: `host_multiply_merge()` (32-bit and 64/2-mod) and `host_multiply_merge_native()` (64-native) in `src/host_multiply.cpp`.

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
| `multiply.h` | Host multiply: `host_multiply_merge` (32/64-2mod), `host_multiply_merge_native` (64-native) |
| `zero_pad.h` | GPU zero-padding / mod-reduce ingress wrapper |
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
| `zero_pad.cu` | `zero_pad_kernel` (uint32 widen); `zero_pad_kernel_u64` (native: mod p_i + pad) |
| `crt_gpu.cu` | Garner CRT on GPU; 128-bit or U160 coefficient output |
| `crt.cpp` | Host CRT reference (`crt_combine_2`, `crt_combine_many`) |
| `crt_utils.cpp` | `modinv_u64` implementation |
| `carry_prop.cu` | Intra-segment, inter-segment, and fixup carry kernels |

### `tests/` — verification

| Test | What it checks |
|------|----------------|
| `test_zero_pad.cu` | Zero-padding / native mod-reduce ingress |
| `test_crt_gpu.cu` | GPU CRT vs host reference (32, 64, 64native/U160) |
| `test_carry_prop.cu` | Carry propagation (128-bit→32-out and U160→64-out) |
| `test_pointwise_mul.cu` | Pointwise multiply kernel |
| `test_crt.cpp` | Host CRT |
| `test_gpu_ntt_smoke.cu` | Minimal GPU NTT smoke test |
| `test_full_multiply.cpp` | End-to-end multiply vs CPU schoolbook and GMP |

Run all tests: `make test`  
Run one: `make test TEST=test_crt`  
Component targets: `make test_zero_pad`, `make test_crt_gpu`, `make test_carry_prop`  
Dual-width E2E: `make main_32`, `make main_64`, `make main_64native`  
Native unit tests: `make test_unit_64native`, `make test_pipeline_crt_64native`

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
| `make bench_full_64native` | `build/bench_full_multiply_64native` | Full pipeline, native `uint64_t` I/O, 3×59-bit RNS |
| `make bench_full` | all three above | |

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
| `rng.py` | Generate random limb pairs → binary input (`--limb-bits 32\|64`) |
| `run_gpu_bench.py` | Build and run `bench_full_multiply_{32,64,64native}`; `--limb-bits 32\|64\|64native\|both\|all` |
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

Set at compile time via `-DLIMB_BITS=32|64` and optional `-DNATIVE_HOST_LIMBS` (default 32 in `config.h`):

| Pipeline | Flags | NTT type | Host in | Host out | `NUM_MODULI` | RNS primes (`gpu_ntt.cu`) |
|---|---|---|---|---|---|---|
| 32-bit | `-DLIMB_BITS=32` | `Data32` | `uint32_t` | `uint32_t` | 3 | `0x2d000001`, `0x23800001`, `0x26800001` |
| 64/2-mod | `-DLIMB_BITS=64` | `Data64` | `uint32_t` (widened) | `uint32_t` | 2 | `0x400002600000001`, `0x400004200000001` |
| 64-native | `-DLIMB_BITS=64 -DNATIVE_HOST_LIMBS` | `Data64` | `uint64_t` | `uint64_t` | 3 | above + `0x400001100000001` |

### Input vs output limb width (per pipeline)

| Pipeline | Host input | After ingress | After carry |
|---|---|---|---|
| 32-bit | `uint32_t` | `uint32_t` in NTT field | `uint32_t` |
| 64/2-mod | `uint32_t` | `uint64_t` (zero-extended) | `uint32_t` |
| 64-native | `uint64_t` | `uint64_t` reduced mod each `p_i` | `uint64_t` |

The **64/2-mod** build uses a **2-prime RNS** with **32-bit input limbs widened to 64 bits** before NTT. Verified by `tests/test_zero_pad.cu` (`test_no_upper_bits_set`).

**64-native** reduces each host limb modulo the stream's NTT prime before padding (limbs may exceed 59-bit primes). `NTTContext` stores raw inputs as `InputLimbType*` (`uint32_t*` or `uint64_t*`).

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

Mutable GPU buffers: padded operands `a_dev`/`b_dev`, frequency-domain products `c_dev`, CRT output (`d_C_hi`/`d_C_lo` or native `d_C_lo`/`d_C_mid`/`d_C_hi32`), final limbs `d_out`, segment carries `d_seg_carry`. Allocated via `allocate_ntt_context(pre, L_A, L_B)`.

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
- [ ] **Dual-width binary**: 32- and 64-bit pipelines require separate executables today (`-DLIMB_BITS` at compile time); see [Dual-width unified executable](#dual-width-unified-executable--investigation)

### Dual-width unified executable — investigation

Today the 32-bit, 64/2-mod, and 64-native multiply pipelines are **mutually exclusive at link time**. Each is a full recompile with `-DLIMB_BITS=32`, `-DLIMB_BITS=64`, or `-DLIMB_BITS=64 -DNATIVE_HOST_LIMBS`, producing separate binaries (`build/bench_full_multiply_{32,64,64native}`, `build/test_full_multiply_{32,64,64native}`). The default `make` target and shared `build/*.o` cache always use `LIMB_BITS=32` (from `config.h`).

**Goal:** one executable that can run either pipeline at runtime (e.g. `--limb-bits 32|64` or automatic selection by operand size).

#### What is *not* a barrier

| Area | Status |
|------|--------|
| **GPU-NTT** (`libntt-1.0.a`) | Already ships explicit instantiations for both `Data32` and `Data64`. `bench/gpu_ntt_benchmark.cu` calls both types in the same translation unit. |
| **Host input format** | Always `uint32_t` limbs; zero-pad widens to `TestDataTypeUint` on device. No input-format fork between builds. |
| **CRT coefficient width** | Garner output is always 128-bit (`d_C_hi` / `d_C_lo`); only final carry-propagation limb width differs. |

The blockers are entirely inside this repo's compile-time configuration and symbol layout.

#### Barrier 1 — `LIMB_BITS` is a global compile-time switch

`include/config.h` gates almost everything through a single `#if LIMB_BITS == 64`:

| Symbol | 32-bit build | 64-bit build |
|--------|--------------|--------------|
| `TestDataTypeUint` | `uint32_t` | `uint64_t` |
| `NUM_MODULI` | 3 | 2 |
| `LIMB_MASK` | `0xFFFFFFFF` | `0xFFFFFFFFFFFFFFFF` |
| NTT element type (`gpu_ntt.h`) | `Data32` | `Data64` |
| RNS primes (`gpu_ntt.cu`) | 3 × ~29-bit | 2 × 62-bit |

Every `.cu` and most `.cpp` files include `config.h` (directly or via `gpu_ntt.h`). There is no runtime width parameter anywhere in the public API (`host_multiply_merge`, `execute_ntt_multiply`, `NTTContext`, etc.).

`if constexpr (sizeof(TestDataTypeUint) == 4)` appears in a few kernels (`pointwise_mul_kernel`, `mulmod64`), but those branches are resolved at compile time — the TU still emits only one width's machine code.

#### Barrier 2 — duplicate linker symbols

If you naïvely compile `gpu_ntt.cu`, `crt_gpu.cu`, `carry_prop.cu`, etc. twice (once per width) and link both object sets, the linker sees **identical unmangled names** with incompatible definitions:

- **Globals:** `moduli`, `roots_of_unity_max` (`gpu_ntt.cu`) — different vector contents and element types.
- **Host functions:** `precompute_ntt`, `allocate_ntt_context`, `execute_ntt_multiply`, `compute_garner_params`, `zero_pad_gpu`, … — same signatures, different struct layouts and loop bounds (`NUM_MODULI`).
- **CUDA device symbols:** `crt_combine_kernel`, `carry_intra_segment_kernel`, `zero_pad_kernel`, `pointwise_mul_kernel` — one mangled name per kernel, but body differs (limb mask/shift, unroll counts, residue pointer element size).

A single `LIMB_BITS` value must be chosen for the whole link unit. The current dual-width Makefile targets sidestep this by passing all sources to one `nvcc` invocation per binary instead of linking width-specific `.o` files.

#### Barrier 3 — CUDA constant memory is width-specific

`src/crt_gpu.cu` declares device constant arrays sized by `NUM_MODULI`:

```cpp
__constant__ uint64_t d_primes[NUM_MODULI];
__constant__ uint64_t d_M_mod_table[NUM_MODULI][NUM_MODULI];
__constant__ TestDataTypeUint* d_residue_ptrs[NUM_MODULI];
```

The 32-bit build allocates 3-modulus tables (9-element `M_mod_table`); the 64-bit build allocates 2-modulus tables (4 elements). `crt_combine_kernel` uses `#pragma unroll` over `NUM_MODULI` and calls `mulmod64`, which contains a `#if LIMB_BITS == 64` device-code fork (Barrett vs exact division). Two width variants cannot share these `__constant__` symbol names in one module.

Upload functions (`upload_garner_params`, `upload_residue_ptrs`) also assume a single active constant-memory layout. Running both pipelines in one process would require **separate constant-memory namespaces** (separate `.cu` TUs or renamed symbols via macros) and a host-side dispatch layer that uploads the correct tables before each multiply.

#### Barrier 4 — struct and buffer layout depends on width

`NTTContext` and `NTTPrecomputed` (`include/gpu_ntt.h`) embed `vector<NTTParameters<TestDataType>>` where `TestDataType` is `Data32` or `Data64`. That changes:

- Per-modulus buffer sizes (`sizeof(TestDataType) * N`)
- Twiddle / modulus / n⁻¹ table types (`Root<Data32>` vs `Root<Data64>`, etc.)
- Number of modulus slots (`NUM_MODULI`)

`CRTGarnerParams` (`include/crt_gpu.h`) has `uint64_t M_mod_table[NUM_MODULI][NUM_MODULI]` — **different struct sizes** between builds (3×3 vs 2×2). A unified host struct would need `MAX_MODULI` padding or separate per-width context types.

Output buffers (`d_out`, `C_out`) are `TestDataTypeUint*`, so the host result vector is either 32- or 64-bit limbs. A unified API must either expose two output modes or always emit one width and convert on the host.

#### Barrier 5 — build system assumes one width per object directory

`make/sources.mk` compiles each `src/*.cu` → `build/*.o` once, with no `LIMB_BITS` suffix. Dual-width test targets (`make/tests.mk`) either:

- Recompile selected sources in a **single** `nvcc` link line (`main_32` / `main_64`, `bench_full_*`), or
- For `test_ntt_limits` only, compile `ntt_limits.cpp` twice into `ntt_limits_32.o` / `ntt_limits_64.o` with distinct output names.

The shared `build/*.o` cache cannot safely mix widths: an object built with `LIMB_BITS=32` would poison a subsequent `LIMB_BITS=64` link if reused. Unifying into one executable requires either width-suffixed object files for **all** pipeline sources or a template/facade split where width-neutral code links once.

#### Existing partial pattern: `test_ntt_limits`

`make/tests.mk` already dual-compiles `ntt_limits.cpp` into separate objects (`ntt_limits_32.o`, `ntt_limits_64.o`) and links each with a matching test driver. This works because `ntt_limits.cpp` has no CUDA kernels, no `__constant__` symbols, and no shared globals — it only reads `extern moduli`. The full pipeline cannot use this pattern without extending it to every `.cu` file and renaming all exported symbols.

#### Viable unification approaches (ordered by fit)

1. **Template pipeline + explicit instantiation + runtime facade** *(recommended)*
   - Refactor core logic into `PipelineTraits<32>` / `PipelineTraits<64>` (or a `LimbWidth` enum + partial specializations).
   - One `.cu` per width with `template struct Pipeline<32>;` / `Pipeline<64>` explicit instantiations and **distinct symbol prefixes** (`ntt32::execute`, `ntt64::execute`).
   - Thin host facade (`multiply_unified.cpp`) holds `std::variant` or two optional contexts and dispatches on a runtime flag.
   - GPU-NTT calls already support both `Data32` and `Data64`; no library change needed.

2. **Macro-prefixed dual compilation** *(minimal refactor, Makefile-heavy)*
   - Compile each pipeline `.cu` twice: `-DLIMB_BITS=32 -DPIPELINE_NS=ntt32` and `-DLIMB_BITS=64 -DPIPELINE_NS=ntt64`, wrapping exports in `PIPELINE_NS`.
   - Same idea for `__constant__` arrays (`d_primes` → `ntt32_d_primes` via macro).
   - Fastest path to one binary; leaves preprocessor forks in place and doubles compile time for GPU sources.

3. **`MAX_MODULI = 3` padding** *(partial, insufficient alone)*
   - Pad 64-bit RNS to 3 moduli with a dummy third prime to unify `CRTGarnerParams` and constant-memory sizes.
   - Does **not** solve `Data32` vs `Data64` NTT types, carry limb width, or kernel monomorphism.

4. **Two shared objects + `dlopen`** *(operational split)*
   - Build `libbirdwing_32.so` and `libbirdwing_64.so`; loader picks at runtime.
   - Avoids symbol collisions but does not produce a single static binary; doubles deployment surface.

#### Suggested implementation order

1. Introduce `PipelineTraits` / width tag without changing behaviour; keep separate binaries passing tests.
2. Split `gpu_ntt.cu`, `crt_gpu.cu`, `carry_prop.cu`, `zero_pad.cu` into width-specific TUs with explicit symbol prefixes.
3. Add unified `NTTContextUnified` (or dual optional contexts) and `execute_ntt_multiply(width, …)` facade.
4. Extend Makefile: `build/*_32.o`, `build/*_64.o`, single `main` / `bench_full` link target.
5. Add runtime flag to `main` and `gpu_full_multiply_benchmark.cpp`; update `run_gpu_bench.py` to invoke one binary with `--limb-bits both`.

**Estimated scope:** moderate refactor (~6–8 source files, Makefile rules, API header); not blocked by GPU-NTT or CUDA hardware limits.

### Suggested next steps

1. **Productionize pointwise mul** — Barrett or library modular multiply for 32/64-bit limbs
2. **Reconnect `main.cpp`** — enable `merge` path and optional file I/O for manual testing
3. **Parameter tooling** — promote `scripts/find_prim_roots.py` output into generated headers for new prime sets
4. **64-bit scaling** — use `bench_full_64` to profile larger `N` and limb counts; compare CRT/carry vs NTT (`TIMING=1`)
5. **CI / reproducibility** — document exact GPU-NTT and GMP install steps; pin moduli in a single config source
6. **Unified dual-width binary** — follow the approach in [Dual-width unified executable](#dual-width-unified-executable--investigation) above

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
