# GPU multiply timing model

How wall-clock time is spent in the birdwing GPU multiply pipeline, how it maps
to code, and how benchmarks measure it.

See [ROADMAP.md](../ROADMAP.md) for architecture and module layout.

## The five buckets

| Bucket | When it runs | Depends on | Amortization |
|--------|--------------|------------|--------------|
| **init** | First CUDA use in a process | — | Once per process |
| **precompute** | Once per transform size `N` | `N`, `LIMB_BITS`, RNS primes | Once per `N`; **host-only, cacheable to disk** |
| **setup** | Before a batch of multiplies at a given size | `N`, `L_A`, `L_B` | **Once per size** — amortize over many multiplies at that size |
| **teardown** | After the batch (or on shutdown) | Same as setup | Paired with setup; skip until the size is done |
| **multiply** | Every `A × B` | Operand values | Never amortized |

Only **multiply** runs on every call. **precompute**, **setup**, and **teardown**
are fixed costs for a given size; divide by the number of multiplies at that
size to get per-multiply overhead.

### precompute vs setup

**precompute** is **host-only**: NTT parameters, twiddle tables in RAM, Garner
CRT coefficients. No `cudaMalloc` / H2D. Safe to save to disk and reload without
a GPU.

**setup** is **everything that touches the GPU** before the multiply loop:

- `upload_ntt_precomputed` — H2D twiddles, modulus / n⁻¹, Garner constants
- `allocate_ntt_context` — mutable operand / CRT / carry buffers
- pinned host buffer allocation for `A`, `B`, and the output `C`
- staging copy of operands from paged memory into pinned `A` / `B`

### Same-size batches

Keep `NTTPrecomputed` (host tables) and `NTTContext` (GPU buffers) alive across
many **multiply** calls at the same `(L_A, L_B)` / `N`. Call **teardown** only
when that size batch ends.

```
L_C = L_A + L_B - 1
N   = padded_ntt_size(L_A, L_B)
```

## Lifecycle

```mermaid
flowchart TD
    subgraph init_b [init]
        I[CUDA driver init]
    end

    subgraph pre_b [precompute — host only]
        P[precompute_ntt]
    end

    subgraph setup_b [setup]
        U[upload_ntt_precomputed]
        S[allocate_ntt_context + pinned A/B/C alloc]
        G[stage A/B: paged -> pinned]
        U --> S --> G
    end

    subgraph mult_b [multiply]
        M[execute_ntt_multiply — D2H lands in pinned C]
    end

    subgraph tear_b [teardown]
        X[unstage C: pinned -> paged]
        T[cleanup context / GPU pre / pinned]
        X --> T
    end

    I --> P
    P --> U
    G --> M
    M --> M
    M --> X
```

## init

First CUDA API call in a process can cost hundreds of milliseconds.

`bench_full_multiply_32` calls `warmup_cuda_runtime()` once before the `L`
sweep so **init** is not attributed to the first bar.

## precompute

**API:** `precompute_ntt(N, PrecomputeTiming* timing_out = nullptr)`  
**Produces:** `NTTPrecomputed` with host twiddle tables; `gpu_uploaded == false`.

| `PrecomputeTiming` field | Measures |
|--------------------------|----------|
| `factors_ms` | `generate_factors_for_N` |
| `params_ms` | `NTTParameters` construction per modulus |
| `twiddle_host_ms` | `gpu_root_of_unity_table_generator` (fwd + inv) |
| `garner_host_ms` | `compute_garner_params` |
| `total_ms` | Wall clock (host only) |

Host data lives in `forward_omega_host` / `inverse_omega_host` and is the
natural **disk-cache payload**. Re-upload via `upload_ntt_precomputed` after
load without re-running host generation.

## setup

**Upload API:** `upload_ntt_precomputed(pre, SetupUploadTiming* timing_out = nullptr)`  
Must run before `allocate_ntt_context`. Idempotent if already uploaded.

| `SetupUploadTiming` field | Measures |
|---------------------------|----------|
| `twiddle_upload_ms` | Twiddle `cudaMalloc` + H2D + sync drain |
| `mod_constants_ms` | Modulus and n⁻¹ H2D |
| `garner_upload_ms` | `upload_garner_params` |
| `total_ms` | Wall clock for upload |

**Context API:** `allocate_ntt_context(pre, L_A, L_B)` — mutable GPU buffers
(requires `pre.gpu_uploaded`).

**Pinned buffers:** `cudaMallocHost` reserves page-locked host buffers for the
inputs `A` / `B` and the output `C`. The output buffer is what
`execute_ntt_multiply` copies into on D2H, so the transfer takes the DMA fast
path instead of the driver's pageable staging path.

**Staging:** operands are `memcpy`'d from paged memory into pinned `A` / `B`
once per batch, timed separately from allocation (`setup_stage_ms`).

**Benchmark columns:** `setup_upload_ms`, `setup_alloc_ms`, `setup_pinned_ms`
(allocation only), `setup_stage_ms` (paged → pinned copy), `upload_*`
breakdown.

## teardown

- unstage `C` — `memcpy` result from pinned buffer to paged memory, timed
  separately (`teardown_unstage_ms`)
- `cleanup_ntt_context(ctx)` — frees mutable GPU buffers
- `cleanup_ntt_precomputed(pre)` — frees **GPU** twiddle / constant allocations;
  **host** tables remain so `upload_ntt_precomputed` can run again without full
  precompute
- `cudaFreeHost` — pinned A/B/C

Defer **teardown** until the size batch is finished. You may free context only
and keep host precompute for another upload cycle at the same `N`.

## multiply

**API:** `execute_ntt_multiply(..., NTTTiming* timing_out = nullptr)`

| `NTTTiming` field | Stackable? |
|-------------------|------------|
| `ingress_fwd_ms` | Yes (wall-clock ingress phase) |
| `h2d_a_ms` | Yes (A transfer only; last clean checkpoint before stream overlap) |
| `pointwise_mul_ms` … `d2h_ms` | Yes |
| `h2d_ms`, `fwd_pad_ntt_*` | Diagnostics only (streams overlap) |

`ingress_fwd_ms` ≈ `h2d_a_ms` + overlapped fwd pad+NTT on the critical path. After
`h2d_stop_a`, stream A runs zero-pad+NTT while stream B may still be copying.

## Benchmarks and CSV

```bash
make bench_full_32
./build/bench_full_multiply_32 --warmup 2 --iters 20 --csv gpu_multiply_bench.csv
python scripts/plot_bench.py gpu_multiply_bench.csv
```

Plots use **operand bits** (`L × host_limb_bits`) on the x-axis by default so 32-bit, hybrid, and 64-bit pipelines compare at equal input size. Use `--x L` for raw limb count.

Per `L`: single-shot **precompute**, **upload**, **setup** alloc/pinned/stage,
averaged **multiply**, single-shot **teardown**.

| Columns | Bucket |
|---------|--------|
| `setup_precompute_ms`, `pre_*` | precompute (host) |
| `setup_upload_ms`, `upload_*` | setup (GPU upload) |
| `setup_alloc_ms`, `setup_pinned_ms`, `setup_stage_ms` | setup |
| `mean_ms`, stage columns | multiply |
| `teardown_unstage_ms`, `teardown_free_*` | teardown |

## Code map

| Bucket | API | Timing |
|--------|-----|--------|
| init | (implicit) | — |
| precompute | `precompute_ntt` | `PrecomputeTiming` |
| setup | `upload_ntt_precomputed`, `allocate_ntt_context`, `cudaMallocHost`, stage `memcpy` | `SetupUploadTiming`, `setup_*` |
| teardown | unstage `memcpy`, `cleanup_ntt_context`, `cleanup_ntt_precomputed`, `cudaFreeHost` | `teardown_*` |
| multiply | `execute_ntt_multiply` (D2H into pinned `C`) | `NTTTiming` |
| Legacy | `host_multiply_merge` | `duration` |

Correct sequence:

```cpp
NTTPrecomputed pre = precompute_ntt(N);
upload_ntt_precomputed(pre);
NTTContext ctx = allocate_ntt_context(pre, L_A, L_B);
// ... execute_ntt_multiply in a loop ...
cleanup_ntt_context(ctx);
cleanup_ntt_precomputed(pre);  // optional; frees GPU copy of pre
```
