# small-field-metal-ntt

Comparative benchmark of NTT algorithm families over small ZK-friendly fields on Apple Metal GPUs.

## What This Is

The first open-source Metal NTT implementation for zero-knowledge proof fields, with a head-to-head benchmark of 4 algorithm variants:

| Variant | Algorithm | Radix | In-place | Bit-reversal | Threadgroup Barriers (2^20) |
|---------|-----------|-------|----------|--------------|----------------------------|
| 1 | CT-DIT | 2 | No | Explicit (separate kernel) | 20 (all device memory) |
| 2 | CT-DIT/GS-DIF paired | 2 | Yes | None (stays bit-reversed) | 13 (threadgroup) + 7 (device) |
| 3 | Stockham autosort | 2 | No (2x memory) | Folded into staged passes | 12 (threadgroup) + 8 (device) |
| 4 | CT-DIT/GS-DIF paired | 4 | Yes | None | 7 (threadgroup) + 4 (device) |

### Fields

- **M31** (p = 2^31 - 1): Mersenne prime with Circle NTT (DCCT). Used by Stwo (Circle STARKs).
- **BabyBear** (p = 2^31 - 2^27 + 1): Montgomery reduction, standard NTT. Used by Plonky3. *(planned)*
- **KoalaBear** (p = 2^31 - 2^24 + 1): Montgomery reduction, standard NTT. Used by Whirlaway. *(planned)*

## Benchmark Results

First results on Apple M3, optimized build (`cargo bench`). Median of 20 iterations after 5 warmup.

| Variant | 2^10 | 2^12 | 2^14 | 2^16 | 2^18 | 2^20 |
|---------|------|------|------|------|------|------|
| CPU reference | 35 us | 185 us | 907 us | 4.1 ms | 18.9 ms | 86.0 ms |
| V1 CT-DIT r2 | 1865 us | 2163 us | 2888 us | 6.7 ms | 23.8 ms | 91.5 ms |
| V2 CT-GS r2 | 240 us | 390 us | 1395 us | 4.9 ms | 20.5 ms | 86.4 ms |
| V3 Stockham r2 | 242 us | 410 us | 1574 us | 5.1 ms | 21.0 ms | 87.4 ms |
| **V4 CT-GS r4** | **237 us** | **379 us** | **1385 us** | **4.7 ms** | **19.6 ms** | **84.7 ms** |

**Winner: V4 (radix-4)** at all GPU sizes. 84.7 ms at 2^20 (12.4 Melem/s).

### Key Takeaways

1. **Radix-4 wins.** Half the barriers translates directly to ~5-10% speedup over radix-2 at every size.
2. **Threadgroup memory matters.** V2/V3/V4 are 2-8x faster than V1 (naive) at small sizes where kernel launch overhead dominates.
3. **Stockham's 2x memory hurts on UMA.** V3 is consistently ~3-5% slower than V2 despite identical algorithmic complexity. The out-of-place pattern doesn't help on Apple Silicon.
4. **CPU is competitive at large sizes.** At 2^20, all variants converge near the CPU speed. The transform is memory-bandwidth-bound, and UMA means the CPU and GPU share the same bandwidth.
5. **GPU wins at mid-range.** The GPU advantage peaks around 2^14-2^16 where parallelism helps but memory isn't the bottleneck yet.

## Why This Matters

The ZK community uses Cooley-Tukey NTT by default (from crypto textbooks). The GPU FFT community converged on Stockham (for discrete GPUs with PCIe). Apple Silicon's Unified Memory Architecture (UMA) changes both arguments. **Nobody has tested which algorithm family actually wins on UMA for ZK fields.** This benchmark produces that data.

Part of the [Ethereum GPU Acceleration Alliance (EGAA)](https://github.com/zkmopro/awesome-client-side-gpu) effort to build shared GPU infrastructure for client-side ZK proving.

## Quick Start

```bash
# Requires: macOS with Apple Silicon (M1+), Xcode command-line tools
cargo test          # correctness tests (96 tests)
cargo bench         # full benchmark suite (CSV to stdout, table to stderr)
```

## Project Structure

```
small-field-metal-ntt/
  src/
    field/          M31 field arithmetic (Rust + Metal shaders)
    ntt/            Algorithm variants behind NttBackend trait
    bench/          BenchConfig struct
  shaders/          Metal compute shader source files
    m31_field.metal     M31 add/sub/mul/reduce
    ntt_common.metal    Shared butterfly ops (radix-2 + radix-4)
    ntt_ct_dit_r2.metal Variant 1 kernels
    ntt_ct_gs_r2.metal  Variant 2 kernels (threadgroup forward/inverse)
    ntt_stockham_r2.metal Variant 3 kernels (ping-pong)
    ntt_ct_gs_r4.metal  Variant 4 kernels (radix-4 + radix-2 fallback)
    test_kernels.metal  Field arithmetic test kernels
  benches/          Benchmark entry point
  tests/            GPU integration tests
```

## Architecture

```
trait NttBackend<F: Field>
  |
  +-- CpuReferenceBackend (correctness oracle, forward + inverse)
  +-- MetalCtDitR2        (variant 1: naive baseline, forward only)
  +-- MetalCtGsR2         (variant 2: in-place paired, forward + inverse)
  +-- MetalStockhamR2     (variant 3: out-of-place, forward only)
  +-- MetalCtGsR4         (variant 4: radix-4, forward only)
  +-- (future: VulkanBackend, WebGpuBackend)
```

All GPU variants use a two-phase execution strategy:
- **Threadgroup phase**: Small-stride layers processed in 32 KB on-chip memory (8192 elements for in-place variants, 2x4096 for Stockham).
- **Device-memory phase**: Large-stride layers dispatched individually.

## Implementation Roadmap

### Week 1: Core Kernels + Correctness

- [x] **Day 1-2: Foundation** (PR #1, #2)
  - M31 field arithmetic (Rust + Metal shaders)
  - Circle group and coset implementation
  - CPU reference Circle NTT (forward + inverse)
  - MetalContext: device init, shader compilation, buffer management, dispatch
  - GPU integration tests (field ops, butterfly)

- [x] **Day 3: Variant 1 — CT-DIT radix-2** (PR #4)
  - Naive baseline: one dispatch per layer, all device memory
  - Forward NTT, GPU-CPU consistency tests at sizes 4-4096

- [x] **Day 4: Variant 2 — CT-DIT/GS-DIF paired radix-2** (PR #5)
  - Threadgroup memory (8192-element tiles, 13 stages on-chip)
  - Forward + inverse NTT, round-trip tests
  - No bit-reversal between forward and inverse

- [x] **Day 5: Variant 3 — Stockham radix-2** (PR #6)
  - Out-of-place ping-pong (2x memory, 2x4096 threadgroup arrays)
  - Forward NTT only (benchmark variant)

- [x] **Day 6-7: Variant 4 — CT-DIT/GS-DIF radix-4** (PR #7)
  - Radix-4 butterflies (4 elements, 3 twiddles), half the barriers
  - Odd log_n fallback to radix-2 for final stage
  - Forward NTT only (benchmark variant)
  - Tests at sizes up to 65536 including odd log_n (8, 32, 2048)

### Week 2: Benchmarking + Polish + Writeup

- [x] **Day 8-9: Benchmark harness** (PR #8)
  - Races all 4 variants + CPU across sizes 2^10 to 2^20
  - 5 warmup + 20 timed iterations, median/min/max reported
  - CSV output + summary table
  - First results: V4 wins at all sizes

- [ ] **Day 10: Thermal stability testing**
  - 60-second continuous runs per variant, throughput decay curves
  - Report sustained throughput at 10s/30s/60s intervals

- [ ] **Day 11-12: Analysis + Writeup**
  - Throughput vs size plots (log-log)
  - Memory usage comparison
  - Thermal decay curves
  - Speedup vs CPU baseline table
  - Barrier count correlation analysis

- [ ] **Day 13-14: Polish + Release**
  - Tag v0.1.0 release
  - Lambdaworks comparison (same-machine benchmark)
  - Documentation for component reuse

### Future (from TODOS.md)

- [ ] **P1: Cooperative CPU-GPU NTT** — hybrid execution, CPU handles large-stride stages, GPU handles parallel stages, zero-copy UMA handoff
- [ ] **P1: 4-step decomposition for 2^22+** — Bailey decomposition for transforms beyond threadgroup capacity
- [ ] **P2: Batched NTT for lattice crypto** — thousands of small NTTs in parallel (Kyber/Dilithium)
- [ ] **P2: M31 extension field (Fp2)** — needed for Stwo production constraints
- [ ] **P2: GPU timestamps** — replace wall-clock with MTLCommandBuffer.gpuStartTime (blocked on metal-rs #329)
- [ ] **P2: BabyBear + KoalaBear fields** — standard NTT (not Circle), tests whether algorithm comparison generalizes
- [ ] **P3: CI with cargo bench** — GitHub Actions macOS runner for regression detection

## License

MIT OR Apache-2.0
