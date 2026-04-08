# small-field-metal-ntt

Comparative benchmark of NTT algorithm families over small ZK-friendly fields on Apple Metal GPUs.

## What This Is

The first open-source Metal NTT implementation for zero-knowledge proof fields, with head-to-head benchmarks of 4 algorithm variants across 2 fields:

| # | Algorithm | Radix | In-place | Barriers (2^20) |
|---|-----------|-------|----------|-----------------|
| 1 | CT-DIT | 2 | No | 20 (all device) |
| 2 | CT-DIT/GS-DIF | 2 | Yes | 13 + 7 |
| 3 | Stockham | 2 | No (2x mem) | 12 + 8 |
| 4 | CT-DIT/GS-DIF | 4 | Yes | 7 + 4 |

### Fields

- **M31** (p = 2^31 - 1): Circle NTT (DCCT). Used by Stwo. All 4 variants implemented.
- **BabyBear** (p = 2^31 - 2^27 + 1): Montgomery reduction, standard NTT. Used by Plonky3. All 4 variants + batched coset LDE implemented.
- **KoalaBear** (p = 2^31 - 2^24 + 1): standard NTT. Used by Whirlaway. *(planned)*

## Benchmark Results

Apple M3, optimized build. Median of 20 iterations, 5 warmup. Forward NTT over M31 (Circle NTT). Twiddle factors cached, single command buffer per NTT call.

| Variant | 2^10 | 2^12 | 2^14 | 2^16 | 2^18 | 2^20 |
|---------|------|------|------|------|------|------|
| CPU ref | 6 us | 24 us | 107 us | 379 us | 1.5 ms | 6.5 ms |
| V1 CT-DIT r2 | 474 us | 426 us | 462 us | 651 us | 1.5 ms | 3.9 ms |
| V2 CT-GS r2 | 294 us | 317 us | 401 us | 506 us | 1.1 ms | 2.98 ms |
| V3 Stockham | 441 us | 348 us | 413 us | 555 us | 1.3 ms | 3.3 ms |
| **V4 CT-GS r4** | **276 us** | **306 us** | **384 us** | **471 us** | **970 us** | **2.98 ms** |

**V2 and V4 tied at 2^20** (2981 vs 2983 us, ~352 Melem/s). V4 wins at 2^18 (970 us vs 1112 us). GPU is **2.2x faster** than CPU at 2^20.

### BabyBear NTT (Montgomery, Standard NTT)

Apple M3, optimized build. Forward NTT over BabyBear with Montgomery reduction.

| Variant | 2^10 | 2^12 | 2^14 | 2^16 | 2^18 | 2^20 |
|---------|------|------|------|------|------|------|
| CPU ref | 6 us | 28 us | 132 us | 588 us | 2.6 ms | 11.7 ms |
| V1 CT-DIT r2 | 472 us | 366 us | 474 us | 634 us | 1.5 ms | 4.3 ms |
| V2 CT-GS r2 | 271 us | 300 us | 397 us | 514 us | 1.3 ms | 3.0 ms |
| V3 Stockham | 433 us | 315 us | 389 us | 601 us | 1.3 ms | 3.4 ms |
| **V4 CT-GS r4** | **253 us** | **278 us** | **392 us** | **495 us** | **1.0 ms** | **2.7 ms** |

**V4 radix-4 wins at 2^20** (2721 us, 385 Melem/s). GPU is **4.3x faster** than CPU at 2^20.

### Batched Coset LDE (BabyBear)

Full coset Low-Degree Extension pipeline: iDFT_batch -> zero_pad -> coset_shift -> forward_DFT_batch. Same-machine comparison against [Plonky3](https://github.com/Plonky3/Plonky3) `coset_lde_batch` (Radix2DitParallel, `--features parallel`, all cores).

| Config | Metal GPU (M3) | Plonky3 CPU (M3, Rayon) | GPU vs CPU |
|--------|---------------|------------------------|------------|
| 2^16 x 256, 2x LDE | **31 ms** | 69 ms | **2.3x faster** |
| 2^18 x 256, 2x LDE | **146 ms** | 237 ms | **1.6x faster** |
| 2^20 x 256, 2x LDE | **821 ms** | 1177 ms | **1.4x faster** |
| 2^20 x 256, 4x LDE | 1661 ms | — | — |

GPU beats Plonky3 CPU at all sizes. Fused kernel (normalize + zero-pad + coset-shift → 1 dispatch) + radix-4 batched NTT (20 → 11 dispatches) yield **1.4x–2.3x** speedup over Plonky3 CPU.

Run with `cargo bench --bench ntt_benchmark -- --coset-lde`.

### Cooperative CPU-GPU Split-Point Sweep

The key finding: on UMA, splitting NTT layers between CPU and GPU beats both pure approaches.

CPU does the first S layers (high stride, cache-friendly), GPU does the rest (high parallelism). Run with `cargo bench --bench ntt_benchmark -- --cooperative`.

| Size | Optimal split | CPU layers | GPU layers | Total | vs all-GPU | vs all-CPU |
|------|--------------|------------|------------|-------|------------|------------|
| 2^10 | 10 (all CPU) | 10 | 0 | 5 us | ~80x faster | same |
| 2^14 | 14 (all CPU) | 14 | 0 | 158 us | ~3x faster | same |
| 2^16 | 16 (all CPU) | 16 | 0 | 427 us | ~3x faster | same |
| **2^18** | **5** | **5** | **13** | **1.3 ms** | **1.7x faster** | **1.5x faster** |
| **2^20** | **2** | **2** | **18** | **3.7 ms** | **1.1x faster** | **1.9x faster** |

At prover-relevant sizes (2^18+), the cooperative split beats both pure approaches. At 2^20, splitting at layer 2 is **7% faster than all-GPU** and **48% faster than all-CPU**.

### Key Takeaways

1. **GPU beats CPU for full LDE at all sizes.** Metal GPU is 1.4x–2.3x faster than Plonky3 CPU (Rayon) for the coset LDE pipeline.
2. **Kernel fusion matters more than raw compute.** Reducing 20 dispatches to 11 (fused normalize+zeropad+shift, batched radix-4) yields the largest pipeline speedup.
3. **UMA enables cooperative splitting.** On Apple Silicon, splitting NTT between CPU and GPU outperforms both pure approaches at 2^18+.
4. **GPU wins at 2^18+.** GPU is 2.2x faster than CPU at 2^20 for M31, 4.3x for BabyBear. The crossover is around 2^18 where dispatch overhead is amortized.
5. **V2 and V4 converge for M31.** At 2^20, radix-2 (V2) and radix-4 (V4) are tied. V4 wins at mid-sizes (2^18). M31's cheap Mersenne reduction doesn't penalize extra butterflies.
6. **V4 radix-4 wins clearly for BabyBear.** 2721 us vs 3005 us at 2^20 — radix-4's fewer dispatches matter more when Montgomery multiply is expensive.
7. **CPU dominates at small sizes.** GPU dispatch overhead (~250us) makes pure GPU slower than CPU below 2^16.
8. **Twiddle generation dominates naive benchmarks.** Without caching, twiddle gen dwarfs both CPU butterfly and GPU dispatch time.

## Why This Matters

The ZK community defaults to Cooley-Tukey NTT. The GPU FFT community converged on Stockham (for discrete GPUs with PCIe). Apple Silicon UMA changes both assumptions. **Nobody has tested which algorithm family actually wins on UMA for ZK fields, or whether splitting work between CPU and GPU is better than either alone.** This benchmark produces that data.

Part of the [Ethereum GPU Acceleration Alliance (EGAA)](https://github.com/zkmopro/awesome-client-side-gpu) effort to build shared GPU infrastructure for client-side ZK proving.

## Quick Start

```bash
# Requires: macOS with Apple Silicon (M1+), Xcode command-line tools
cargo test                                    # correctness tests (115 tests)
cargo bench                                   # M31 algorithm shootout (default)
cargo bench -- --bb-shootout                  # BabyBear algorithm shootout
cargo bench -- --coset-lde                    # Batched coset LDE benchmark
cargo bench -- --batch-ntt                    # Raw batched NTT throughput
cargo bench -- --cooperative                  # CPU-GPU split-point sweep
```

## Architecture

```
trait NttBackend<F: Field>
  |
  M31 (Circle NTT):
  +-- CpuReferenceBackend (forward + inverse)
  +-- MetalCtDitR2        (V1: naive baseline)
  +-- MetalCtGsR2         (V2: in-place, forward + inverse)
  +-- MetalStockhamR2     (V3: out-of-place, forward + inverse)
  +-- MetalCtGsR4         (V4: radix-4, forward + inverse)
  |
  BabyBear (Standard NTT, Montgomery):
  +-- BbCpuReferenceBackend (forward + inverse)
  +-- BbMetalCtDitR2        (V1: naive baseline, forward only)
  +-- BbMetalR2             (V2: in-place, forward + inverse + batch)
  +-- BbMetalStockhamR2     (V3: out-of-place, forward + inverse)
  +-- BbMetalCtGsR4         (V4: radix-4, forward + inverse + batch)
  |
  LDE Pipeline (radix-4 + fused kernels):
  +-- CosetLdeBatch         (iDFT -> fused[normalize+zero_pad+coset_shift] -> DFT)
```

All GPU variants use a two-phase strategy:
- **Threadgroup phase**: small-stride layers in 32 KB on-chip memory
- **Device-memory phase**: large-stride layers, one dispatch each

## Roadmap

### Core (complete)

- [x] M31 field arithmetic, Circle group, CPU reference NTT (PR #1, #2)
- [x] Metal GPU infrastructure: device, shaders, dispatch (PR #2)
- [x] Variant 1: CT-DIT radix-2, naive baseline (PR #4)
- [x] Variant 2: CT-DIT/GS-DIF radix-2, threadgroup memory, forward+inverse (PR #5)
- [x] Variant 3: Stockham radix-2, out-of-place ping-pong, forward+inverse (PR #6, #11)
- [x] Variant 4: CT-DIT/GS-DIF radix-4, half barriers, forward+inverse (PR #7, #11)
- [x] Benchmark harness: all variants head-to-head, CSV + summary (PR #8)
- [x] BabyBear field: Montgomery reduction, standard NTT twiddles (PR #9)
- [x] UMA cache coherency verified: CPU-GPU shared buffer works without flush (PR #10)
- [x] Cooperative CPU-GPU NTT: split-point sweep showing U-curve at 2^18+ (PR #12)

- [x] BabyBear GPU NTT: all 4 variants ported (V1-V4) with position-indexed twiddles (PR #19)
- [x] Batched coset LDE: iDFT -> zero_pad -> coset_shift -> DFT on Metal GPU (PR #19)
- [x] Optimized LDE: fused kernel + batched radix-4, GPU 1.4x faster than Plonky3 CPU at 2^20 (PR #19)

### Next

- [ ] Thermal stability testing (60s sustained throughput decay)
- [ ] Analysis writeup with plots (throughput vs size, thermal curves)
- [ ] Lambdaworks same-machine comparison
- [ ] v0.1.0 release

### Future

- [x] Modular reduction strategy comparison: M31 Mersenne vs BabyBear Montgomery on GPU (PR #19)
- [x] Cooperative CPU-GPU NTT on UMA with split-point sweep (PR #12)
- [ ] 4-step decomposition for 2^22+ transforms
- [ ] Batched small NTTs for lattice crypto (Kyber/Dilithium)
- [ ] M31 extension field (Fp2) for Stwo production constraints
- [ ] GPU timestamps via MTLCommandBuffer (blocked on metal-rs #329)
- [ ] KoalaBear field (standard NTT, not Circle)
- [x] CI with cargo bench on GitHub Actions macOS runner (PR #23)

## License

MIT OR Apache-2.0
