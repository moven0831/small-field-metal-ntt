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
| CPU ref | 5 us | 23 us | 95 us | 356 us | 1.5 ms | 6.4 ms |
| V1 CT-DIT r2 | 412 us | 320 us | 395 us | 558 us | 1.4 ms | 3.2 ms |
| V2 CT-GS r2 | 357 us | 260 us | 341 us | 420 us | 1.1 ms | 2.7 ms |
| V3 Stockham | 224 us | 273 us | 334 us | 356 us | 1.2 ms | 4.1 ms |
| **V4 CT-GS r4** | **216 us** | **253 us** | **309 us** | **392 us** | **880 us** | **2.5 ms** |

**Winner: V4 (radix-4)** at all GPU sizes. 2.5 ms at 2^20 (420 Melem/s). GPU is **2.6x faster** than CPU at 2^20.

### BabyBear NTT (Montgomery, Standard NTT)

Apple M3, optimized build. Forward NTT over BabyBear with Montgomery reduction.

| Variant | 2^10 | 2^12 | 2^14 | 2^16 | 2^18 | 2^20 |
|---------|------|------|------|------|------|------|
| CPU ref | 6 us | 28 us | 125 us | 622 us | 2.7 ms | 12.2 ms |
| V1 CT-DIT r2 | 392 us | 309 us | 386 us | 550 us | 1.4 ms | 3.2 ms |
| V2 CT-GS r2 | 378 us | 306 us | 351 us | 432 us | 1.2 ms | 2.9 ms |
| V3 Stockham | 229 us | 278 us | 340 us | 421 us | 1.3 ms | 3.2 ms |
| **V4 CT-GS r4** | 364 us | **254 us** | **336 us** | **347 us** | **1.0 ms** | 2.9 ms |

**V4 radix-4 wins at 2^16-2^18**, V2 and V4 are neck-and-neck at 2^20. Montgomery multiply cost reduces radix-4's advantage compared to M31's near-free Mersenne reduction. GPU is **4.2x faster** than CPU at 2^20.

### Batched Coset LDE (BabyBear)

Full coset Low-Degree Extension pipeline: iDFT_batch -> zero_pad -> coset_shift -> forward_DFT_batch. Matches [zk-autoresearch](https://github.com/Barnadrot/zk-autoresearch) parameters for direct comparison.

| Config | Metal GPU (M3) | zk-autoresearch (EPYC AVX512) | Speedup |
|--------|---------------|-------------------------------|---------|
| 2^20 x 256, 2x LDE | **1413 ms** | ~2699 ms | **1.9x** |
| 2^20 x 256, 4x LDE | 2689 ms | — | — |

Run with `cargo bench --bench ntt_benchmark -- --coset-lde`.

### Cooperative CPU-GPU Split-Point Sweep

The key finding: on UMA, splitting NTT layers between CPU and GPU beats both pure approaches.

CPU does the first S layers (high stride, cache-friendly), GPU does the rest (high parallelism). Run with `cargo bench --bench ntt_benchmark -- --cooperative`.

| Size | Optimal split | CPU layers | GPU layers | Total | vs all-GPU | vs all-CPU |
|------|--------------|------------|------------|-------|------------|------------|
| 2^10 | 10 (all CPU) | 10 | 0 | 5 us | 60x faster | same |
| 2^14 | 14 (all CPU) | 14 | 0 | 87 us | 7x faster | same |
| 2^16 | 16 (all CPU) | 16 | 0 | 388 us | 3x faster | same |
| **2^18** | **6** | **6** | **12** | **1.2 ms** | **1.9x faster** | **1.4x faster** |
| **2^20** | **7** | **7** | **13** | **3.8 ms** | **1.5x faster** | **1.9x faster** |

At prover-relevant sizes (2^18+), the U-curve appears: neither all-CPU nor all-GPU is optimal. The cooperative split at 2^20 is **34% faster than all-GPU** and **47% faster than all-CPU**.

### Key Takeaways

1. **UMA changes the game.** On Apple Silicon, splitting NTT between CPU and GPU outperforms both pure approaches at prover-relevant sizes (2^18+). This is the first published evidence.
2. **GPU wins at 2^18+.** V4 (radix-4) is 2.6x faster than CPU at 2^20. The crossover is around 2^18 where GPU dispatch overhead is amortized.
3. **Radix-4 wins** the GPU-only shootout. Half the barriers = ~8% speedup over radix-2 at 2^20.
4. **CPU dominates at small sizes.** GPU dispatch overhead (~200us) makes pure GPU slower than CPU below 2^16.
5. **The split point is size-dependent.** At 2^18, CPU does 6 layers. At 2^20, CPU does 7. The crossover shifts as memory pressure increases.
6. **Montgomery fields shift the ranking.** BabyBear's Montgomery multiply costs ~3x M31's Mersenne reduction on GPU. This narrows radix-4's advantage (from 8% to ~2% at 2^20) and makes V2/V4 nearly tied at large sizes.
7. **Stockham's 2x memory hurts on UMA.** V3 is ~51% slower than V2 at 2^20. Out-of-place + 2x memory pressure doesn't help.
8. **Twiddle generation dominates naive benchmarks.** Without caching, twiddle gen (79ms at 2^20) dwarfs both CPU butterfly (6.4ms) and GPU dispatch (2.5ms) time. Previous results were artifacts of twiddle overhead.

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
  +-- BbMetalCtGsR4         (V4: radix-4, forward + inverse)
  |
  LDE Pipeline:
  +-- CosetLdeBatch         (iDFT -> zero_pad -> coset_shift -> DFT)
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
- [x] zk-autoresearch comparison: 1.9x faster than EPYC AVX512 at 2^20 x 256 (PR #19)

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
- [ ] CI with cargo bench on GitHub Actions macOS runner

## License

MIT OR Apache-2.0
