# small-field-metal-ntt

Comparative benchmark of NTT algorithm families over small ZK-friendly fields on Apple Metal GPUs.

## What This Is

The first open-source Metal NTT implementation for zero-knowledge proof fields, with a head-to-head benchmark of 4 algorithm variants:

| # | Algorithm | Radix | In-place | Barriers (2^20) |
|---|-----------|-------|----------|-----------------|
| 1 | CT-DIT | 2 | No | 20 (all device) |
| 2 | CT-DIT/GS-DIF | 2 | Yes | 13 + 7 |
| 3 | Stockham | 2 | No (2x mem) | 12 + 8 |
| 4 | CT-DIT/GS-DIF | 4 | Yes | 7 + 4 |

### Fields

- **M31** (p = 2^31 - 1): Circle NTT (DCCT). Used by Stwo.
- **BabyBear** (p = 2^31 - 2^27 + 1): Montgomery reduction, standard NTT. Used by Plonky3.
- **KoalaBear** (p = 2^31 - 2^24 + 1): standard NTT. Used by Whirlaway. *(planned)*

## Benchmark Results

Apple M3, optimized build. Median of 20 iterations, 5 warmup. Forward NTT over M31 (Circle NTT).

| Variant | 2^10 | 2^12 | 2^14 | 2^16 | 2^18 | 2^20 |
|---------|------|------|------|------|------|------|
| CPU ref | 35 us | 176 us | 879 us | 4.1 ms | 19.1 ms | 86.2 ms |
| V1 CT-DIT r2 | 1916 us | 2354 us | 3743 us | 7.1 ms | 24.1 ms | 91.6 ms |
| V2 CT-GS r2 | 240 us | 400 us | 1525 us | 5.3 ms | 20.1 ms | 87.8 ms |
| V3 Stockham | 255 us | 415 us | 1676 us | 5.8 ms | 20.8 ms | 87.2 ms |
| **V4 CT-GS r4** | **234 us** | **383 us** | **1420 us** | **5.4 ms** | **19.7 ms** | **85.2 ms** |

**Winner: V4 (radix-4)** at all GPU sizes. 85.2 ms at 2^20 (12.3 Melem/s).

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
2. **Radix-4 wins** the GPU-only shootout. Half the barriers = ~5-10% speedup over radix-2.
3. **CPU dominates at small sizes.** GPU dispatch overhead (~200us) makes pure GPU slower than CPU below 2^16.
4. **The split point is size-dependent.** At 2^18, CPU does 6 layers. At 2^20, CPU does 7. The crossover shifts as memory pressure increases.
5. **M31's Mersenne reduction is nearly free on GPU.** The algorithm ranking may shift for Montgomery fields (BabyBear) where each butterfly costs a full 32x32 multiply.
6. **Stockham's 2x memory hurts on UMA.** V3 is ~3-5% slower than V2. Out-of-place doesn't help.

## Why This Matters

The ZK community defaults to Cooley-Tukey NTT. The GPU FFT community converged on Stockham (for discrete GPUs with PCIe). Apple Silicon UMA changes both assumptions. **Nobody has tested which algorithm family actually wins on UMA for ZK fields, or whether splitting work between CPU and GPU is better than either alone.** This benchmark produces that data.

Part of the [Ethereum GPU Acceleration Alliance (EGAA)](https://github.com/zkmopro/awesome-client-side-gpu) effort to build shared GPU infrastructure for client-side ZK proving.

## Quick Start

```bash
# Requires: macOS with Apple Silicon (M1+), Xcode command-line tools
cargo test          # correctness tests (132 tests)
cargo bench         # full benchmark (CSV to stdout, table to stderr)
```

## Architecture

```
trait NttBackend<F: Field>
  |
  +-- CpuReferenceBackend (forward + inverse)
  +-- MetalCtDitR2        (V1: naive baseline)
  +-- MetalCtGsR2         (V2: in-place, forward + inverse)
  +-- MetalStockhamR2     (V3: out-of-place, forward + inverse)
  +-- MetalCtGsR4         (V4: radix-4, forward + inverse)
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

### Next

- [ ] Thermal stability testing (60s sustained throughput decay)
- [ ] Analysis writeup with plots (throughput vs size, thermal curves)
- [ ] Lambdaworks same-machine comparison
- [ ] v0.1.0 release

### Future

- [ ] Modular reduction strategy comparison (Mersenne vs Montgomery vs Barrett on GPU — which maps best to Metal ALUs and SIMD width?)
- [x] Cooperative CPU-GPU NTT on UMA with split-point sweep (PR #12)
- [ ] 4-step decomposition for 2^22+ transforms
- [ ] Batched small NTTs for lattice crypto (Kyber/Dilithium)
- [ ] M31 extension field (Fp2) for Stwo production constraints
- [ ] GPU timestamps via MTLCommandBuffer (blocked on metal-rs #329)
- [ ] KoalaBear field (standard NTT, not Circle)
- [ ] CI with cargo bench on GitHub Actions macOS runner

## License

MIT OR Apache-2.0
