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

Apple M3, optimized build. Median of 20 iterations, 5 warmup.

| Variant | 2^10 | 2^14 | 2^16 | 2^20 |
|---------|------|------|------|------|
| CPU ref | 35 us | 907 us | 4.1 ms | 86.0 ms |
| V1 CT-DIT r2 | 1865 us | 2888 us | 6.7 ms | 91.5 ms |
| V2 CT-GS r2 | 240 us | 1395 us | 4.9 ms | 86.4 ms |
| V3 Stockham | 242 us | 1574 us | 5.1 ms | 87.4 ms |
| **V4 CT-GS r4** | **237 us** | **1385 us** | **4.7 ms** | **84.7 ms** |

**Winner: V4 (radix-4)** at all GPU sizes. 84.7 ms at 2^20 (12.4 Melem/s).

### Key Takeaways

1. **Radix-4 wins.** Half the barriers = ~5-10% speedup over radix-2 at every size.
2. **Threadgroup memory matters.** V2/V3/V4 are 2-8x faster than V1 at small sizes.
3. **Stockham's 2x memory hurts on UMA.** V3 is ~3-5% slower than V2. Out-of-place doesn't help on Apple Silicon.
4. **CPU is competitive at 2^20.** Memory-bandwidth-bound. UMA means CPU and GPU share the same bandwidth.
5. **GPU wins at mid-range.** Peak advantage at 2^14-2^16 where parallelism helps but memory isn't the bottleneck.
6. **M31's Mersenne reduction is nearly free on GPU.** `reduce(x) = (x >> 31) + (x & 0x7FFFFFFF)` is a shift and a mask, no multiply. This makes butterflies arithmetic-cheap, so barrier and memory overhead dominate. The algorithm ranking may shift for Montgomery-reduced fields (BabyBear, KoalaBear) where each butterfly costs a full 32x32 multiply for reduction, making the compute-to-memory ratio heavier.

## Why This Matters

The ZK community defaults to Cooley-Tukey NTT. The GPU FFT community converged on Stockham (for discrete GPUs with PCIe). Apple Silicon UMA changes both assumptions. **Nobody has tested which algorithm family actually wins on UMA for ZK fields.** This benchmark produces that data.

Part of the [Ethereum GPU Acceleration Alliance (EGAA)](https://github.com/zkmopro/awesome-client-side-gpu) effort to build shared GPU infrastructure for client-side ZK proving.

## Quick Start

```bash
# Requires: macOS with Apple Silicon (M1+), Xcode command-line tools
cargo test          # correctness tests (96 tests)
cargo bench         # full benchmark (CSV to stdout, table to stderr)
```

## Architecture

```
trait NttBackend<F: Field>
  |
  +-- CpuReferenceBackend (forward + inverse)
  +-- MetalCtDitR2        (V1: naive baseline)
  +-- MetalCtGsR2         (V2: in-place, forward + inverse)
  +-- MetalStockhamR2     (V3: out-of-place, forward only)
  +-- MetalCtGsR4         (V4: radix-4, forward only)
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
- [x] Variant 3: Stockham radix-2, out-of-place ping-pong (PR #6)
- [x] Variant 4: CT-DIT/GS-DIF radix-4, half barriers, odd-log_n fallback (PR #7)
- [x] Benchmark harness: all variants head-to-head, CSV + summary (PR #8)
- [x] BabyBear field: Montgomery reduction, standard NTT twiddles (PR #9)

### Next

- [ ] Thermal stability testing (60s sustained throughput decay)
- [ ] Analysis writeup with plots (throughput vs size, thermal curves)
- [ ] Lambdaworks same-machine comparison
- [ ] v0.1.0 release

### Future

- [ ] Modular reduction strategy comparison (Mersenne vs Montgomery vs Barrett on GPU — which maps best to Metal ALUs and SIMD width?)
- [ ] Cooperative CPU-GPU NTT on UMA (zero-copy handoff at stage boundary)
- [ ] 4-step decomposition for 2^22+ transforms
- [ ] Batched small NTTs for lattice crypto (Kyber/Dilithium)
- [ ] M31 extension field (Fp2) for Stwo production constraints
- [ ] GPU timestamps via MTLCommandBuffer (blocked on metal-rs #329)
- [ ] KoalaBear field (standard NTT, not Circle)
- [ ] CI with cargo bench on GitHub Actions macOS runner

## License

MIT OR Apache-2.0
