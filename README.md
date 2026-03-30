# small-field-metal-ntt

Comparative benchmark of NTT algorithm families over small ZK-friendly fields on Apple Metal GPUs.

## What This Is

The first open-source Metal NTT implementation for zero-knowledge proof fields, with a head-to-head benchmark of 4 algorithm variants:

| Variant | Algorithm | Radix | In-place | Bit-reversal |
|---------|-----------|-------|----------|--------------|
| 1 | CT-DIT | 2 | No | Explicit (separate kernel) |
| 2 | CT-DIT/GS-DIF paired | 2 | Yes | None (stays bit-reversed between fwd/inv) |
| 3 | Stockham autosort | 2 | No (2x memory) | Folded into staged passes |
| 4 | CT-DIT/GS-DIF paired | 4 | Yes | None |

### Fields

- **M31** (p = 2^31 - 1): Mersenne prime with Circle NTT (DCCT). Used by Stwo (Circle STARKs).
- **BabyBear** (p = 2^31 - 2^27 + 1): Montgomery reduction, standard NTT. Used by Plonky3. *(stretch)*
- **KoalaBear** (p = 2^31 - 2^24 + 1): Montgomery reduction, standard NTT. Used by Whirlaway. *(stretch)*

## Why This Matters

The ZK community uses Cooley-Tukey NTT by default (from crypto textbooks). The GPU FFT community converged on Stockham (for discrete GPUs with PCIe). Apple Silicon's Unified Memory Architecture (UMA) changes both arguments. **Nobody has tested which algorithm family actually wins on UMA for ZK fields.** This benchmark produces that data.

Part of the [Ethereum GPU Acceleration Alliance (EGAA)](https://github.com/zkmopro/awesome-client-side-gpu) effort to build shared GPU infrastructure for client-side ZK proving.

## Quick Start

```bash
# Requires: macOS with Apple Silicon (M1+), Xcode command-line tools
cargo test          # correctness tests
cargo bench         # full benchmark suite
```

## Project Structure

```
small-field-metal-ntt/
  src/
    field/          M31, BabyBear, KoalaBear arithmetic (Rust + Metal)
    ntt/            Algorithm variants behind NttBackend trait
    bench/          Benchmark harness (timing, profiling, thermal, HTML report)
  shaders/          Metal compute shader source files
  benches/          Criterion/custom benchmark entry point
```

## Architecture

```
trait NttBackend<F: Field>
  |
  +-- CpuReferenceBackend (correctness oracle)
  +-- MetalBackend
  |     +-- ct_dit_r2      (variant 1: naive baseline)
  |     +-- ct_gs_r2       (variant 2: in-place paired)
  |     +-- stockham_r2    (variant 3: out-of-place autosort)
  |     +-- ct_gs_r4       (variant 4: radix-4, half barriers)
  +-- (future: VulkanBackend, WebGpuBackend)
```

## License

MIT OR Apache-2.0
