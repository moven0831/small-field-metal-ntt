# TODOS — Metal NTT Algorithm Shootout

Deferred from /office-hours and /plan-ceo-review on 2026-03-30.

## P1 — High Priority (next sprint)

### Cooperative CPU-GPU NTT on UMA
- **What:** Prototype hybrid execution where CPU handles first 2-3 NTT stages (large stride, cache-friendly) and GPU handles remaining high-parallelism stages, with zero-copy handoff via shared UMA buffer.
- **Why:** This is the most novel claim in the UMA research docs. Nobody has tested it. Could show that the CPU-GPU boundary is NOT at "all GPU" or "all CPU" but at a specific stage threshold.
- **Effort:** M (human: ~1 week / CC: ~2-3 hours)
- **Depends on:** Core benchmark crate (this sprint)

### 4-Step Decomposition for 2^22+ Transforms
- **What:** Implement the 4-step NTT decomposition (Bailey) for transforms exceeding threadgroup memory. Decompose N = N1 x N2, sub-NTTs fit in threadgroup, transpose via device memory.
- **Why:** Real ZK provers need 2^22-2^28 transforms. The core benchmark caps at 2^20 without this.
- **Effort:** M (human: ~1 week / CC: ~2-3 hours)
- **Depends on:** Core benchmark crate (this sprint)

## P2 — Medium Priority (month 2)

### Batched NTT Benchmark for Lattice Crypto
- **What:** Benchmark thousands of small NTTs (n=256-1024) in parallel, matching Kyber/Dilithium use case. One NTT per threadgroup, massive batching across threadgroups.
- **Why:** EGAA covers both STARK-scale and lattice-crypto NTT. Different bottleneck (kernel launch overhead, not memory bandwidth).
- **Effort:** S-M (human: ~4 days / CC: ~1-2 hours)
- **Depends on:** Core benchmark crate

### M31 Quadratic Extension Field Support
- **What:** Add M31 extension field (Fp2) arithmetic in Metal shaders. Needed for Stwo production constraints.
- **Why:** Stwo uses extension fields for certain constraint evaluations. The benchmark should eventually cover this.
- **Effort:** S (human: ~2 days / CC: ~1 hour)
- **Depends on:** Core M31 arithmetic (this sprint)

### GPU Timestamps for Precise Kernel Timing
- **What:** Replace wall-clock timing in `MetalContext::dispatch_and_wait()` with native `MTLCommandBuffer.gpuStartTime/gpuEndTime`. Currently blocked by metal-rs crate not exposing these methods (through v0.33). PR [gfx-rs/metal-rs#329](https://github.com/gfx-rs/metal-rs/pull/329) adds them but is not merged.
- **Why:** Wall-clock includes command submission overhead (~10-50us). For micro-benchmarking individual NTT stages, GPU timestamps give ~nanosecond precision.
- **Workaround:** Wall-clock is fine for algorithm comparison (overhead is constant across variants). Metal System Trace profiling (CEO expansion #1) provides per-kernel GPU timing independently.
- **Effort:** S (human: ~2 hours / CC: ~15 min) once metal-rs#329 merges, or use raw objc FFI now
- **Depends on:** metal-rs#329 merge or objc FFI wrapper

## P3 — Low Priority (when needed)

### CI with cargo bench on GitHub Actions macOS Runner
- **What:** Set up GitHub Actions workflow with macOS runner that runs `cargo bench --release` on every PR.
- **Why:** Prevents performance regressions, enables reproducible results.
- **Effort:** S (human: ~4 hours / CC: ~30 min)
- **Depends on:** GitHub repo exists with benchmark harness
