//! Cooperative (split-point) CPU-GPU NTT over M31 Circle NTT.
//!
//! The core research artifact: splits NTT layers between CPU and GPU to find
//! the optimal partition point on Apple Silicon UMA.
//!
//! ```text
//! split_layer = S for a 2^k NTT:
//!
//!   CPU phase (layers 0..S):          GPU phase (layers S..k):
//!   ┌─────────────────────┐           ┌─────────────────────┐
//!   │ Forward: layers      │           │ Forward: layers      │
//!   │ (k-1) down to (k-S) │           │ (k-S-1) down to 0   │
//!   │ on shared MTLBuffer  │           │ via Metal dispatch   │
//!   │ using CPU butterflies│           │ on same buffer       │
//!   └─────────────────────┘           └─────────────────────┘
//!        ↓ sequential handoff (waitUntilCompleted) ↓
//!
//! split_layer = 0  → all GPU (no CPU phase)
//! split_layer = k  → all CPU (no GPU phase)
//! ```
//!
//! The forward NTT processes layers from (log_n-1) down to 0.
//! CPU takes the first `split_layer` layers (highest stride, from the top).
//! GPU takes the remaining layers (lower stride).
//!
//! This is a standalone function, not part of the NttBackend trait.

use crate::field::circle::Coset;
use crate::field::m31::M31;
use crate::field::Field;
use crate::gpu::MetalContext;
use crate::ntt::NttError;
use crate::ntt::twiddles::generate_twiddles;
use metal::*;
use std::path::Path;
use std::time::Instant;

/// Result of a cooperative NTT benchmark run.
pub struct CooperativeResult {
    /// The NTT output data.
    pub data: Vec<u32>,
    /// Wall-clock time for the CPU phase (microseconds).
    pub cpu_time_us: f64,
    /// Wall-clock time for the GPU phase (microseconds).
    pub gpu_time_us: f64,
    /// Total wall-clock time including handoff (microseconds).
    pub total_time_us: f64,
}

/// Pipelines needed for the GPU phase of cooperative NTT.
pub struct CooperativeNttContext {
    pub ctx: MetalContext,
    device_pipeline: ComputePipelineState,
    tg_pipeline: ComputePipelineState,
}

const MAX_TILE_LOG: usize = 13;

impl CooperativeNttContext {
    pub fn new(shader_dir: &Path) -> Result<Self, NttError> {
        let ctx = MetalContext::new(shader_dir)?;
        // Reuse V2's device and threadgroup kernels for the GPU phase
        let device_pipeline = ctx.make_pipeline("ct_gs_r2_butterfly_device")?;
        let tg_pipeline = ctx.make_pipeline("ct_gs_r2_forward_tg")?;
        Ok(CooperativeNttContext {
            ctx,
            device_pipeline,
            tg_pipeline,
        })
    }
}

/// Run a cooperative forward Circle NTT with CPU-GPU split.
///
/// `split_layer`: number of layers the CPU processes (from the top of the
/// butterfly network). The GPU processes the remaining layers.
///
/// - `split_layer = 0`: all GPU (equivalent to V2 forward NTT)
/// - `split_layer = log_n`: all CPU (equivalent to CPU reference)
/// - `split_layer = S`: CPU does layers (log_n-1) down to (log_n-S),
///   GPU does layers (log_n-S-1) down to 0
///
/// Returns timing breakdown (CPU phase, GPU phase, total).
pub fn cooperative_forward_ntt(
    coop: &CooperativeNttContext,
    input: &[u32],
    log_n: usize,
    split_layer: usize,
) -> Result<CooperativeResult, NttError> {
    let n = input.len();
    if n != (1 << log_n) {
        return Err(NttError::InvalidSize(n));
    }
    if split_layer > log_n {
        return Err(NttError::InvalidSize(split_layer));
    }
    if log_n == 0 {
        return Ok(CooperativeResult {
            data: input.to_vec(),
            cpu_time_us: 0.0,
            gpu_time_us: 0.0,
            total_time_us: 0.0,
        });
    }

    // Generate twiddles (same as CPU reference and V2)
    let coset = Coset::odds(log_n as u32);
    let twiddles = generate_twiddles(&coset);

    // Allocate shared buffer — CPU and GPU operate on the same memory
    let buf_data = coop.ctx.buffer_from_slice(input)?;

    let total_start = Instant::now();

    // ── CPU phase: layers (log_n-1) down to (log_n - split_layer) ──────
    let cpu_start = Instant::now();

    if split_layer > 0 {
        // Get raw pointer to shared buffer for CPU-side butterflies
        let ptr = buf_data.contents() as *mut u32;
        let data_slice = unsafe { std::slice::from_raw_parts_mut(ptr, n) };

        // Process layers from top (highest stride) down
        let cpu_start_layer = log_n - 1;
        let cpu_end_layer = log_n - split_layer; // exclusive (don't process this layer)

        for layer in (cpu_end_layer..=cpu_start_layer).rev() {
            let half = 1usize << layer;
            let full = half << 1;
            let tw = &twiddles[layer];

            for block in 0..(n / full) {
                let t = tw[block];
                for j in 0..half {
                    let idx0 = block * full + j;
                    let idx1 = idx0 + half;
                    // CT-DIT butterfly: v0' = v0 + t*v1, v1' = v0 - t*v1
                    let tmp = M31(data_slice[idx1]).mul(t);
                    let v0 = M31(data_slice[idx0]);
                    data_slice[idx0] = v0.add(tmp).0;
                    data_slice[idx1] = v0.sub(tmp).0;
                }
            }
        }

        // Compiler fence: ensure CPU stores are ordered before GPU dispatch.
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
    }

    let cpu_time_us = cpu_start.elapsed().as_secs_f64() * 1_000_000.0;

    // ── GPU phase: layers (log_n - split_layer - 1) down to 0 ──────────
    let gpu_start = Instant::now();
    let gpu_layers = log_n - split_layer; // number of remaining layers for GPU

    if gpu_layers > 0 {
        let tile_log = gpu_layers.min(MAX_TILE_LOG);

        // Device-memory stages: layers from (gpu_layers-1) down to tile_log
        for layer in (tile_log..gpu_layers).rev() {
            let stride = 1usize << layer;
            coop.ctx.dispatch_butterfly_r2(
                &coop.device_pipeline,
                &buf_data,
                &twiddles[layer],
                stride,
                n,
            )?;
        }

        // Threadgroup stages: layers (tile_log-1) down to 0
        if tile_log > 0 {
            let num_tg_layers = tile_log;
            let start_layer = tile_log - 1;

            let mut flat_tw = Vec::new();
            let mut tw_offsets = Vec::new();
            for li in 0..num_tg_layers {
                let layer = start_layer - li;
                tw_offsets.push(flat_tw.len() as u32);
                flat_tw.extend(twiddles[layer].iter().map(|m| m.0));
            }
            let buf_tw = coop.ctx.buffer_from_slice(&flat_tw)?;

            let mut params: Vec<u32> = vec![
                n as u32,
                tile_log as u32,
                num_tg_layers as u32,
                start_layer as u32,
            ];
            params.extend(tw_offsets);
            let buf_p = coop.ctx.buffer_from_slice(&params)?;

            let tile_size = 1usize << tile_log;
            let num_tiles = n / tile_size;
            let max_tg_threads =
                MetalContext::max_threads_per_threadgroup(&coop.tg_pipeline) as u64;
            let threads = max_tg_threads.min(tile_size as u64 / 2).max(1);

            let tg = MTLSize::new(num_tiles as u64, 1, 1);
            let tpg = MTLSize::new(threads, 1, 1);

            coop.ctx.dispatch_and_wait(
                &coop.tg_pipeline,
                &[&buf_data, &buf_tw, &buf_p],
                tg,
                tpg,
            )?;
        }
    }

    let gpu_time_us = gpu_start.elapsed().as_secs_f64() * 1_000_000.0;
    let total_time_us = total_start.elapsed().as_secs_f64() * 1_000_000.0;

    let result = MetalContext::read_buffer(&buf_data, n);

    Ok(CooperativeResult {
        data: result,
        cpu_time_us,
        gpu_time_us,
        total_time_us,
    })
}

// Twiddle generation and bit-reversal utilities are in crate::ntt::twiddles.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ntt::cpu_reference::CpuReferenceBackend;
    use crate::ntt::NttBackend;
    use crate::ntt::test_utils::try_init_metal;

    fn init() -> Option<CooperativeNttContext> {
        try_init_metal(|p| CooperativeNttContext::new(p))
    }

    fn lcg_data_u32(n: usize, seed: u64) -> Vec<u32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((s >> 33) as u32) % M31::P
            })
            .collect()
    }

    // ── split=0 (all GPU) must match V2 / CPU reference ─────────────────

    #[test]
    fn test_split0_matches_cpu_size1024() {
        let coop = match init() {
            Some(c) => c,
            None => return,
        };
        let cpu = CpuReferenceBackend;
        let input = lcg_data_u32(1024, 11111);
        let mut cpu_data: Vec<M31> = input.iter().map(|&v| M31(v)).collect();
        cpu.forward_ntt(&mut cpu_data, &[]).unwrap();

        let result = cooperative_forward_ntt(&coop, &input, 10, 0).unwrap();
        let expected: Vec<u32> = cpu_data.iter().map(|m| m.0).collect();
        assert_eq!(result.data, expected, "split=0 should match CPU reference");
    }

    #[test]
    fn test_split0_matches_cpu_size16384() {
        let coop = match init() {
            Some(c) => c,
            None => return,
        };
        let cpu = CpuReferenceBackend;
        let input = lcg_data_u32(16384, 22222);
        let mut cpu_data: Vec<M31> = input.iter().map(|&v| M31(v)).collect();
        cpu.forward_ntt(&mut cpu_data, &[]).unwrap();

        let result = cooperative_forward_ntt(&coop, &input, 14, 0).unwrap();
        let expected: Vec<u32> = cpu_data.iter().map(|m| m.0).collect();
        assert_eq!(result.data, expected, "split=0 at 16384 should match CPU");
    }

    // ── split=k (all CPU) must match CPU reference ──────────────────────

    #[test]
    fn test_split_all_cpu_size1024() {
        let coop = match init() {
            Some(c) => c,
            None => return,
        };
        let cpu = CpuReferenceBackend;
        let input = lcg_data_u32(1024, 33333);
        let mut cpu_data: Vec<M31> = input.iter().map(|&v| M31(v)).collect();
        cpu.forward_ntt(&mut cpu_data, &[]).unwrap();

        let result = cooperative_forward_ntt(&coop, &input, 10, 10).unwrap();
        let expected: Vec<u32> = cpu_data.iter().map(|m| m.0).collect();
        assert_eq!(result.data, expected, "split=k should match CPU reference");
    }

    #[test]
    fn test_split_all_cpu_size16384() {
        let coop = match init() {
            Some(c) => c,
            None => return,
        };
        let cpu = CpuReferenceBackend;
        let input = lcg_data_u32(16384, 44444);
        let mut cpu_data: Vec<M31> = input.iter().map(|&v| M31(v)).collect();
        cpu.forward_ntt(&mut cpu_data, &[]).unwrap();

        let result = cooperative_forward_ntt(&coop, &input, 14, 14).unwrap();
        let expected: Vec<u32> = cpu_data.iter().map(|m| m.0).collect();
        assert_eq!(result.data, expected, "split=14 should match CPU reference");
    }

    // ── Intermediate split values must also match ───────────────────────

    #[test]
    fn test_split_1_size1024() {
        let coop = match init() {
            Some(c) => c,
            None => return,
        };
        let cpu = CpuReferenceBackend;
        let input = lcg_data_u32(1024, 55555);
        let mut cpu_data: Vec<M31> = input.iter().map(|&v| M31(v)).collect();
        cpu.forward_ntt(&mut cpu_data, &[]).unwrap();

        let result = cooperative_forward_ntt(&coop, &input, 10, 1).unwrap();
        let expected: Vec<u32> = cpu_data.iter().map(|m| m.0).collect();
        assert_eq!(result.data, expected, "split=1 should match CPU reference");
    }

    #[test]
    fn test_split_half_size1024() {
        let coop = match init() {
            Some(c) => c,
            None => return,
        };
        let cpu = CpuReferenceBackend;
        let input = lcg_data_u32(1024, 66666);
        let mut cpu_data: Vec<M31> = input.iter().map(|&v| M31(v)).collect();
        cpu.forward_ntt(&mut cpu_data, &[]).unwrap();

        let result = cooperative_forward_ntt(&coop, &input, 10, 5).unwrap();
        let expected: Vec<u32> = cpu_data.iter().map(|m| m.0).collect();
        assert_eq!(result.data, expected, "split=5 (half) should match CPU reference");
    }

    #[test]
    fn test_split_sweep_size256() {
        // Test ALL split values for a small size
        let coop = match init() {
            Some(c) => c,
            None => return,
        };
        let cpu = CpuReferenceBackend;
        let log_n = 8;
        let input = lcg_data_u32(256, 77777);
        let mut cpu_data: Vec<M31> = input.iter().map(|&v| M31(v)).collect();
        cpu.forward_ntt(&mut cpu_data, &[]).unwrap();
        let expected: Vec<u32> = cpu_data.iter().map(|m| m.0).collect();

        for split in 0..=log_n {
            let result = cooperative_forward_ntt(&coop, &input, log_n, split).unwrap();
            assert_eq!(
                result.data, expected,
                "split={} at size 256 should match CPU reference",
                split
            );
        }
    }

    #[test]
    fn test_split_sweep_size4096() {
        // Test all split values at a larger size (exercises TG boundary)
        let coop = match init() {
            Some(c) => c,
            None => return,
        };
        let cpu = CpuReferenceBackend;
        let log_n = 12;
        let input = lcg_data_u32(4096, 88888);
        let mut cpu_data: Vec<M31> = input.iter().map(|&v| M31(v)).collect();
        cpu.forward_ntt(&mut cpu_data, &[]).unwrap();
        let expected: Vec<u32> = cpu_data.iter().map(|m| m.0).collect();

        for split in 0..=log_n {
            let result = cooperative_forward_ntt(&coop, &input, log_n, split).unwrap();
            assert_eq!(
                result.data, expected,
                "split={} at size 4096 should match CPU reference",
                split
            );
        }
    }

    // ── Edge cases ──────────────────────────────────────────────────────

    #[test]
    fn test_split_invalid() {
        let coop = match init() {
            Some(c) => c,
            None => return,
        };
        let input = lcg_data_u32(1024, 99999);
        // split_layer > log_n should error
        assert!(cooperative_forward_ntt(&coop, &input, 10, 11).is_err());
    }

    #[test]
    fn test_size4_all_splits() {
        let coop = match init() {
            Some(c) => c,
            None => return,
        };
        let cpu = CpuReferenceBackend;
        let input = vec![1u32, 2, 3, 4];
        let mut cpu_data: Vec<M31> = input.iter().map(|&v| M31(v)).collect();
        cpu.forward_ntt(&mut cpu_data, &[]).unwrap();
        let expected: Vec<u32> = cpu_data.iter().map(|m| m.0).collect();

        for split in 0..=2 {
            let result = cooperative_forward_ntt(&coop, &input, 2, split).unwrap();
            assert_eq!(result.data, expected, "split={} at size 4 failed", split);
        }
    }

    // ── Timing sanity ───────────────────────────────────────────────────

    #[test]
    fn test_timing_breakdown() {
        let coop = match init() {
            Some(c) => c,
            None => return,
        };
        let input = lcg_data_u32(1024, 12345);

        // All GPU: cpu_time should be ~0
        let r0 = cooperative_forward_ntt(&coop, &input, 10, 0).unwrap();
        // Generous threshold: skipped phase measures only Instant::now() overhead + OS jitter
        assert!(r0.cpu_time_us < 100.0, "split=0 should have near-zero CPU time");
        assert!(r0.gpu_time_us > 0.0, "split=0 should have nonzero GPU time");

        // All CPU: gpu_time should be ~0
        let rk = cooperative_forward_ntt(&coop, &input, 10, 10).unwrap();
        assert!(rk.cpu_time_us > 0.0, "split=k should have nonzero CPU time");
        assert!(rk.gpu_time_us < 100.0, "split=k should have near-zero GPU time");

        // Total should be >= max(cpu, gpu)
        let r5 = cooperative_forward_ntt(&coop, &input, 10, 5).unwrap();
        assert!(r5.total_time_us >= r5.cpu_time_us);
        assert!(r5.total_time_us >= r5.gpu_time_us);
    }
}
