//! Variant 2: CT-DIT/GS-DIF radix-2 NTT on Metal GPU — in-place with threadgroup memory.
//!
//! Two-phase execution:
//! - Threadgroup phase: Layers with stride ≤ tile_size/2 are processed entirely
//!   in on-chip threadgroup memory. One dispatch covers all small-stride layers.
//!   Cost: 1 device read + 1 device write + N on-chip barriers.
//! - Device-memory phase: Large-stride layers (stride > tile_size/2), one
//!   dispatch per layer.
//!
//! Key advantages over Variant 1:
//! - Threadgroup phase eliminates most device-memory round-trips
//! - NO bit-reversal permutation between forward and inverse
//!   (forward produces bit-reversed output; inverse expects it)
//!
//! Total dispatches for 2^20 NTT: 1 (threadgroup, 13 stages) + 7 (device) = 8
//! vs Variant 1's 20 dispatches (all device memory).

use crate::field::circle::Coset;
use crate::field::m31::M31;
use crate::field::Field;
use crate::gpu::MetalContext;
use crate::ntt::{NttBackend, NttError};
use metal::*;
use std::path::Path;

/// Maximum elements per threadgroup tile: 8192 (= 32 KB of threadgroup memory).
/// Handles up to 13 butterfly stages on-chip.
const MAX_TILE_LOG: usize = 13;

pub struct MetalCtGsR2 {
    ctx: MetalContext,
    forward_tg_pipeline: ComputePipelineState,
    inverse_tg_pipeline: ComputePipelineState,
    forward_device_pipeline: ComputePipelineState,
    inverse_device_pipeline: ComputePipelineState,
    normalize_pipeline: ComputePipelineState,
}

impl MetalCtGsR2 {
    pub fn new(shader_dir: &Path) -> Result<Self, NttError> {
        let ctx = MetalContext::new(shader_dir)?;
        let forward_tg = ctx.make_pipeline("ct_gs_r2_forward_tg")?;
        let inverse_tg = ctx.make_pipeline("ct_gs_r2_inverse_tg")?;
        let forward_dev = ctx.make_pipeline("ct_gs_r2_butterfly_device")?;
        let inverse_dev = ctx.make_pipeline("ct_gs_r2_butterfly_device_inv")?;
        let normalize = ctx.make_pipeline("ct_gs_r2_normalize")?;

        Ok(MetalCtGsR2 {
            ctx,
            forward_tg_pipeline: forward_tg,
            inverse_tg_pipeline: inverse_tg,
            forward_device_pipeline: forward_dev,
            inverse_device_pipeline: inverse_dev,
            normalize_pipeline: normalize,
        })
    }

    /// Run forward Circle NTT (CT-DIT) on GPU.
    /// Returns (result_data, total_gpu_time_ns).
    pub fn forward_ntt_gpu(
        &self,
        input: &[u32],
        log_n: usize,
    ) -> Result<(Vec<u32>, u64), NttError> {
        let n = input.len();
        if n != (1 << log_n) {
            return Err(NttError::InvalidSize(n));
        }
        if log_n > 30 {
            return Err(NttError::InvalidSize(n));
        }
        if log_n == 0 {
            return Ok((input.to_vec(), 0));
        }

        let coset = Coset::odds(log_n as u32);
        let twiddles = generate_twiddles(&coset);

        let buf_data = self.ctx.buffer_from_slice(input)?;
        let mut total_ns: u64 = 0;

        let tile_log = log_n.min(MAX_TILE_LOG);

        // ── Phase 1: device-memory stages (large strides) ──────────────
        // Forward: layers from (log_n-1) down to tile_log
        let max_tpg =
            MetalContext::max_threads_per_threadgroup(&self.forward_device_pipeline) as u64;
        let tpg_width = max_tpg.min(256);

        for layer in (tile_log..log_n).rev() {
            let stride = 1usize << layer;
            let num_butterflies = (n / 2) as u64;

            let tw_data: Vec<u32> = twiddles[layer].iter().map(|m| m.0).collect();
            let buf_tw = self.ctx.buffer_from_slice(&tw_data)?;
            let params: Vec<u32> = vec![stride as u32, n as u32];
            let buf_p = self.ctx.buffer_from_slice(&params)?;

            let tg = MTLSize::new(
                (num_butterflies + tpg_width - 1) / tpg_width,
                1,
                1,
            );
            let tpg = MTLSize::new(tpg_width.min(num_butterflies), 1, 1);

            let ns = self.ctx.dispatch_and_wait(
                &self.forward_device_pipeline,
                &[&buf_data, &buf_tw, &buf_p],
                tg,
                tpg,
            )?;
            total_ns += ns;
        }

        // ── Phase 2: threadgroup stages (small strides) ────────────────
        // Forward: layers from (tile_log-1) down to 0
        if tile_log > 0 {
            let ns = self.dispatch_threadgroup_forward(
                &buf_data,
                &twiddles,
                n,
                tile_log,
            )?;
            total_ns += ns;
        }

        let result = MetalContext::read_buffer(&buf_data, n);
        Ok((result, total_ns))
    }

    /// Run inverse Circle NTT (GS-DIF) on GPU.
    /// Returns (result_data, total_gpu_time_ns).
    pub fn inverse_ntt_gpu(
        &self,
        input: &[u32],
        log_n: usize,
    ) -> Result<(Vec<u32>, u64), NttError> {
        let n = input.len();
        if n != (1 << log_n) {
            return Err(NttError::InvalidSize(n));
        }
        if log_n > 30 {
            return Err(NttError::InvalidSize(n));
        }
        if log_n == 0 {
            return Ok((input.to_vec(), 0));
        }

        let coset = Coset::odds(log_n as u32);
        let itwiddles = generate_itwiddles(&coset);

        let buf_data = self.ctx.buffer_from_slice(input)?;
        let mut total_ns: u64 = 0;

        let tile_log = log_n.min(MAX_TILE_LOG);

        // ── Phase 1: threadgroup stages (small strides) ────────────────
        // Inverse: layers from 0 up to (tile_log-1)
        if tile_log > 0 {
            let ns = self.dispatch_threadgroup_inverse(
                &buf_data,
                &itwiddles,
                n,
                tile_log,
            )?;
            total_ns += ns;
        }

        // ── Phase 2: device-memory stages (large strides) ──────────────
        // Inverse: layers from tile_log up to (log_n-1)
        let max_tpg =
            MetalContext::max_threads_per_threadgroup(&self.inverse_device_pipeline) as u64;
        let tpg_width = max_tpg.min(256);

        for layer in tile_log..log_n {
            let stride = 1usize << layer;
            let num_butterflies = (n / 2) as u64;

            let tw_data: Vec<u32> = itwiddles[layer].iter().map(|m| m.0).collect();
            let buf_tw = self.ctx.buffer_from_slice(&tw_data)?;
            let params: Vec<u32> = vec![stride as u32, n as u32];
            let buf_p = self.ctx.buffer_from_slice(&params)?;

            let tg = MTLSize::new(
                (num_butterflies + tpg_width - 1) / tpg_width,
                1,
                1,
            );
            let tpg = MTLSize::new(tpg_width.min(num_butterflies), 1, 1);

            let ns = self.ctx.dispatch_and_wait(
                &self.inverse_device_pipeline,
                &[&buf_data, &buf_tw, &buf_p],
                tg,
                tpg,
            )?;
            total_ns += ns;
        }

        // ── Normalize: multiply all elements by inv_n ──────────────────
        let inv_n = M31::reduce(n as u64).inv();
        let norm_params: Vec<u32> = vec![n as u32, inv_n.0];
        let buf_norm_p = self.ctx.buffer_from_slice(&norm_params)?;

        let norm_max_tpg =
            MetalContext::max_threads_per_threadgroup(&self.normalize_pipeline) as u64;
        let norm_tpg_width = norm_max_tpg.min(256);
        let norm_tg = MTLSize::new(
            (n as u64 + norm_tpg_width - 1) / norm_tpg_width,
            1,
            1,
        );
        let norm_tpg = MTLSize::new(norm_tpg_width.min(n as u64), 1, 1);

        let ns = self.ctx.dispatch_and_wait(
            &self.normalize_pipeline,
            &[&buf_data, &buf_norm_p],
            norm_tg,
            norm_tpg,
        )?;
        total_ns += ns;

        let result = MetalContext::read_buffer(&buf_data, n);
        Ok((result, total_ns))
    }

    /// Dispatch the forward threadgroup kernel for layers (tile_log-1) down to 0.
    fn dispatch_threadgroup_forward(
        &self,
        buf_data: &Buffer,
        twiddles: &[Vec<M31>],
        n: usize,
        tile_log: usize,
    ) -> Result<u64, NttError> {
        let num_tg_layers = tile_log;
        let start_layer = tile_log - 1;

        // Flatten twiddles for all threadgroup layers (processing order: high to low)
        let mut flat_tw = Vec::new();
        let mut tw_offsets = Vec::new();
        for li in 0..num_tg_layers {
            let layer = start_layer - li;
            tw_offsets.push(flat_tw.len() as u32);
            flat_tw.extend(twiddles[layer].iter().map(|m| m.0));
        }
        let buf_tw = self.ctx.buffer_from_slice(&flat_tw)?;

        // params: [n, tile_log, num_layers, start_layer, offsets...]
        let mut params: Vec<u32> = vec![
            n as u32,
            tile_log as u32,
            num_tg_layers as u32,
            start_layer as u32,
        ];
        params.extend(tw_offsets);
        let buf_p = self.ctx.buffer_from_slice(&params)?;

        let tile_size = 1usize << tile_log;
        let num_tiles = n / tile_size;
        let max_tg_threads =
            MetalContext::max_threads_per_threadgroup(&self.forward_tg_pipeline) as u64;
        let threads = max_tg_threads.min(tile_size as u64 / 2).max(1);

        let tg = MTLSize::new(num_tiles as u64, 1, 1);
        let tpg = MTLSize::new(threads, 1, 1);

        self.ctx.dispatch_and_wait(
            &self.forward_tg_pipeline,
            &[buf_data, &buf_tw, &buf_p],
            tg,
            tpg,
        )
    }

    /// Dispatch the inverse threadgroup kernel for layers 0 up to (tile_log-1).
    fn dispatch_threadgroup_inverse(
        &self,
        buf_data: &Buffer,
        itwiddles: &[Vec<M31>],
        n: usize,
        tile_log: usize,
    ) -> Result<u64, NttError> {
        let num_tg_layers = tile_log;
        let start_layer = 0;

        // Flatten inverse twiddles (processing order: low to high)
        let mut flat_tw = Vec::new();
        let mut tw_offsets = Vec::new();
        for li in 0..num_tg_layers {
            let layer = start_layer + li;
            tw_offsets.push(flat_tw.len() as u32);
            flat_tw.extend(itwiddles[layer].iter().map(|m| m.0));
        }
        let buf_tw = self.ctx.buffer_from_slice(&flat_tw)?;

        let mut params: Vec<u32> = vec![
            n as u32,
            tile_log as u32,
            num_tg_layers as u32,
            start_layer as u32,
        ];
        params.extend(tw_offsets);
        let buf_p = self.ctx.buffer_from_slice(&params)?;

        let tile_size = 1usize << tile_log;
        let num_tiles = n / tile_size;
        let max_tg_threads =
            MetalContext::max_threads_per_threadgroup(&self.inverse_tg_pipeline) as u64;
        let threads = max_tg_threads.min(tile_size as u64 / 2).max(1);

        let tg = MTLSize::new(num_tiles as u64, 1, 1);
        let tpg = MTLSize::new(threads, 1, 1);

        self.ctx.dispatch_and_wait(
            &self.inverse_tg_pipeline,
            &[buf_data, &buf_tw, &buf_p],
            tg,
            tpg,
        )
    }
}

impl NttBackend<M31> for MetalCtGsR2 {
    fn name(&self) -> &str {
        "metal-ct-gs-r2"
    }

    fn forward_ntt(&self, data: &mut [M31], _twiddles: &[M31]) -> Result<(), NttError> {
        let n = data.len();
        if n == 0 || (n & (n - 1)) != 0 {
            return Err(NttError::InvalidSize(n));
        }
        if n == 1 {
            return Ok(());
        }
        let log_n = n.trailing_zeros() as usize;

        let input: Vec<u32> = data.iter().map(|m| m.0).collect();
        let (result, _) = self.forward_ntt_gpu(&input, log_n)?;
        for (i, val) in result.iter().enumerate() {
            data[i] = M31(*val);
        }
        Ok(())
    }

    fn inverse_ntt(&self, data: &mut [M31], _twiddles: &[M31]) -> Result<(), NttError> {
        let n = data.len();
        if n == 0 || (n & (n - 1)) != 0 {
            return Err(NttError::InvalidSize(n));
        }
        if n == 1 {
            return Ok(());
        }
        let log_n = n.trailing_zeros() as usize;

        let input: Vec<u32> = data.iter().map(|m| m.0).collect();
        let (result, _) = self.inverse_ntt_gpu(&input, log_n)?;
        for (i, val) in result.iter().enumerate() {
            data[i] = M31(*val);
        }
        Ok(())
    }

    fn pointwise_mul(&self, a: &[M31], b: &[M31], out: &mut [M31]) -> Result<(), NttError> {
        if a.len() != b.len() || a.len() != out.len() {
            return Err(NttError::InvalidSize(a.len()));
        }
        for i in 0..a.len() {
            out[i] = a[i].mul(b[i]);
        }
        Ok(())
    }
}

// ─── Twiddle generation (same algorithm as CPU reference and Variant 1) ──

fn generate_twiddles(coset: &Coset) -> Vec<Vec<M31>> {
    let log_n = coset.log_size as usize;
    let mut result = Vec::with_capacity(log_n);
    let mut current = coset.clone();

    for layer_idx in 0..log_n {
        let half_size = current.size() / 2;
        let is_last = layer_idx == log_n - 1;
        let layer_tw: Vec<M31> = (0..half_size)
            .map(|i| {
                let p = current.at(bit_reverse_idx(i, current.log_size - 1));
                if is_last {
                    p.y
                } else {
                    p.x
                }
            })
            .collect();
        result.push(layer_tw);
        current = current.double();
    }
    result
}

fn generate_itwiddles(coset: &Coset) -> Vec<Vec<M31>> {
    let log_n = coset.log_size as usize;
    let mut result = Vec::with_capacity(log_n);
    let mut current = coset.clone();

    for layer_idx in 0..log_n {
        let half_size = current.size() / 2;
        let is_last = layer_idx == log_n - 1;
        let layer_tw: Vec<M31> = (0..half_size)
            .map(|i| {
                let p = current.at(bit_reverse_idx(i, current.log_size - 1));
                let t = if is_last { p.y } else { p.x };
                t.inv()
            })
            .collect();
        result.push(layer_tw);
        current = current.double();
    }
    result
}

fn bit_reverse_idx(index: usize, log_size: u32) -> usize {
    let mut val = index as u32;
    let mut result = 0u32;
    for _ in 0..log_size {
        result = (result << 1) | (val & 1);
        val >>= 1;
    }
    result as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ntt::cpu_reference::CpuReferenceBackend;
    use crate::ntt::NttBackend;
    use std::path::PathBuf;

    fn shader_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders")
    }

    fn skip_if_no_metal() -> Option<MetalCtGsR2> {
        match MetalCtGsR2::new(&shader_dir()) {
            Ok(g) => Some(g),
            Err(NttError::DeviceNotFound) => {
                eprintln!("No Metal device — skipping");
                None
            }
            Err(e) => panic!("Failed: {}", e),
        }
    }

    // ── Forward NTT: GPU vs CPU reference ────────────────────────────────

    #[test]
    fn test_forward_matches_cpu_size4() {
        let gpu = match skip_if_no_metal() {
            Some(g) => g,
            None => return,
        };
        let cpu = CpuReferenceBackend;
        let original = vec![M31(1), M31(2), M31(3), M31(4)];
        let mut cpu_data = original.clone();
        cpu.forward_ntt(&mut cpu_data, &[]).unwrap();
        let mut gpu_data = original.clone();
        gpu.forward_ntt(&mut gpu_data, &[]).unwrap();
        assert_eq!(gpu_data, cpu_data, "Forward mismatch at size 4");
    }

    #[test]
    fn test_forward_matches_cpu_size16() {
        let gpu = match skip_if_no_metal() {
            Some(g) => g,
            None => return,
        };
        let cpu = CpuReferenceBackend;
        let n = 16;
        let original: Vec<M31> = (0..n).map(|i| M31(i as u32 * 7 + 3)).collect();
        let mut cpu_data = original.clone();
        cpu.forward_ntt(&mut cpu_data, &[]).unwrap();
        let mut gpu_data = original.clone();
        gpu.forward_ntt(&mut gpu_data, &[]).unwrap();
        assert_eq!(gpu_data, cpu_data, "Forward mismatch at size {}", n);
    }

    #[test]
    fn test_forward_matches_cpu_size256() {
        let gpu = match skip_if_no_metal() {
            Some(g) => g,
            None => return,
        };
        let cpu = CpuReferenceBackend;
        let n = 256;
        let original: Vec<M31> = (0..n).map(|i| M31((i as u32 * 13 + 7) % M31::P)).collect();
        let mut cpu_data = original.clone();
        cpu.forward_ntt(&mut cpu_data, &[]).unwrap();
        let mut gpu_data = original.clone();
        gpu.forward_ntt(&mut gpu_data, &[]).unwrap();
        assert_eq!(gpu_data, cpu_data, "Forward mismatch at size {}", n);
    }

    #[test]
    fn test_forward_matches_cpu_size1024() {
        let gpu = match skip_if_no_metal() {
            Some(g) => g,
            None => return,
        };
        let cpu = CpuReferenceBackend;
        let n = 1024;
        let original: Vec<M31> = (0..n).map(|i| M31((i as u32 * 17 + 11) % M31::P)).collect();
        let mut cpu_data = original.clone();
        cpu.forward_ntt(&mut cpu_data, &[]).unwrap();
        let mut gpu_data = original.clone();
        gpu.forward_ntt(&mut gpu_data, &[]).unwrap();
        assert_eq!(gpu_data, cpu_data, "Forward mismatch at size {}", n);
    }

    #[test]
    fn test_forward_matches_cpu_size4096() {
        let gpu = match skip_if_no_metal() {
            Some(g) => g,
            None => return,
        };
        let cpu = CpuReferenceBackend;
        let n = 4096;
        let original: Vec<M31> = lcg_data(n, 98765);
        let mut cpu_data = original.clone();
        cpu.forward_ntt(&mut cpu_data, &[]).unwrap();
        let mut gpu_data = original.clone();
        gpu.forward_ntt(&mut gpu_data, &[]).unwrap();
        assert_eq!(gpu_data, cpu_data, "Forward mismatch at size {}", n);
    }

    #[test]
    fn test_forward_matches_cpu_size8192() {
        // Exactly one threadgroup tile
        let gpu = match skip_if_no_metal() {
            Some(g) => g,
            None => return,
        };
        let cpu = CpuReferenceBackend;
        let n = 8192;
        let original: Vec<M31> = lcg_data(n, 11111);
        let mut cpu_data = original.clone();
        cpu.forward_ntt(&mut cpu_data, &[]).unwrap();
        let mut gpu_data = original.clone();
        gpu.forward_ntt(&mut gpu_data, &[]).unwrap();
        assert_eq!(gpu_data, cpu_data, "Forward mismatch at size {}", n);
    }

    #[test]
    fn test_forward_matches_cpu_size16384() {
        // Two tiles: tests device-memory phase + threadgroup phase
        let gpu = match skip_if_no_metal() {
            Some(g) => g,
            None => return,
        };
        let cpu = CpuReferenceBackend;
        let n = 16384;
        let original: Vec<M31> = lcg_data(n, 22222);
        let mut cpu_data = original.clone();
        cpu.forward_ntt(&mut cpu_data, &[]).unwrap();
        let mut gpu_data = original.clone();
        gpu.forward_ntt(&mut gpu_data, &[]).unwrap();
        assert_eq!(gpu_data, cpu_data, "Forward mismatch at size {}", n);
    }

    // ── Round-trip: forward + inverse = identity ─────────────────────────

    #[test]
    fn test_roundtrip_size4() {
        let gpu = match skip_if_no_metal() {
            Some(g) => g,
            None => return,
        };
        let original = vec![M31(1), M31(2), M31(3), M31(4)];
        let mut data = original.clone();
        gpu.forward_ntt(&mut data, &[]).unwrap();
        assert_ne!(data, original, "Forward should change data");
        gpu.inverse_ntt(&mut data, &[]).unwrap();
        assert_eq!(data, original, "Round-trip failed at size 4");
    }

    #[test]
    fn test_roundtrip_size256() {
        let gpu = match skip_if_no_metal() {
            Some(g) => g,
            None => return,
        };
        let original: Vec<M31> = (0..256).map(|i| M31(i * 7 + 3)).collect();
        let mut data = original.clone();
        gpu.forward_ntt(&mut data, &[]).unwrap();
        gpu.inverse_ntt(&mut data, &[]).unwrap();
        assert_eq!(data, original, "Round-trip failed at size 256");
    }

    #[test]
    fn test_roundtrip_size1024() {
        let gpu = match skip_if_no_metal() {
            Some(g) => g,
            None => return,
        };
        let original: Vec<M31> = (0..1024).map(|i| M31((i * 13 + 7) % M31::P)).collect();
        let mut data = original.clone();
        gpu.forward_ntt(&mut data, &[]).unwrap();
        gpu.inverse_ntt(&mut data, &[]).unwrap();
        assert_eq!(data, original, "Round-trip failed at size 1024");
    }

    #[test]
    fn test_roundtrip_size8192() {
        let gpu = match skip_if_no_metal() {
            Some(g) => g,
            None => return,
        };
        let original: Vec<M31> = lcg_data(8192, 33333);
        let mut data = original.clone();
        gpu.forward_ntt(&mut data, &[]).unwrap();
        gpu.inverse_ntt(&mut data, &[]).unwrap();
        assert_eq!(data, original, "Round-trip failed at size 8192");
    }

    #[test]
    fn test_roundtrip_size16384() {
        // Tests two-phase round-trip (device + threadgroup)
        let gpu = match skip_if_no_metal() {
            Some(g) => g,
            None => return,
        };
        let original: Vec<M31> = lcg_data(16384, 44444);
        let mut data = original.clone();
        gpu.forward_ntt(&mut data, &[]).unwrap();
        gpu.inverse_ntt(&mut data, &[]).unwrap();
        assert_eq!(data, original, "Round-trip failed at size 16384");
    }

    // ── Edge cases ───────────────────────────────────────────────────────

    #[test]
    fn test_all_zeros() {
        let gpu = match skip_if_no_metal() {
            Some(g) => g,
            None => return,
        };
        let mut data = vec![M31(0); 64];
        gpu.forward_ntt(&mut data, &[]).unwrap();
        assert!(data.iter().all(|&x| x == M31(0)));
    }

    #[test]
    fn test_size2() {
        let gpu = match skip_if_no_metal() {
            Some(g) => g,
            None => return,
        };
        let cpu = CpuReferenceBackend;
        let original = vec![M31(100), M31(200)];
        let mut cpu_data = original.clone();
        cpu.forward_ntt(&mut cpu_data, &[]).unwrap();
        let mut gpu_data = original.clone();
        gpu.forward_ntt(&mut gpu_data, &[]).unwrap();
        assert_eq!(gpu_data, cpu_data, "Forward mismatch at size 2");

        // Round-trip
        gpu.inverse_ntt(&mut gpu_data, &[]).unwrap();
        assert_eq!(gpu_data, original, "Round-trip failed at size 2");
    }

    // ── Helpers ──────────────────────────────────────────────────────────

    fn lcg_data(n: usize, seed: u64) -> Vec<M31> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                M31(((s >> 33) as u32) % M31::P)
            })
            .collect()
    }
}
