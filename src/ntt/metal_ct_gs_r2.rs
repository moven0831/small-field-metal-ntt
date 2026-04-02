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
use crate::ntt::twiddles::{generate_twiddles, generate_itwiddles};
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

// Twiddle generation and bit-reversal utilities are in crate::ntt::twiddles.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ntt::test_utils::*;

    fn init() -> Option<MetalCtGsR2> {
        try_init_metal(|p| MetalCtGsR2::new(p))
    }

    #[test]
    fn test_forward_matches_cpu() {
        let gpu = match init() { Some(g) => g, None => return };
        assert_forward_matches_cpu(&gpu, &[4, 16, 256, 1024, 4096, 8192, 16384]);
    }

    #[test]
    fn test_roundtrip() {
        let gpu = match init() { Some(g) => g, None => return };
        assert_roundtrip(&gpu, &[4, 256, 1024, 8192, 16384]);
    }

    #[test]
    fn test_edge_cases() {
        let gpu = match init() { Some(g) => g, None => return };
        assert_edge_cases(&gpu);
        assert_inverse_edge_cases(&gpu);
    }

    #[test]
    fn test_pointwise_mul() {
        let gpu = match init() { Some(g) => g, None => return };
        assert_pointwise_mul(&gpu);
    }

    #[test]
    fn test_size2_forward_and_roundtrip() {
        let gpu = match init() { Some(g) => g, None => return };
        assert_forward_matches_cpu(&gpu, &[2]);
        assert_roundtrip(&gpu, &[2]);
    }
}
