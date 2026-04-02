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

use crate::field::m31::M31;
use crate::field::Field;
use crate::ntt::twiddles::TwiddleCache;
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
    twiddle_cache: TwiddleCache,
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
            twiddle_cache: TwiddleCache::new(),
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

        let twiddles = self.twiddle_cache.forward(log_n as u32);

        let buf_data = self.ctx.buffer_from_slice(input)?;
        let cmd = self.ctx.begin_batch();
        let mut retain = Vec::new();

        let tile_log = log_n.min(MAX_TILE_LOG);

        // ── Phase 1: device-memory stages (large strides) ──────────────
        // Forward: layers from (log_n-1) down to tile_log
        for layer in (tile_log..log_n).rev() {
            let stride = 1usize << layer;
            self.ctx.encode_butterfly_r2(
                cmd,
                &mut retain,
                &self.forward_device_pipeline,
                &buf_data,
                &twiddles[layer],
                stride,
                n,
            )?;
        }

        // ── Phase 2: threadgroup stages (small strides) ────────────────
        // Forward: layers from (tile_log-1) down to 0
        if tile_log > 0 {
            self.encode_threadgroup_forward(
                cmd,
                &mut retain,
                &buf_data,
                &twiddles,
                n,
                tile_log,
            )?;
        }

        let total_ns = MetalContext::submit_batch(cmd, &retain)?;
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

        let itwiddles = self.twiddle_cache.inverse(log_n as u32);

        let buf_data = self.ctx.buffer_from_slice(input)?;
        let cmd = self.ctx.begin_batch();
        let mut retain = Vec::new();

        let tile_log = log_n.min(MAX_TILE_LOG);

        // ── Phase 1: threadgroup stages (small strides) ────────────────
        // Inverse: layers from 0 up to (tile_log-1)
        if tile_log > 0 {
            self.encode_threadgroup_inverse(
                cmd,
                &mut retain,
                &buf_data,
                &itwiddles,
                n,
                tile_log,
            )?;
        }

        // ── Phase 2: device-memory stages (large strides) ──────────────
        // Inverse: layers from tile_log up to (log_n-1)
        for layer in tile_log..log_n {
            let stride = 1usize << layer;
            self.ctx.encode_butterfly_r2(
                cmd,
                &mut retain,
                &self.inverse_device_pipeline,
                &buf_data,
                &itwiddles[layer],
                stride,
                n,
            )?;
        }

        // ── Normalize: multiply all elements by inv_n ──────────────────
        let inv_n = M31::reduce(n as u64).inv();
        self.ctx.encode_normalize(
            cmd,
            &mut retain,
            &self.normalize_pipeline,
            &buf_data,
            n,
            inv_n,
        )?;

        let total_ns = MetalContext::submit_batch(cmd, &retain)?;
        let result = MetalContext::read_buffer(&buf_data, n);
        Ok((result, total_ns))
    }

    /// Encode the forward threadgroup kernel for layers (tile_log-1) down to 0.
    fn encode_threadgroup_forward(
        &self,
        cmd: &CommandBufferRef,
        retain: &mut Vec<Buffer>,
        buf_data: &Buffer,
        twiddles: &[Vec<M31>],
        n: usize,
        tile_log: usize,
    ) -> Result<(), NttError> {
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

        MetalContext::encode_dispatch(
            cmd,
            &self.forward_tg_pipeline,
            &[buf_data, &buf_tw, &buf_p],
            tg,
            tpg,
        );
        retain.push(buf_tw);
        retain.push(buf_p);
        Ok(())
    }

    /// Encode the inverse threadgroup kernel for layers 0 up to (tile_log-1).
    fn encode_threadgroup_inverse(
        &self,
        cmd: &CommandBufferRef,
        retain: &mut Vec<Buffer>,
        buf_data: &Buffer,
        itwiddles: &[Vec<M31>],
        n: usize,
        tile_log: usize,
    ) -> Result<(), NttError> {
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

        MetalContext::encode_dispatch(
            cmd,
            &self.inverse_tg_pipeline,
            &[buf_data, &buf_tw, &buf_p],
            tg,
            tpg,
        );
        retain.push(buf_tw);
        retain.push(buf_p);
        Ok(())
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
