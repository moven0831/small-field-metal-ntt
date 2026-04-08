//! Variant 3: Stockham radix-2 out-of-place NTT on Metal GPU.
//!
//! Key difference from Variants 1 & 2: reads from buffer A, writes to buffer B,
//! then swaps. This tests whether the out-of-place ping-pong memory pattern
//! helps or hurts on Apple Silicon's Unified Memory Architecture.
//!
//! Two-phase execution:
//! - Device-memory phase: Large-stride layers, one dispatch per layer.
//!   Each dispatch reads from one device buffer and writes to the other.
//! - Threadgroup phase: Small-stride layers processed on-chip with two
//!   threadgroup arrays (2 x 4096 = 32 KB) ping-ponging between stages.
//!
//! Uses the same CT-DIT butterfly and twiddle factors as Variants 1 & 2.
//! Output is bit-reversed (same as CPU reference forward NTT).
//!
//! Memory overhead: 2x (two full-size device buffers).
//! Threadgroup tile: 4096 elements (half of Variant 2's 8192, due to needing
//! two arrays). Handles 12 stages on-chip vs Variant 2's 13.

use crate::field::m31::M31;
use crate::field::Field;
use crate::gpu::MetalContext;
use crate::ntt::twiddles::TwiddleCache;
use crate::ntt::{NttBackend, NttError};
use metal::*;
use std::path::Path;

/// Maximum elements per threadgroup tile: 4096 (two arrays of 16 KB each = 32 KB total).
const MAX_TILE_LOG: usize = 12;

pub struct MetalStockhamR2 {
    ctx: MetalContext,
    forward_tg_pipeline: ComputePipelineState,
    forward_device_pipeline: ComputePipelineState,
    inverse_tg_pipeline: ComputePipelineState,
    inverse_device_pipeline: ComputePipelineState,
    normalize_pipeline: ComputePipelineState,
    twiddle_cache: TwiddleCache,
}

impl MetalStockhamR2 {
    pub fn new(shader_dir: &Path) -> Result<Self, NttError> {
        let ctx = MetalContext::new(shader_dir)?;
        let forward_tg = ctx.make_pipeline("stockham_r2_forward_tg")?;
        let forward_dev = ctx.make_pipeline("stockham_r2_butterfly_device")?;
        let inverse_tg = ctx.make_pipeline("stockham_r2_inverse_tg")?;
        let inverse_dev = ctx.make_pipeline("stockham_r2_butterfly_device_inv")?;
        let normalize = ctx.make_pipeline("stockham_r2_normalize")?;

        Ok(MetalStockhamR2 {
            ctx,
            forward_tg_pipeline: forward_tg,
            forward_device_pipeline: forward_dev,
            inverse_tg_pipeline: inverse_tg,
            inverse_device_pipeline: inverse_dev,
            normalize_pipeline: normalize,
            twiddle_cache: TwiddleCache::new(),
        })
    }

    /// Run forward Circle NTT (CT-DIT, out-of-place) on GPU.
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

        // Allocate two device buffers for ping-pong
        let buf_a = self.ctx.buffer_from_slice(input)?;
        let buf_b = self.ctx.alloc_buffer(std::mem::size_of_val(input))?;
        let cmd = self.ctx.begin_batch();
        let mut retain = Vec::new();

        let tile_log = log_n.min(MAX_TILE_LOG);

        // Track which buffer is current input vs output
        // read_from_a: true means read from buf_a, write to buf_b
        let mut read_from_a = true;

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

            let tg = MTLSize::new(num_butterflies.div_ceil(tpg_width), 1, 1);
            let tpg = MTLSize::new(tpg_width.min(num_butterflies), 1, 1);

            let (cur_in, cur_out) = if read_from_a {
                (&buf_a, &buf_b)
            } else {
                (&buf_b, &buf_a)
            };

            MetalContext::encode_dispatch(
                cmd,
                &self.forward_device_pipeline,
                &[cur_in, cur_out, &buf_tw, &buf_p],
                tg,
                tpg,
            );
            retain.push(buf_tw);
            retain.push(buf_p);
            read_from_a = !read_from_a;
        }

        // ── Phase 2: threadgroup stages (small strides) ────────────────
        // Forward: layers from (tile_log-1) down to 0
        if tile_log > 0 {
            let num_tg_layers = tile_log;
            let start_layer = tile_log - 1;

            // Flatten twiddles for all threadgroup layers
            let mut flat_tw = Vec::new();
            let mut tw_offsets = Vec::new();
            for li in 0..num_tg_layers {
                let layer = start_layer - li;
                tw_offsets.push(flat_tw.len() as u32);
                flat_tw.extend(twiddles[layer].iter().map(|m| m.0));
            }
            let buf_tw = self.ctx.buffer_from_slice(&flat_tw)?;

            // params: [tile_log, num_layers, start_layer, offsets...]
            let mut params: Vec<u32> =
                vec![tile_log as u32, num_tg_layers as u32, start_layer as u32];
            params.extend(tw_offsets);
            let buf_p = self.ctx.buffer_from_slice(&params)?;

            let tile_size = 1usize << tile_log;
            let num_tiles = n / tile_size;
            let max_tg_threads =
                MetalContext::max_threads_per_threadgroup(&self.forward_tg_pipeline) as u64;
            let threads = max_tg_threads.min(tile_size as u64 / 2).max(1);

            let tg = MTLSize::new(num_tiles as u64, 1, 1);
            let tpg = MTLSize::new(threads, 1, 1);

            let (cur_in, cur_out) = if read_from_a {
                (&buf_a, &buf_b)
            } else {
                (&buf_b, &buf_a)
            };

            MetalContext::encode_dispatch(
                cmd,
                &self.forward_tg_pipeline,
                &[cur_in, cur_out, &buf_tw, &buf_p],
                tg,
                tpg,
            );
            retain.push(buf_tw);
            retain.push(buf_p);
            read_from_a = !read_from_a;
        }

        // Result is in whichever buffer was last written to
        let total_ns = MetalContext::submit_batch(cmd, &retain)?;
        let result_buf = if read_from_a { &buf_a } else { &buf_b };
        let result = MetalContext::read_buffer(result_buf, n);
        Ok((result, total_ns))
    }

    /// Run inverse Circle NTT (GS-DIF, out-of-place) on GPU.
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

        // Allocate two device buffers for ping-pong
        let buf_a = self.ctx.buffer_from_slice(input)?;
        let buf_b = self.ctx.alloc_buffer(std::mem::size_of_val(input))?;
        let cmd = self.ctx.begin_batch();
        let mut retain = Vec::new();

        let tile_log = log_n.min(MAX_TILE_LOG);

        // Track which buffer is current input vs output
        let mut read_from_a = true;

        // ── Phase 1: threadgroup stages (small strides) ────────────────
        // Inverse: layers from 0 up to (tile_log-1)
        if tile_log > 0 {
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

            // params: [tile_log, num_layers, start_layer, offsets...]
            let mut params: Vec<u32> =
                vec![tile_log as u32, num_tg_layers as u32, start_layer as u32];
            params.extend(tw_offsets);
            let buf_p = self.ctx.buffer_from_slice(&params)?;

            let tile_size = 1usize << tile_log;
            let num_tiles = n / tile_size;
            let max_tg_threads =
                MetalContext::max_threads_per_threadgroup(&self.inverse_tg_pipeline) as u64;
            let threads = max_tg_threads.min(tile_size as u64 / 2).max(1);

            let tg = MTLSize::new(num_tiles as u64, 1, 1);
            let tpg = MTLSize::new(threads, 1, 1);

            let (cur_in, cur_out) = if read_from_a {
                (&buf_a, &buf_b)
            } else {
                (&buf_b, &buf_a)
            };

            MetalContext::encode_dispatch(
                cmd,
                &self.inverse_tg_pipeline,
                &[cur_in, cur_out, &buf_tw, &buf_p],
                tg,
                tpg,
            );
            retain.push(buf_tw);
            retain.push(buf_p);
            read_from_a = !read_from_a;
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

            let tg = MTLSize::new(num_butterflies.div_ceil(tpg_width), 1, 1);
            let tpg = MTLSize::new(tpg_width.min(num_butterflies), 1, 1);

            let (cur_in, cur_out) = if read_from_a {
                (&buf_a, &buf_b)
            } else {
                (&buf_b, &buf_a)
            };

            MetalContext::encode_dispatch(
                cmd,
                &self.inverse_device_pipeline,
                &[cur_in, cur_out, &buf_tw, &buf_p],
                tg,
                tpg,
            );
            retain.push(buf_tw);
            retain.push(buf_p);
            read_from_a = !read_from_a;
        }

        // ── Normalize: multiply all elements by inv_n ──────────────────
        // Normalize is in-place on whichever buffer holds the result
        let result_buf = if read_from_a { &buf_a } else { &buf_b };
        let inv_n = M31::reduce(n as u64).inv();
        self.ctx.encode_normalize(
            cmd,
            &mut retain,
            &self.normalize_pipeline,
            result_buf,
            n,
            inv_n,
        )?;

        let total_ns = MetalContext::submit_batch(cmd, &retain)?;
        let result = MetalContext::read_buffer(result_buf, n);
        Ok((result, total_ns))
    }
}

impl NttBackend<M31> for MetalStockhamR2 {
    fn name(&self) -> &str {
        "metal-stockham-r2"
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

    fn init() -> Option<MetalStockhamR2> {
        try_init_metal(|p| MetalStockhamR2::new(p))
    }

    #[test]
    fn test_forward_matches_cpu() {
        let gpu = match init() {
            Some(g) => g,
            None => return,
        };
        assert_forward_matches_cpu(&gpu, &[4, 16, 256, 1024, 4096, 8192, 16384]);
    }

    #[test]
    fn test_roundtrip() {
        let gpu = match init() {
            Some(g) => g,
            None => return,
        };
        assert_roundtrip(&gpu, &[4, 256, 1024, 8192, 16384]);
    }

    #[test]
    fn test_edge_cases() {
        let gpu = match init() {
            Some(g) => g,
            None => return,
        };
        assert_edge_cases(&gpu);
        assert_inverse_edge_cases(&gpu);
    }

    #[test]
    fn test_pointwise_mul() {
        let gpu = match init() {
            Some(g) => g,
            None => return,
        };
        assert_pointwise_mul(&gpu);
    }

    #[test]
    fn test_size2_forward_and_roundtrip() {
        let gpu = match init() {
            Some(g) => g,
            None => return,
        };
        assert_forward_matches_cpu(&gpu, &[2]);
        assert_roundtrip(&gpu, &[2]);
    }

    // ── V3-specific: TG tile boundary at MAX_TILE_LOG=12 ────────────────

    #[test]
    fn test_roundtrip_size4096_tg_boundary() {
        // Exactly one TG tile (MAX_TILE_LOG=12) — no device phase
        let gpu = match init() {
            Some(g) => g,
            None => return,
        };
        assert_roundtrip(&gpu, &[4096]);
    }
}
