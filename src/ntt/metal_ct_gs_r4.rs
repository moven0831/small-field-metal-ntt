//! Variant 4: CT-DIT/GS-DIF radix-4 NTT on Metal GPU — in-place with threadgroup memory.
//!
//! Each radix-4 butterfly replaces two radix-2 stages, halving the barrier count.
//! When log_n is odd, the final stage falls back to radix-2.
//!
//! For a 2^20 NTT:
//! - Threadgroup phase: 6 radix-4 + 1 radix-2 = 7 barriers (vs V2's 13)
//! - Device phase: 3 radix-4 + 1 radix-2 = 4 dispatches (vs V2's 7)
//! - Total: 5 dispatches (vs V2's 8)
//!
//! Forward NTT uses CT-DIT radix-4 (high-to-low layer pairs).
//! Inverse NTT uses GS-DIF radix-4 (low-to-high layer pairs) + normalization.

use crate::field::m31::M31;
use crate::field::Field;
use crate::gpu::MetalContext;
use crate::ntt::twiddles::TwiddleCache;
use crate::ntt::{NttBackend, NttError};
use metal::*;
use std::path::Path;

const MAX_TILE_LOG: usize = 13;

pub struct MetalCtGsR4 {
    ctx: MetalContext,
    forward_tg_pipeline: ComputePipelineState,
    r4_device_pipeline: ComputePipelineState,
    r2_device_pipeline: ComputePipelineState,
    inverse_tg_pipeline: ComputePipelineState,
    r4_device_inv_pipeline: ComputePipelineState,
    r2_device_inv_pipeline: ComputePipelineState,
    normalize_pipeline: ComputePipelineState,
    twiddle_cache: TwiddleCache,
}

#[allow(clippy::too_many_arguments)]
impl MetalCtGsR4 {
    pub fn new(shader_dir: &Path) -> Result<Self, NttError> {
        let ctx = MetalContext::new(shader_dir)?;
        let forward_tg = ctx.make_pipeline("ct_gs_r4_forward_tg")?;
        let r4_dev = ctx.make_pipeline("ct_gs_r4_butterfly_device")?;
        let r2_dev = ctx.make_pipeline("ct_gs_r4_butterfly_device_r2")?;
        let inverse_tg = ctx.make_pipeline("ct_gs_r4_inverse_tg")?;
        let r4_dev_inv = ctx.make_pipeline("ct_gs_r4_butterfly_device_inv")?;
        let r2_dev_inv = ctx.make_pipeline("ct_gs_r4_r2_device_inv")?;
        let normalize = ctx.make_pipeline("ct_gs_r4_normalize")?;

        Ok(MetalCtGsR4 {
            ctx,
            forward_tg_pipeline: forward_tg,
            r4_device_pipeline: r4_dev,
            r2_device_pipeline: r2_dev,
            inverse_tg_pipeline: inverse_tg,
            r4_device_inv_pipeline: r4_dev_inv,
            r2_device_inv_pipeline: r2_dev_inv,
            normalize_pipeline: normalize,
            twiddle_cache: crate::ntt::twiddles::new_m31_twiddle_cache(),
        })
    }

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

        let tile_log = log_n.min(MAX_TILE_LOG);
        let cmd = self.ctx.begin_batch();
        let mut retain = Vec::new();

        // ── Device-memory phase: layers (log_n-1) down to tile_log ─────
        let num_device_layers = log_n - tile_log;
        if num_device_layers > 0 {
            self.encode_device_phase(cmd, &mut retain, &buf_data, &twiddles, n, log_n, tile_log)?;
        }

        // ── Threadgroup phase: layers (tile_log-1) down to 0 ───────────
        if tile_log > 0 {
            self.encode_threadgroup_phase(cmd, &mut retain, &buf_data, &twiddles, n, tile_log)?;
        }

        let total_ns = MetalContext::submit_batch(cmd, &retain)?;
        let result = MetalContext::read_buffer(&buf_data, n);
        Ok((result, total_ns))
    }

    fn encode_device_phase(
        &self,
        cmd: &CommandBufferRef,
        retain: &mut Vec<Buffer>,
        buf_data: &Buffer,
        twiddles: &[Vec<M31>],
        n: usize,
        log_n: usize,
        tile_log: usize,
    ) -> Result<(), NttError> {
        let num_device_layers = log_n - tile_log;
        let num_r4 = num_device_layers / 2;
        let has_r2 = num_device_layers % 2 == 1;

        let mut current_layer = log_n - 1;

        // Radix-4 paired dispatches
        for _ in 0..num_r4 {
            let k = current_layer;
            let outer = 1usize << k;
            let inner = 1usize << (k - 1);
            self.ctx.encode_butterfly_r4(
                cmd,
                retain,
                &self.r4_device_pipeline,
                buf_data,
                &twiddles[k],
                &twiddles[k - 1],
                outer,
                inner,
                n,
            )?;
            current_layer -= 2;
        }

        // Final radix-2 if odd
        if has_r2 {
            let k = current_layer;
            let stride = 1usize << k;
            self.ctx.encode_butterfly_r2(
                cmd,
                retain,
                &self.r2_device_pipeline,
                buf_data,
                &twiddles[k],
                stride,
                n,
            )?;
        }

        Ok(())
    }

    fn encode_threadgroup_phase(
        &self,
        cmd: &CommandBufferRef,
        retain: &mut Vec<Buffer>,
        buf_data: &Buffer,
        twiddles: &[Vec<M31>],
        n: usize,
        tile_log: usize,
    ) -> Result<(), NttError> {
        let num_tg_layers = tile_log;
        let num_r4 = num_tg_layers / 2;
        let has_final_r2 = num_tg_layers % 2 == 1;
        let start_layer = tile_log - 1;

        // Flatten twiddles for all threadgroup stages
        let mut flat_tw = Vec::new();
        let mut tw_offsets: Vec<u32> = Vec::new();

        // Radix-4 stages: pairs (start_layer, start_layer-1), (start_layer-2, start_layer-3), ...
        for s in 0..num_r4 {
            let k = start_layer - 2 * s;
            // Outer twiddles (layer k)
            tw_offsets.push(flat_tw.len() as u32);
            flat_tw.extend(twiddles[k].iter().map(|m| m.0));
            // Inner twiddles (layer k-1)
            tw_offsets.push(flat_tw.len() as u32);
            flat_tw.extend(twiddles[k - 1].iter().map(|m| m.0));
        }

        // Final radix-2 stage (layer 0 if odd)
        if has_final_r2 {
            tw_offsets.push(flat_tw.len() as u32);
            flat_tw.extend(twiddles[0].iter().map(|m| m.0));
        }

        let buf_tw = self.ctx.buffer_from_slice(&flat_tw)?;

        // params: [tile_log, num_r4, has_final_r2, offsets...]
        let mut params: Vec<u32> = vec![tile_log as u32, num_r4 as u32, has_final_r2 as u32];
        params.extend(tw_offsets);
        let buf_p = self.ctx.buffer_from_slice(&params)?;

        let tile_size = 1usize << tile_log;
        let num_tiles = n / tile_size;
        let max_tg_threads =
            MetalContext::max_threads_per_threadgroup(&self.forward_tg_pipeline) as u64;
        let threads = max_tg_threads.min(tile_size as u64 / 4).max(1);

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

        // ── Threadgroup phase: layers 0 up to (tile_log-1) ────────────
        if tile_log > 0 {
            self.encode_threadgroup_inverse(cmd, &mut retain, &buf_data, &itwiddles, n, tile_log)?;
        }

        // ── Device-memory phase: layers tile_log up to (log_n-1) ──────
        let num_device_layers = log_n - tile_log;
        if num_device_layers > 0 {
            self.encode_device_inverse(
                cmd,
                &mut retain,
                &buf_data,
                &itwiddles,
                n,
                log_n,
                tile_log,
            )?;
        }

        // ── Normalize: multiply all elements by n^{-1} ───────────────
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

    fn encode_device_inverse(
        &self,
        cmd: &CommandBufferRef,
        retain: &mut Vec<Buffer>,
        buf_data: &Buffer,
        itwiddles: &[Vec<M31>],
        n: usize,
        log_n: usize,
        tile_log: usize,
    ) -> Result<(), NttError> {
        let num_device_layers = log_n - tile_log;
        let has_r2 = num_device_layers % 2 == 1;
        let num_r4 = num_device_layers / 2;

        // Inverse processes layers from low to high: tile_log, tile_log+1, ...
        let mut current_layer = tile_log;

        // If odd number of device layers, start with radix-2 at lowest layer
        if has_r2 {
            let k = current_layer;
            let stride = 1usize << k;
            self.ctx.encode_butterfly_r2(
                cmd,
                retain,
                &self.r2_device_inv_pipeline,
                buf_data,
                &itwiddles[k],
                stride,
                n,
            )?;
            current_layer += 1;
        }

        // Radix-4 paired dispatches (from low to high)
        for _ in 0..num_r4 {
            let k_inner = current_layer;
            let k_outer = current_layer + 1;
            let outer = 1usize << k_outer;
            let inner = 1usize << k_inner;
            self.ctx.encode_butterfly_r4(
                cmd,
                retain,
                &self.r4_device_inv_pipeline,
                buf_data,
                &itwiddles[k_outer],
                &itwiddles[k_inner],
                outer,
                inner,
                n,
            )?;
            current_layer += 2;
        }

        Ok(())
    }

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
        let has_initial_r2 = num_tg_layers % 2 == 1;
        let num_r4 = num_tg_layers / 2;

        // Flatten twiddles for all threadgroup stages (processing order: low to high)
        let mut flat_tw = Vec::new();
        let mut tw_offsets: Vec<u32> = Vec::new();

        let mut current_layer = 0usize;

        // Initial radix-2 at layer 0 if odd
        if has_initial_r2 {
            tw_offsets.push(flat_tw.len() as u32);
            flat_tw.extend(itwiddles[0].iter().map(|m| m.0));
            current_layer = 1;
        }

        // Radix-4 stages: pairs (current_layer, current_layer+1), ...
        for s in 0..num_r4 {
            let k_inner = current_layer + 2 * s;
            let k_outer = k_inner + 1;
            // Inner twiddles (layer k_inner)
            tw_offsets.push(flat_tw.len() as u32);
            flat_tw.extend(itwiddles[k_inner].iter().map(|m| m.0));
            // Outer twiddles (layer k_outer)
            tw_offsets.push(flat_tw.len() as u32);
            flat_tw.extend(itwiddles[k_outer].iter().map(|m| m.0));
        }

        let buf_tw = self.ctx.buffer_from_slice(&flat_tw)?;

        // params: [tile_log, num_r4, has_initial_r2, offsets...]
        let mut params: Vec<u32> = vec![tile_log as u32, num_r4 as u32, has_initial_r2 as u32];
        params.extend(tw_offsets);
        let buf_p = self.ctx.buffer_from_slice(&params)?;

        let tile_size = 1usize << tile_log;
        let num_tiles = n / tile_size;
        let max_tg_threads =
            MetalContext::max_threads_per_threadgroup(&self.inverse_tg_pipeline) as u64;
        let threads = max_tg_threads.min(tile_size as u64 / 4).max(1);

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

impl NttBackend<M31> for MetalCtGsR4 {
    fn name(&self) -> &str {
        "metal-ct-gs-r4"
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

    fn init() -> Option<MetalCtGsR4> {
        try_init_metal(|p| MetalCtGsR4::new(p))
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
        assert_roundtrip(&gpu, &[4, 16, 256, 1024, 8192, 16384]);
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
    fn test_size2_forward() {
        let gpu = match init() {
            Some(g) => g,
            None => return,
        };
        assert_forward_matches_cpu(&gpu, &[2]);
    }

    // ── V4-specific: odd/even log_n radix-4 coverage ────────────────────

    #[test]
    fn test_forward_odd_log_n() {
        // Odd log_n exercises radix-4 + final radix-2 fallback
        let gpu = match init() {
            Some(g) => g,
            None => return,
        };
        assert_forward_matches_cpu(&gpu, &[8, 32, 2048]);
    }

    #[test]
    fn test_forward_device_phase_coverage() {
        // log_n=15 (odd): 1 r4 device dispatch
        // log_n=16 (even): 1 r4 + 1 r2 device dispatch
        let gpu = match init() {
            Some(g) => g,
            None => return,
        };
        assert_forward_matches_cpu(&gpu, &[32768, 65536]);
    }

    #[test]
    fn test_roundtrip_odd_device_layers() {
        // log_n=15 (odd, 2 device layers): exercises ct_gs_r4_butterfly_device_inv
        let gpu = match init() {
            Some(g) => g,
            None => return,
        };
        assert_roundtrip(&gpu, &[32768]);
    }
}
