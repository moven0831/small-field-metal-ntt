//! BabyBear V4: CT-DIT/GS-DIF radix-4 NTT on Metal GPU.
//!
//! Each radix-4 butterfly replaces two radix-2 stages, halving the barrier count.
//! When log_n is odd, the final/initial stage falls back to radix-2.
//!
//! Key difference from M31 radix-4: standard NTT uses position-indexed twiddles.
//! The radix-4 butterfly needs 2 distinct outer twiddles + 1 shared inner twiddle
//! (vs M31's 1 outer + 2 inner).

use crate::field::babybear::BabyBear;
use crate::field::Field;
use crate::gpu::MetalContext;
use crate::ntt::bb_twiddles::BbTwiddleCache;
use crate::ntt::{NttBackend, NttError};
use metal::*;
use std::path::Path;

const MAX_TILE_LOG: usize = 13;

pub struct BbMetalCtGsR4 {
    ctx: MetalContext,
    forward_tg_pipeline: ComputePipelineState,
    r4_device_pipeline: ComputePipelineState,
    r2_device_pipeline: ComputePipelineState,
    inverse_tg_pipeline: ComputePipelineState,
    r4_device_inv_pipeline: ComputePipelineState,
    r2_device_inv_pipeline: ComputePipelineState,
    normalize_pipeline: ComputePipelineState,
    // Batch pipelines
    batch_forward_tg_pipeline: ComputePipelineState,
    batch_inverse_tg_pipeline: ComputePipelineState,
    batch_r4_device_pipeline: ComputePipelineState,
    batch_r4_device_inv_pipeline: ComputePipelineState,
    batch_r2_device_pipeline: ComputePipelineState,
    batch_r2_device_inv_pipeline: ComputePipelineState,
    _batch_normalize_pipeline: ComputePipelineState,
    // LDE helper pipelines
    fused_norm_zeropad_shift_pipeline: ComputePipelineState,
    twiddle_cache: BbTwiddleCache,
}

#[allow(clippy::too_many_arguments)]
impl BbMetalCtGsR4 {
    pub fn new(shader_dir: &Path) -> Result<Self, NttError> {
        let ctx = MetalContext::new(shader_dir)?;
        let forward_tg = ctx.make_pipeline("bb_r4_forward_tg")?;
        let r4_dev = ctx.make_pipeline("bb_r4_butterfly_device")?;
        let r2_dev = ctx.make_pipeline("bb_r4_butterfly_device_r2")?;
        let inverse_tg = ctx.make_pipeline("bb_r4_inverse_tg")?;
        let r4_dev_inv = ctx.make_pipeline("bb_r4_butterfly_device_inv")?;
        let r2_dev_inv = ctx.make_pipeline("bb_r4_r2_device_inv")?;
        let normalize = ctx.make_pipeline("bb_r4_normalize")?;

        // Batch pipelines
        let batch_forward_tg = ctx.make_pipeline("bb_r4_batch_forward_tg")?;
        let batch_inverse_tg = ctx.make_pipeline("bb_r4_batch_inverse_tg")?;
        let batch_r4_dev = ctx.make_pipeline("bb_r4_batch_butterfly_device")?;
        let batch_r4_dev_inv = ctx.make_pipeline("bb_r4_batch_butterfly_device_inv")?;
        let batch_r2_dev = ctx.make_pipeline("bb_r4_batch_butterfly_device_r2")?;
        let batch_r2_dev_inv = ctx.make_pipeline("bb_r4_batch_r2_device_inv")?;
        let batch_normalize = ctx.make_pipeline("bb_r4_batch_normalize")?;

        // LDE helpers
        let fused_norm_zeropad_shift = ctx.make_pipeline("bb_fused_normalize_zeropad_shift")?;

        Ok(Self {
            ctx,
            forward_tg_pipeline: forward_tg,
            r4_device_pipeline: r4_dev,
            r2_device_pipeline: r2_dev,
            inverse_tg_pipeline: inverse_tg,
            r4_device_inv_pipeline: r4_dev_inv,
            r2_device_inv_pipeline: r2_dev_inv,
            normalize_pipeline: normalize,
            batch_forward_tg_pipeline: batch_forward_tg,
            batch_inverse_tg_pipeline: batch_inverse_tg,
            batch_r4_device_pipeline: batch_r4_dev,
            batch_r4_device_inv_pipeline: batch_r4_dev_inv,
            batch_r2_device_pipeline: batch_r2_dev,
            batch_r2_device_inv_pipeline: batch_r2_dev_inv,
            _batch_normalize_pipeline: batch_normalize,
            fused_norm_zeropad_shift_pipeline: fused_norm_zeropad_shift,
            twiddle_cache: crate::ntt::bb_twiddles::new_bb_twiddle_cache(),
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
        if log_n == 0 {
            return Ok((input.to_vec(), 0));
        }

        let twiddles = self.twiddle_cache.forward(log_n as u32);
        let buf_data = self.ctx.buffer_from_slice(input)?;
        let tile_log = log_n.min(MAX_TILE_LOG);
        let cmd = self.ctx.begin_batch();
        let mut retain = Vec::new();

        // Device-memory phase: layers (log_n-1) down to tile_log
        let num_device_layers = log_n - tile_log;
        if num_device_layers > 0 {
            self.encode_device_forward(cmd, &mut retain, &buf_data, &twiddles, n, log_n, tile_log)?;
        }

        // Threadgroup phase: layers (tile_log-1) down to 0
        if tile_log > 0 {
            self.encode_threadgroup_forward(cmd, &mut retain, &buf_data, &twiddles, n, tile_log)?;
        }

        let total_ns = MetalContext::submit_batch(cmd, &retain)?;
        let result = MetalContext::read_buffer(&buf_data, n);
        Ok((result, total_ns))
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
        if log_n == 0 {
            return Ok((input.to_vec(), 0));
        }

        let itwiddles = self.twiddle_cache.inverse(log_n as u32);
        let buf_data = self.ctx.buffer_from_slice(input)?;
        let cmd = self.ctx.begin_batch();
        let mut retain = Vec::new();
        let tile_log = log_n.min(MAX_TILE_LOG);

        // Threadgroup phase: layers 0 up to (tile_log-1)
        if tile_log > 0 {
            self.encode_threadgroup_inverse(cmd, &mut retain, &buf_data, &itwiddles, n, tile_log)?;
        }

        // Device-memory phase: layers tile_log up to (log_n-1)
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

        // Normalize
        let inv_n = BabyBear::reduce(n as u64).inv();
        let params: Vec<u32> = vec![n as u32, inv_n.0];
        let buf_p = self.ctx.buffer_from_slice(&params)?;
        let max_tpg = MetalContext::max_threads_per_threadgroup(&self.normalize_pipeline) as u64;
        let (tg, tpg) = MetalContext::compute_grid_1d(n as u64, max_tpg.min(256));
        MetalContext::encode_dispatch(cmd, &self.normalize_pipeline, &[&buf_data, &buf_p], tg, tpg);
        retain.push(buf_p);

        let total_ns = MetalContext::submit_batch(cmd, &retain)?;
        let result = MetalContext::read_buffer(&buf_data, n);
        Ok((result, total_ns))
    }

    pub fn ctx(&self) -> &MetalContext {
        &self.ctx
    }

    // ── Batched NTT (SoA layout: data[col * n + row]) ───────────────────

    /// Encode batched inverse NTT without normalize (for fused LDE pipeline).
    pub fn encode_inverse_ntt_batch_no_normalize(
        &self,
        cmd: &CommandBufferRef,
        retain: &mut Vec<Buffer>,
        buf_data: &Buffer,
        log_n: usize,
        n: usize,
        batch_size: usize,
    ) -> Result<(), NttError> {
        if log_n == 0 {
            return Ok(());
        }
        let itwiddles = self.twiddle_cache.inverse(log_n as u32);
        let tile_log = log_n.min(MAX_TILE_LOG);

        // Threadgroup phase: layers 0 up to (tile_log-1)
        if tile_log > 0 {
            self.encode_batch_threadgroup_inverse(
                cmd, retain, buf_data, &itwiddles, n, tile_log, batch_size,
            )?;
        }

        // Device-memory phase: layers tile_log up to (log_n-1), radix-4
        let num_device_layers = log_n - tile_log;
        if num_device_layers > 0 {
            self.encode_batch_device_inverse(
                cmd, retain, buf_data, &itwiddles, n, log_n, tile_log, batch_size,
            )?;
        }

        Ok(())
    }

    /// Encode batched forward NTT on a pre-allocated buffer.
    pub fn encode_forward_ntt_batch(
        &self,
        cmd: &CommandBufferRef,
        retain: &mut Vec<Buffer>,
        buf_data: &Buffer,
        log_n: usize,
        n: usize,
        batch_size: usize,
    ) -> Result<(), NttError> {
        if log_n == 0 {
            return Ok(());
        }
        let twiddles = self.twiddle_cache.forward(log_n as u32);
        let tile_log = log_n.min(MAX_TILE_LOG);

        // Device-memory phase: layers (log_n-1) down to tile_log, radix-4
        let num_device_layers = log_n - tile_log;
        if num_device_layers > 0 {
            self.encode_batch_device_forward(
                cmd, retain, buf_data, &twiddles, n, log_n, tile_log, batch_size,
            )?;
        }

        // Threadgroup phase: layers (tile_log-1) down to 0
        if tile_log > 0 {
            self.encode_batch_threadgroup_forward(
                cmd, retain, buf_data, &twiddles, n, tile_log, batch_size,
            )?;
        }

        Ok(())
    }

    /// Encode fused normalize + zero-pad + coset-shift dispatch.
    pub fn encode_fused_norm_zeropad_shift(
        &self,
        cmd: &CommandBufferRef,
        retain: &mut Vec<Buffer>,
        buf_input: &Buffer,
        buf_output: &Buffer,
        shift_powers: &[u32],
        n_orig: usize,
        n_ext: usize,
        batch_size: usize,
        inv_n: BabyBear,
    ) -> Result<(), NttError> {
        let buf_shifts = self.ctx.buffer_from_slice(shift_powers)?;
        let params: Vec<u32> = vec![n_orig as u32, n_ext as u32, batch_size as u32, inv_n.0];
        let buf_p = self.ctx.buffer_from_slice(&params)?;

        let max_tpg =
            MetalContext::max_threads_per_threadgroup(&self.fused_norm_zeropad_shift_pipeline)
                as u64;
        let tpg_x = max_tpg.min(n_ext as u64).max(1);
        let tg_x = (n_ext as u64).div_ceil(tpg_x);

        MetalContext::encode_dispatch(
            cmd,
            &self.fused_norm_zeropad_shift_pipeline,
            &[buf_input, buf_output, &buf_shifts, &buf_p],
            MTLSize::new(tg_x, batch_size as u64, 1),
            MTLSize::new(tpg_x, 1, 1),
        );
        retain.push(buf_shifts);
        retain.push(buf_p);
        Ok(())
    }

    // ── Batch device phase encoding ─────────────────────────────────────

    fn encode_batch_device_forward(
        &self,
        cmd: &CommandBufferRef,
        retain: &mut Vec<Buffer>,
        buf_data: &Buffer,
        twiddles: &[Vec<BabyBear>],
        n: usize,
        log_n: usize,
        tile_log: usize,
        batch_size: usize,
    ) -> Result<(), NttError> {
        let num_device_layers = log_n - tile_log;
        let num_r4 = num_device_layers / 2;
        let has_r2 = num_device_layers % 2 == 1;
        let mut current_layer = log_n - 1;

        for _ in 0..num_r4 {
            let k = current_layer;
            self.encode_batch_r4_dispatch(
                cmd,
                retain,
                &self.batch_r4_device_pipeline,
                buf_data,
                &twiddles[k],
                &twiddles[k - 1],
                1 << k,
                1 << (k - 1),
                n,
                batch_size,
            )?;
            current_layer -= 2;
        }

        if has_r2 {
            self.encode_batch_r2_dispatch(
                cmd,
                retain,
                &self.batch_r2_device_pipeline,
                buf_data,
                &twiddles[current_layer],
                1 << current_layer,
                n,
                batch_size,
            )?;
        }
        Ok(())
    }

    fn encode_batch_device_inverse(
        &self,
        cmd: &CommandBufferRef,
        retain: &mut Vec<Buffer>,
        buf_data: &Buffer,
        itwiddles: &[Vec<BabyBear>],
        n: usize,
        log_n: usize,
        tile_log: usize,
        batch_size: usize,
    ) -> Result<(), NttError> {
        let num_device_layers = log_n - tile_log;
        let has_r2 = num_device_layers % 2 == 1;
        let num_r4 = num_device_layers / 2;
        let mut current_layer = tile_log;

        if has_r2 {
            self.encode_batch_r2_dispatch(
                cmd,
                retain,
                &self.batch_r2_device_inv_pipeline,
                buf_data,
                &itwiddles[current_layer],
                1 << current_layer,
                n,
                batch_size,
            )?;
            current_layer += 1;
        }

        for _ in 0..num_r4 {
            let k_inner = current_layer;
            let k_outer = current_layer + 1;
            self.encode_batch_r4_dispatch(
                cmd,
                retain,
                &self.batch_r4_device_inv_pipeline,
                buf_data,
                &itwiddles[k_outer],
                &itwiddles[k_inner],
                1 << k_outer,
                1 << k_inner,
                n,
                batch_size,
            )?;
            current_layer += 2;
        }
        Ok(())
    }

    // ── Batch threadgroup phase encoding ─────────────────────────────────

    fn encode_batch_threadgroup_forward(
        &self,
        cmd: &CommandBufferRef,
        retain: &mut Vec<Buffer>,
        buf_data: &Buffer,
        twiddles: &[Vec<BabyBear>],
        n: usize,
        tile_log: usize,
        batch_size: usize,
    ) -> Result<(), NttError> {
        let num_tg_layers = tile_log;
        let num_r4 = num_tg_layers / 2;
        let has_final_r2 = num_tg_layers % 2 == 1;
        let start_layer = tile_log - 1;

        let mut flat_tw = Vec::new();
        let mut tw_offsets: Vec<u32> = Vec::new();

        for s in 0..num_r4 {
            let k = start_layer - 2 * s;
            tw_offsets.push(flat_tw.len() as u32);
            flat_tw.extend(twiddles[k].iter().map(|m| m.0));
            tw_offsets.push(flat_tw.len() as u32);
            flat_tw.extend(twiddles[k - 1].iter().map(|m| m.0));
        }

        if has_final_r2 {
            tw_offsets.push(flat_tw.len() as u32);
            flat_tw.extend(twiddles[0].iter().map(|m| m.0));
        }

        let buf_tw = self.ctx.buffer_from_slice(&flat_tw)?;
        let tile_size = 1usize << tile_log;
        let num_tiles_per_col = n / tile_size;

        // params: [n, tile_log, num_r4, has_final_r2, num_tiles_per_col, tw_offsets...]
        let mut params: Vec<u32> = vec![
            n as u32,
            tile_log as u32,
            num_r4 as u32,
            has_final_r2 as u32,
            num_tiles_per_col as u32,
        ];
        params.extend(tw_offsets);
        let buf_p = self.ctx.buffer_from_slice(&params)?;

        let total_threadgroups = (num_tiles_per_col * batch_size) as u64;
        let max_tg_threads =
            MetalContext::max_threads_per_threadgroup(&self.batch_forward_tg_pipeline) as u64;
        let threads = max_tg_threads.min(tile_size as u64 / 4).max(1);

        MetalContext::encode_dispatch(
            cmd,
            &self.batch_forward_tg_pipeline,
            &[buf_data, &buf_tw, &buf_p],
            MTLSize::new(total_threadgroups, 1, 1),
            MTLSize::new(threads, 1, 1),
        );
        retain.push(buf_tw);
        retain.push(buf_p);
        Ok(())
    }

    fn encode_batch_threadgroup_inverse(
        &self,
        cmd: &CommandBufferRef,
        retain: &mut Vec<Buffer>,
        buf_data: &Buffer,
        itwiddles: &[Vec<BabyBear>],
        n: usize,
        tile_log: usize,
        batch_size: usize,
    ) -> Result<(), NttError> {
        let num_tg_layers = tile_log;
        let has_initial_r2 = num_tg_layers % 2 == 1;
        let num_r4 = num_tg_layers / 2;

        let mut flat_tw = Vec::new();
        let mut tw_offsets: Vec<u32> = Vec::new();
        let mut current_layer = 0usize;

        if has_initial_r2 {
            tw_offsets.push(flat_tw.len() as u32);
            flat_tw.extend(itwiddles[0].iter().map(|m| m.0));
            current_layer = 1;
        }

        for s in 0..num_r4 {
            let k_inner = current_layer + 2 * s;
            let k_outer = k_inner + 1;
            tw_offsets.push(flat_tw.len() as u32);
            flat_tw.extend(itwiddles[k_inner].iter().map(|m| m.0));
            tw_offsets.push(flat_tw.len() as u32);
            flat_tw.extend(itwiddles[k_outer].iter().map(|m| m.0));
        }

        let buf_tw = self.ctx.buffer_from_slice(&flat_tw)?;
        let tile_size = 1usize << tile_log;
        let num_tiles_per_col = n / tile_size;

        // params: [n, tile_log, num_r4, has_initial_r2, num_tiles_per_col, tw_offsets...]
        let mut params: Vec<u32> = vec![
            n as u32,
            tile_log as u32,
            num_r4 as u32,
            has_initial_r2 as u32,
            num_tiles_per_col as u32,
        ];
        params.extend(tw_offsets);
        let buf_p = self.ctx.buffer_from_slice(&params)?;

        let total_threadgroups = (num_tiles_per_col * batch_size) as u64;
        let max_tg_threads =
            MetalContext::max_threads_per_threadgroup(&self.batch_inverse_tg_pipeline) as u64;
        let threads = max_tg_threads.min(tile_size as u64 / 4).max(1);

        MetalContext::encode_dispatch(
            cmd,
            &self.batch_inverse_tg_pipeline,
            &[buf_data, &buf_tw, &buf_p],
            MTLSize::new(total_threadgroups, 1, 1),
            MTLSize::new(threads, 1, 1),
        );
        retain.push(buf_tw);
        retain.push(buf_p);
        Ok(())
    }

    // ── Batch dispatch helpers ───────────────────────────────────────────

    fn encode_batch_r4_dispatch(
        &self,
        cmd: &CommandBufferRef,
        retain: &mut Vec<Buffer>,
        pipeline: &ComputePipelineState,
        buf_data: &Buffer,
        tw_outer: &[BabyBear],
        tw_inner: &[BabyBear],
        outer: usize,
        inner: usize,
        n: usize,
        batch_size: usize,
    ) -> Result<(), NttError> {
        let tw_o: Vec<u32> = tw_outer.iter().map(|m| m.0).collect();
        let tw_i: Vec<u32> = tw_inner.iter().map(|m| m.0).collect();
        let buf_tw_o = self.ctx.buffer_from_slice(&tw_o)?;
        let buf_tw_i = self.ctx.buffer_from_slice(&tw_i)?;
        let params: Vec<u32> = vec![outer as u32, inner as u32, n as u32, batch_size as u32];
        let buf_p = self.ctx.buffer_from_slice(&params)?;

        let num_butterflies = (n / 4) as u64;
        let max_tpg = MetalContext::max_threads_per_threadgroup(pipeline) as u64;
        let tpg_x = max_tpg.min(256).min(num_butterflies);
        let tg_x = num_butterflies.div_ceil(tpg_x);

        MetalContext::encode_dispatch(
            cmd,
            pipeline,
            &[buf_data, &buf_tw_o, &buf_tw_i, &buf_p],
            MTLSize::new(tg_x, batch_size as u64, 1),
            MTLSize::new(tpg_x, 1, 1),
        );
        retain.push(buf_tw_o);
        retain.push(buf_tw_i);
        retain.push(buf_p);
        Ok(())
    }

    fn encode_batch_r2_dispatch(
        &self,
        cmd: &CommandBufferRef,
        retain: &mut Vec<Buffer>,
        pipeline: &ComputePipelineState,
        buf_data: &Buffer,
        twiddles: &[BabyBear],
        stride: usize,
        n: usize,
        batch_size: usize,
    ) -> Result<(), NttError> {
        let tw_data: Vec<u32> = twiddles.iter().map(|m| m.0).collect();
        let buf_tw = self.ctx.buffer_from_slice(&tw_data)?;
        let params: Vec<u32> = vec![stride as u32, n as u32, batch_size as u32];
        let buf_p = self.ctx.buffer_from_slice(&params)?;

        let num_butterflies = (n / 2) as u64;
        let max_tpg = MetalContext::max_threads_per_threadgroup(pipeline) as u64;
        let tpg_x = max_tpg.min(256).min(num_butterflies);
        let tg_x = num_butterflies.div_ceil(tpg_x);

        MetalContext::encode_dispatch(
            cmd,
            pipeline,
            &[buf_data, &buf_tw, &buf_p],
            MTLSize::new(tg_x, batch_size as u64, 1),
            MTLSize::new(tpg_x, 1, 1),
        );
        retain.push(buf_tw);
        retain.push(buf_p);
        Ok(())
    }

    // ── Single-column device phase encoding ─────────────────────────────

    fn encode_device_forward(
        &self,
        cmd: &CommandBufferRef,
        retain: &mut Vec<Buffer>,
        buf_data: &Buffer,
        twiddles: &[Vec<BabyBear>],
        n: usize,
        log_n: usize,
        tile_log: usize,
    ) -> Result<(), NttError> {
        let num_device_layers = log_n - tile_log;
        let num_r4 = num_device_layers / 2;
        let has_r2 = num_device_layers % 2 == 1;
        let mut current_layer = log_n - 1;

        for _ in 0..num_r4 {
            let k = current_layer;
            self.encode_r4_dispatch(
                cmd,
                retain,
                &self.r4_device_pipeline,
                buf_data,
                &twiddles[k],
                &twiddles[k - 1],
                1 << k,
                1 << (k - 1),
                n,
            )?;
            current_layer -= 2;
        }

        if has_r2 {
            self.encode_r2_dispatch(
                cmd,
                retain,
                &self.r2_device_pipeline,
                buf_data,
                &twiddles[current_layer],
                1 << current_layer,
                n,
            )?;
        }
        Ok(())
    }

    fn encode_device_inverse(
        &self,
        cmd: &CommandBufferRef,
        retain: &mut Vec<Buffer>,
        buf_data: &Buffer,
        itwiddles: &[Vec<BabyBear>],
        n: usize,
        log_n: usize,
        tile_log: usize,
    ) -> Result<(), NttError> {
        let num_device_layers = log_n - tile_log;
        let has_r2 = num_device_layers % 2 == 1;
        let num_r4 = num_device_layers / 2;
        let mut current_layer = tile_log;

        if has_r2 {
            self.encode_r2_dispatch(
                cmd,
                retain,
                &self.r2_device_inv_pipeline,
                buf_data,
                &itwiddles[current_layer],
                1 << current_layer,
                n,
            )?;
            current_layer += 1;
        }

        for _ in 0..num_r4 {
            let k_inner = current_layer;
            let k_outer = current_layer + 1;
            self.encode_r4_dispatch(
                cmd,
                retain,
                &self.r4_device_inv_pipeline,
                buf_data,
                &itwiddles[k_outer],
                &itwiddles[k_inner],
                1 << k_outer,
                1 << k_inner,
                n,
            )?;
            current_layer += 2;
        }
        Ok(())
    }

    // ── Threadgroup phase encoding ───────────────────────────────────────

    fn encode_threadgroup_forward(
        &self,
        cmd: &CommandBufferRef,
        retain: &mut Vec<Buffer>,
        buf_data: &Buffer,
        twiddles: &[Vec<BabyBear>],
        n: usize,
        tile_log: usize,
    ) -> Result<(), NttError> {
        let num_tg_layers = tile_log;
        let num_r4 = num_tg_layers / 2;
        let has_final_r2 = num_tg_layers % 2 == 1;
        let start_layer = tile_log - 1;

        let mut flat_tw = Vec::new();
        let mut tw_offsets: Vec<u32> = Vec::new();

        for s in 0..num_r4 {
            let k = start_layer - 2 * s;
            tw_offsets.push(flat_tw.len() as u32);
            flat_tw.extend(twiddles[k].iter().map(|m| m.0));
            tw_offsets.push(flat_tw.len() as u32);
            flat_tw.extend(twiddles[k - 1].iter().map(|m| m.0));
        }

        if has_final_r2 {
            tw_offsets.push(flat_tw.len() as u32);
            flat_tw.extend(twiddles[0].iter().map(|m| m.0));
        }

        let buf_tw = self.ctx.buffer_from_slice(&flat_tw)?;

        // params: [n, tile_log, num_r4, has_final_r2, tw_offsets...]
        let mut params: Vec<u32> = vec![
            n as u32,
            tile_log as u32,
            num_r4 as u32,
            has_final_r2 as u32,
        ];
        params.extend(tw_offsets);
        let buf_p = self.ctx.buffer_from_slice(&params)?;

        let tile_size = 1usize << tile_log;
        let num_tiles = n / tile_size;
        let max_tg_threads =
            MetalContext::max_threads_per_threadgroup(&self.forward_tg_pipeline) as u64;
        let threads = max_tg_threads.min(tile_size as u64 / 4).max(1);

        MetalContext::encode_dispatch(
            cmd,
            &self.forward_tg_pipeline,
            &[buf_data, &buf_tw, &buf_p],
            MTLSize::new(num_tiles as u64, 1, 1),
            MTLSize::new(threads, 1, 1),
        );
        retain.push(buf_tw);
        retain.push(buf_p);
        Ok(())
    }

    fn encode_threadgroup_inverse(
        &self,
        cmd: &CommandBufferRef,
        retain: &mut Vec<Buffer>,
        buf_data: &Buffer,
        itwiddles: &[Vec<BabyBear>],
        n: usize,
        tile_log: usize,
    ) -> Result<(), NttError> {
        let num_tg_layers = tile_log;
        let has_initial_r2 = num_tg_layers % 2 == 1;
        let num_r4 = num_tg_layers / 2;

        let mut flat_tw = Vec::new();
        let mut tw_offsets: Vec<u32> = Vec::new();
        let mut current_layer = 0usize;

        if has_initial_r2 {
            tw_offsets.push(flat_tw.len() as u32);
            flat_tw.extend(itwiddles[0].iter().map(|m| m.0));
            current_layer = 1;
        }

        for s in 0..num_r4 {
            let k_inner = current_layer + 2 * s;
            let k_outer = k_inner + 1;
            tw_offsets.push(flat_tw.len() as u32);
            flat_tw.extend(itwiddles[k_inner].iter().map(|m| m.0));
            tw_offsets.push(flat_tw.len() as u32);
            flat_tw.extend(itwiddles[k_outer].iter().map(|m| m.0));
        }

        let buf_tw = self.ctx.buffer_from_slice(&flat_tw)?;

        // params: [n, tile_log, num_r4, has_initial_r2, tw_offsets...]
        let mut params: Vec<u32> = vec![
            n as u32,
            tile_log as u32,
            num_r4 as u32,
            has_initial_r2 as u32,
        ];
        params.extend(tw_offsets);
        let buf_p = self.ctx.buffer_from_slice(&params)?;

        let tile_size = 1usize << tile_log;
        let num_tiles = n / tile_size;
        let max_tg_threads =
            MetalContext::max_threads_per_threadgroup(&self.inverse_tg_pipeline) as u64;
        let threads = max_tg_threads.min(tile_size as u64 / 4).max(1);

        MetalContext::encode_dispatch(
            cmd,
            &self.inverse_tg_pipeline,
            &[buf_data, &buf_tw, &buf_p],
            MTLSize::new(num_tiles as u64, 1, 1),
            MTLSize::new(threads, 1, 1),
        );
        retain.push(buf_tw);
        retain.push(buf_p);
        Ok(())
    }

    // ── Dispatch helpers ─────────────────────────────────────────────────

    fn encode_r4_dispatch(
        &self,
        cmd: &CommandBufferRef,
        retain: &mut Vec<Buffer>,
        pipeline: &ComputePipelineState,
        buf_data: &Buffer,
        tw_outer: &[BabyBear],
        tw_inner: &[BabyBear],
        outer: usize,
        inner: usize,
        n: usize,
    ) -> Result<(), NttError> {
        let tw_o: Vec<u32> = tw_outer.iter().map(|m| m.0).collect();
        let tw_i: Vec<u32> = tw_inner.iter().map(|m| m.0).collect();
        let buf_tw_o = self.ctx.buffer_from_slice(&tw_o)?;
        let buf_tw_i = self.ctx.buffer_from_slice(&tw_i)?;
        let params: Vec<u32> = vec![outer as u32, inner as u32, n as u32];
        let buf_p = self.ctx.buffer_from_slice(&params)?;

        let num_butterflies = (n / 4) as u64;
        let max_tpg = MetalContext::max_threads_per_threadgroup(pipeline) as u64;
        let (tg, tpg) = MetalContext::compute_grid_1d(num_butterflies, max_tpg.min(256));

        MetalContext::encode_dispatch(
            cmd,
            pipeline,
            &[buf_data, &buf_tw_o, &buf_tw_i, &buf_p],
            tg,
            tpg,
        );
        retain.push(buf_tw_o);
        retain.push(buf_tw_i);
        retain.push(buf_p);
        Ok(())
    }

    fn encode_r2_dispatch(
        &self,
        cmd: &CommandBufferRef,
        retain: &mut Vec<Buffer>,
        pipeline: &ComputePipelineState,
        buf_data: &Buffer,
        twiddles: &[BabyBear],
        stride: usize,
        n: usize,
    ) -> Result<(), NttError> {
        let tw_data: Vec<u32> = twiddles.iter().map(|m| m.0).collect();
        let buf_tw = self.ctx.buffer_from_slice(&tw_data)?;
        let params: Vec<u32> = vec![stride as u32, n as u32];
        let buf_p = self.ctx.buffer_from_slice(&params)?;

        let num_butterflies = (n / 2) as u64;
        let max_tpg = MetalContext::max_threads_per_threadgroup(pipeline) as u64;
        let (tg, tpg) = MetalContext::compute_grid_1d(num_butterflies, max_tpg.min(256));

        MetalContext::encode_dispatch(cmd, pipeline, &[buf_data, &buf_tw, &buf_p], tg, tpg);
        retain.push(buf_tw);
        retain.push(buf_p);
        Ok(())
    }
}

impl NttBackend<BabyBear> for BbMetalCtGsR4 {
    fn name(&self) -> &str {
        "bb-metal-ct-gs-r4"
    }

    fn forward_ntt(&self, data: &mut [BabyBear], _twiddles: &[BabyBear]) -> Result<(), NttError> {
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
            data[i] = BabyBear(*val);
        }
        Ok(())
    }

    fn inverse_ntt(&self, data: &mut [BabyBear], _twiddles: &[BabyBear]) -> Result<(), NttError> {
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
            data[i] = BabyBear(*val);
        }
        Ok(())
    }

    fn pointwise_mul(
        &self,
        a: &[BabyBear],
        b: &[BabyBear],
        out: &mut [BabyBear],
    ) -> Result<(), NttError> {
        if a.len() != b.len() || a.len() != out.len() {
            return Err(NttError::InvalidSize(a.len()));
        }
        for i in 0..a.len() {
            out[i] = a[i].mul(b[i]);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ntt::bb_cpu_reference::BbCpuReferenceBackend;
    use crate::ntt::test_utils::shader_dir;

    fn init() -> Option<BbMetalCtGsR4> {
        match BbMetalCtGsR4::new(&shader_dir()) {
            Ok(g) => Some(g),
            Err(NttError::DeviceNotFound) => {
                eprintln!("No Metal device — skipping");
                None
            }
            Err(e) => panic!("Failed to init: {}", e),
        }
    }

    fn bb_test_data(n: usize) -> Vec<BabyBear> {
        let mut seed: u64 = n as u64 * 11111 + 98765;
        (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                BabyBear::to_monty(((seed >> 33) as u32) % BabyBear::P)
            })
            .collect()
    }

    #[test]
    fn test_forward_matches_cpu() {
        let gpu = match init() {
            Some(g) => g,
            None => return,
        };
        let cpu = BbCpuReferenceBackend::new();
        for &n in &[4, 16, 256, 1024, 4096, 8192, 16384] {
            let original = bb_test_data(n);
            let mut cpu_data = original.clone();
            cpu.forward_ntt(&mut cpu_data, &[]).unwrap();
            let mut gpu_data = original.clone();
            gpu.forward_ntt(&mut gpu_data, &[]).unwrap();
            assert_eq!(gpu_data, cpu_data, "V4 forward mismatch at size {}", n);
        }
    }

    #[test]
    fn test_roundtrip() {
        let gpu = match init() {
            Some(g) => g,
            None => return,
        };
        for &n in &[4, 16, 256, 1024, 4096, 8192, 16384] {
            let original = bb_test_data(n);
            let mut data = original.clone();
            gpu.forward_ntt(&mut data, &[]).unwrap();
            assert_ne!(
                data, original,
                "V4 forward should change data at size {}",
                n
            );
            gpu.inverse_ntt(&mut data, &[]).unwrap();
            assert_eq!(data, original, "V4 round-trip failed at size {}", n);
        }
    }

    #[test]
    fn test_edge_cases() {
        let gpu = match init() {
            Some(g) => g,
            None => return,
        };
        let mut data = vec![BabyBear::zero(); 64];
        gpu.forward_ntt(&mut data, &[]).unwrap();
        assert!(data.iter().all(|&x| x == BabyBear::zero()));

        let mut data = vec![BabyBear::to_monty(42)];
        gpu.forward_ntt(&mut data, &[]).unwrap();
        assert_eq!(data[0], BabyBear::to_monty(42));
    }

    #[test]
    fn test_forward_odd_log_n() {
        let gpu = match init() {
            Some(g) => g,
            None => return,
        };
        let cpu = BbCpuReferenceBackend::new();
        // Odd log_n exercises radix-4 + final radix-2 fallback
        for &n in &[8, 32, 2048] {
            let original = bb_test_data(n);
            let mut cpu_data = original.clone();
            cpu.forward_ntt(&mut cpu_data, &[]).unwrap();
            let mut gpu_data = original.clone();
            gpu.forward_ntt(&mut gpu_data, &[]).unwrap();
            assert_eq!(gpu_data, cpu_data, "V4 odd log_n mismatch at size {}", n);
        }
    }

    #[test]
    fn test_roundtrip_odd_log_n() {
        let gpu = match init() {
            Some(g) => g,
            None => return,
        };
        for &n in &[8, 32, 2048, 32768] {
            let original = bb_test_data(n);
            let mut data = original.clone();
            gpu.forward_ntt(&mut data, &[]).unwrap();
            gpu.inverse_ntt(&mut data, &[]).unwrap();
            assert_eq!(data, original, "V4 odd roundtrip failed at size {}", n);
        }
    }
}
