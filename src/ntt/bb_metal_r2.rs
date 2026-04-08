//! BabyBear radix-2 NTT on Metal GPU — in-place with threadgroup memory.
//!
//! Same two-phase strategy as `MetalCtGsR2` (M31), but using BabyBear
//! Montgomery field arithmetic. Supports both single-column and batched
//! NTT for coset LDE.

use crate::field::babybear::BabyBear;
use crate::field::Field;
use crate::gpu::MetalContext;
use crate::ntt::bb_twiddles::BbTwiddleCache;
use crate::ntt::{NttBackend, NttError};
use metal::*;
use std::path::Path;

const MAX_TILE_LOG: usize = 13;

pub struct BbMetalR2 {
    ctx: MetalContext,
    forward_tg_pipeline: ComputePipelineState,
    inverse_tg_pipeline: ComputePipelineState,
    forward_device_pipeline: ComputePipelineState,
    inverse_device_pipeline: ComputePipelineState,
    normalize_pipeline: ComputePipelineState,
    // Batch pipelines
    batch_forward_tg_pipeline: ComputePipelineState,
    batch_inverse_tg_pipeline: ComputePipelineState,
    batch_forward_device_pipeline: ComputePipelineState,
    batch_inverse_device_pipeline: ComputePipelineState,
    batch_normalize_pipeline: ComputePipelineState,
    // LDE helper pipelines
    zero_pad_pipeline: ComputePipelineState,
    coset_shift_pipeline: ComputePipelineState,
    fused_norm_zeropad_shift_pipeline: ComputePipelineState,
    twiddle_cache: BbTwiddleCache,
}

#[allow(clippy::too_many_arguments)]
impl BbMetalR2 {
    pub fn new(shader_dir: &Path) -> Result<Self, NttError> {
        let ctx = MetalContext::new(shader_dir)?;

        // Single-column pipelines
        let forward_tg = ctx.make_pipeline("bb_r2_forward_tg")?;
        let inverse_tg = ctx.make_pipeline("bb_r2_inverse_tg")?;
        let forward_dev = ctx.make_pipeline("bb_r2_butterfly_device")?;
        let inverse_dev = ctx.make_pipeline("bb_r2_butterfly_device_inv")?;
        let normalize = ctx.make_pipeline("bb_r2_normalize")?;

        // Batch pipelines
        let batch_forward_tg = ctx.make_pipeline("bb_r2_batch_forward_tg")?;
        let batch_inverse_tg = ctx.make_pipeline("bb_r2_batch_inverse_tg")?;
        let batch_forward_dev = ctx.make_pipeline("bb_r2_batch_butterfly_device")?;
        let batch_inverse_dev = ctx.make_pipeline("bb_r2_batch_butterfly_device_inv")?;
        let batch_normalize = ctx.make_pipeline("bb_r2_batch_normalize")?;

        // LDE helpers
        let zero_pad = ctx.make_pipeline("bb_zero_pad_batch")?;
        let coset_shift = ctx.make_pipeline("bb_coset_shift_batch")?;
        let fused_norm_zeropad_shift = ctx.make_pipeline("bb_fused_normalize_zeropad_shift")?;

        Ok(Self {
            ctx,
            forward_tg_pipeline: forward_tg,
            inverse_tg_pipeline: inverse_tg,
            forward_device_pipeline: forward_dev,
            inverse_device_pipeline: inverse_dev,
            normalize_pipeline: normalize,
            batch_forward_tg_pipeline: batch_forward_tg,
            batch_inverse_tg_pipeline: batch_inverse_tg,
            batch_forward_device_pipeline: batch_forward_dev,
            batch_inverse_device_pipeline: batch_inverse_dev,
            batch_normalize_pipeline: batch_normalize,
            zero_pad_pipeline: zero_pad,
            coset_shift_pipeline: coset_shift,
            fused_norm_zeropad_shift_pipeline: fused_norm_zeropad_shift,
            twiddle_cache: crate::ntt::bb_twiddles::new_bb_twiddle_cache(),
        })
    }

    pub fn ctx(&self) -> &MetalContext {
        &self.ctx
    }

    // ── Single-column NTT ────────────────────────────────────────────────

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
        let cmd = self.ctx.begin_batch();
        let mut retain = Vec::new();

        let tile_log = log_n.min(MAX_TILE_LOG);

        // Device-memory stages (large strides): layers (log_n-1) down to tile_log
        for layer in (tile_log..log_n).rev() {
            self.encode_butterfly_bb(
                cmd,
                &mut retain,
                &self.forward_device_pipeline,
                &buf_data,
                &twiddles[layer],
                1 << layer,
                n,
            )?;
        }

        // Threadgroup stages (small strides): layers (tile_log-1) down to 0
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

        // Threadgroup stages: layers 0 up to (tile_log-1)
        if tile_log > 0 {
            self.encode_threadgroup_inverse(cmd, &mut retain, &buf_data, &itwiddles, n, tile_log)?;
        }

        // Device-memory stages: layers tile_log up to (log_n-1)
        for layer in tile_log..log_n {
            self.encode_butterfly_bb(
                cmd,
                &mut retain,
                &self.inverse_device_pipeline,
                &buf_data,
                &itwiddles[layer],
                1 << layer,
                n,
            )?;
        }

        // Normalize
        let inv_n = BabyBear::reduce(n as u64).inv();
        self.encode_normalize_bb(
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

    // ── Batched NTT (SoA layout: data[col * n + row]) ────────────────────

    /// Forward NTT on `batch_size` columns of length `n` each.
    /// Input layout: SoA, `data[col * n + row]`.
    /// Returns (result, gpu_time_ns).
    pub fn forward_ntt_batch_gpu(
        &self,
        input: &[u32],
        log_n: usize,
        batch_size: usize,
    ) -> Result<(Vec<u32>, u64), NttError> {
        let n = 1usize << log_n;
        if input.len() != n * batch_size {
            return Err(NttError::InvalidSize(input.len()));
        }
        if log_n == 0 {
            return Ok((input.to_vec(), 0));
        }

        let twiddles = self.twiddle_cache.forward(log_n as u32);
        let buf_data = self.ctx.buffer_from_slice(input)?;
        let cmd = self.ctx.begin_batch();
        let mut retain = Vec::new();

        let tile_log = log_n.min(MAX_TILE_LOG);

        // Device-memory stages
        for layer in (tile_log..log_n).rev() {
            self.encode_batch_butterfly(
                cmd,
                &mut retain,
                &self.batch_forward_device_pipeline,
                &buf_data,
                &twiddles[layer],
                1 << layer,
                n,
                batch_size,
            )?;
        }

        // Threadgroup stages
        if tile_log > 0 {
            self.encode_batch_threadgroup_forward(
                cmd,
                &mut retain,
                &buf_data,
                &twiddles,
                n,
                tile_log,
                batch_size,
            )?;
        }

        let total_ns = MetalContext::submit_batch(cmd, &retain)?;
        let result = MetalContext::read_buffer(&buf_data, n * batch_size);
        Ok((result, total_ns))
    }

    /// Inverse NTT on `batch_size` columns of length `n` each.
    pub fn inverse_ntt_batch_gpu(
        &self,
        input: &[u32],
        log_n: usize,
        batch_size: usize,
    ) -> Result<(Vec<u32>, u64), NttError> {
        let n = 1usize << log_n;
        if input.len() != n * batch_size {
            return Err(NttError::InvalidSize(input.len()));
        }
        if log_n == 0 {
            return Ok((input.to_vec(), 0));
        }

        let itwiddles = self.twiddle_cache.inverse(log_n as u32);
        let buf_data = self.ctx.buffer_from_slice(input)?;
        let cmd = self.ctx.begin_batch();
        let mut retain = Vec::new();

        let tile_log = log_n.min(MAX_TILE_LOG);

        if tile_log > 0 {
            self.encode_batch_threadgroup_inverse(
                cmd,
                &mut retain,
                &buf_data,
                &itwiddles,
                n,
                tile_log,
                batch_size,
            )?;
        }

        for layer in tile_log..log_n {
            self.encode_batch_butterfly(
                cmd,
                &mut retain,
                &self.batch_inverse_device_pipeline,
                &buf_data,
                &itwiddles[layer],
                1 << layer,
                n,
                batch_size,
            )?;
        }

        // Normalize
        let inv_n = BabyBear::reduce(n as u64).inv();
        self.encode_batch_normalize(cmd, &mut retain, &buf_data, n, batch_size, inv_n)?;

        let total_ns = MetalContext::submit_batch(cmd, &retain)?;
        let result = MetalContext::read_buffer(&buf_data, n * batch_size);
        Ok((result, total_ns))
    }

    // ── Batched NTT on pre-allocated buffer (for LDE pipeline) ───────────

    /// Encode batched forward NTT into an existing command buffer on a pre-allocated buffer.
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

        for layer in (tile_log..log_n).rev() {
            self.encode_batch_butterfly(
                cmd,
                retain,
                &self.batch_forward_device_pipeline,
                buf_data,
                &twiddles[layer],
                1 << layer,
                n,
                batch_size,
            )?;
        }
        if tile_log > 0 {
            self.encode_batch_threadgroup_forward(
                cmd, retain, buf_data, &twiddles, n, tile_log, batch_size,
            )?;
        }
        Ok(())
    }

    /// Encode batched inverse NTT into an existing command buffer on a pre-allocated buffer.
    pub fn encode_inverse_ntt_batch(
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

        if tile_log > 0 {
            self.encode_batch_threadgroup_inverse(
                cmd, retain, buf_data, &itwiddles, n, tile_log, batch_size,
            )?;
        }
        for layer in tile_log..log_n {
            self.encode_batch_butterfly(
                cmd,
                retain,
                &self.batch_inverse_device_pipeline,
                buf_data,
                &itwiddles[layer],
                1 << layer,
                n,
                batch_size,
            )?;
        }

        let inv_n = BabyBear::reduce(n as u64).inv();
        self.encode_batch_normalize(cmd, retain, buf_data, n, batch_size, inv_n)?;
        Ok(())
    }

    // ── LDE helper dispatches ────────────────────────────────────────────

    /// Encode zero-pad dispatch: copies input buffer to output buffer with padding.
    pub fn encode_zero_pad(
        &self,
        cmd: &CommandBufferRef,
        retain: &mut Vec<Buffer>,
        buf_input: &Buffer,
        buf_output: &Buffer,
        n_orig: usize,
        n_ext: usize,
        batch_size: usize,
    ) -> Result<(), NttError> {
        let params: Vec<u32> = vec![n_orig as u32, n_ext as u32, batch_size as u32];
        let buf_p = self.ctx.buffer_from_slice(&params)?;

        let max_tpg = MetalContext::max_threads_per_threadgroup(&self.zero_pad_pipeline) as u64;
        let tpg_x = max_tpg.min(n_ext as u64).max(1);
        let tg_x = (n_ext as u64).div_ceil(tpg_x);

        MetalContext::encode_dispatch(
            cmd,
            &self.zero_pad_pipeline,
            &[buf_input, buf_output, &buf_p],
            MTLSize::new(tg_x, batch_size as u64, 1),
            MTLSize::new(tpg_x, 1, 1),
        );
        retain.push(buf_p);
        Ok(())
    }

    /// Encode coset-shift dispatch: multiplies each row by precomputed shift powers.
    pub fn encode_coset_shift(
        &self,
        cmd: &CommandBufferRef,
        retain: &mut Vec<Buffer>,
        buf_data: &Buffer,
        shift_powers: &[u32],
        n_ext: usize,
        batch_size: usize,
    ) -> Result<(), NttError> {
        let buf_shifts = self.ctx.buffer_from_slice(shift_powers)?;
        let params: Vec<u32> = vec![n_ext as u32, batch_size as u32];
        let buf_p = self.ctx.buffer_from_slice(&params)?;

        let max_tpg = MetalContext::max_threads_per_threadgroup(&self.coset_shift_pipeline) as u64;
        let tpg_x = max_tpg.min(n_ext as u64).max(1);
        let tg_x = (n_ext as u64).div_ceil(tpg_x);

        MetalContext::encode_dispatch(
            cmd,
            &self.coset_shift_pipeline,
            &[buf_data, &buf_shifts, &buf_p],
            MTLSize::new(tg_x, batch_size as u64, 1),
            MTLSize::new(tpg_x, 1, 1),
        );
        retain.push(buf_shifts);
        retain.push(buf_p);
        Ok(())
    }

    /// Encode batched inverse NTT without final normalize (for fused LDE pipeline).
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

        if tile_log > 0 {
            self.encode_batch_threadgroup_inverse(
                cmd, retain, buf_data, &itwiddles, n, tile_log, batch_size,
            )?;
        }
        for layer in tile_log..log_n {
            self.encode_batch_butterfly(
                cmd,
                retain,
                &self.batch_inverse_device_pipeline,
                buf_data,
                &itwiddles[layer],
                1 << layer,
                n,
                batch_size,
            )?;
        }
        // No normalize — caller will fuse it into the next step
        Ok(())
    }

    /// Encode fused normalize + zero-pad + coset-shift dispatch.
    /// Reads from buf_input (n_orig per col), writes to buf_output (n_ext per col).
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

    // ── Private encoding helpers ─────────────────────────────────────────

    /// Encode a single-column BabyBear radix-2 butterfly dispatch.
    fn encode_butterfly_bb(
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

    /// Encode single-column normalize dispatch.
    fn encode_normalize_bb(
        &self,
        cmd: &CommandBufferRef,
        retain: &mut Vec<Buffer>,
        pipeline: &ComputePipelineState,
        buf_data: &Buffer,
        n: usize,
        inv_n: BabyBear,
    ) -> Result<(), NttError> {
        let params: Vec<u32> = vec![n as u32, inv_n.0];
        let buf_p = self.ctx.buffer_from_slice(&params)?;

        let max_tpg = MetalContext::max_threads_per_threadgroup(pipeline) as u64;
        let (tg, tpg) = MetalContext::compute_grid_1d(n as u64, max_tpg.min(256));

        MetalContext::encode_dispatch(cmd, pipeline, &[buf_data, &buf_p], tg, tpg);
        retain.push(buf_p);
        Ok(())
    }

    /// Encode single-column forward threadgroup kernel.
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
        let start_layer = tile_log - 1;

        let mut flat_tw = Vec::new();
        let mut tw_offsets = Vec::new();
        for li in 0..num_tg_layers {
            let layer = start_layer - li;
            tw_offsets.push(flat_tw.len() as u32);
            flat_tw.extend(twiddles[layer].iter().map(|m| m.0));
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
            MetalContext::max_threads_per_threadgroup(&self.forward_tg_pipeline) as u64;
        let threads = max_tg_threads.min(tile_size as u64 / 2).max(1);

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

    /// Encode single-column inverse threadgroup kernel.
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
        let start_layer = 0;

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

    // ── Batch encoding helpers ───────────────────────────────────────────

    /// Encode a batched butterfly dispatch (2D grid: x=butterflies, y=batch).
    fn encode_batch_butterfly(
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

    /// Encode batched forward threadgroup kernel (1D grid: tiles_per_col * batch_size).
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
        let start_layer = tile_log - 1;

        let mut flat_tw = Vec::new();
        let mut tw_offsets = Vec::new();
        for li in 0..num_tg_layers {
            let layer = start_layer - li;
            tw_offsets.push(flat_tw.len() as u32);
            flat_tw.extend(twiddles[layer].iter().map(|m| m.0));
        }
        let buf_tw = self.ctx.buffer_from_slice(&flat_tw)?;

        let tile_size = 1usize << tile_log;
        let num_tiles_per_col = n / tile_size;

        // params: [n, tile_log, num_layers, start_layer, num_tiles_per_col, tw_offsets...]
        let mut params: Vec<u32> = vec![
            n as u32,
            tile_log as u32,
            num_tg_layers as u32,
            start_layer as u32,
            num_tiles_per_col as u32,
        ];
        params.extend(tw_offsets);
        let buf_p = self.ctx.buffer_from_slice(&params)?;

        let total_threadgroups = (num_tiles_per_col * batch_size) as u64;
        let max_tg_threads =
            MetalContext::max_threads_per_threadgroup(&self.batch_forward_tg_pipeline) as u64;
        let threads = max_tg_threads.min(tile_size as u64 / 2).max(1);

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

    /// Encode batched inverse threadgroup kernel (1D grid: tiles_per_col * batch_size).
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
        let start_layer = 0;

        let mut flat_tw = Vec::new();
        let mut tw_offsets = Vec::new();
        for li in 0..num_tg_layers {
            let layer = start_layer + li;
            tw_offsets.push(flat_tw.len() as u32);
            flat_tw.extend(itwiddles[layer].iter().map(|m| m.0));
        }
        let buf_tw = self.ctx.buffer_from_slice(&flat_tw)?;

        let tile_size = 1usize << tile_log;
        let num_tiles_per_col = n / tile_size;

        // params: [n, tile_log, num_layers, start_layer, num_tiles_per_col, tw_offsets...]
        let mut params: Vec<u32> = vec![
            n as u32,
            tile_log as u32,
            num_tg_layers as u32,
            start_layer as u32,
            num_tiles_per_col as u32,
        ];
        params.extend(tw_offsets);
        let buf_p = self.ctx.buffer_from_slice(&params)?;

        let total_threadgroups = (num_tiles_per_col * batch_size) as u64;
        let max_tg_threads =
            MetalContext::max_threads_per_threadgroup(&self.batch_inverse_tg_pipeline) as u64;
        let threads = max_tg_threads.min(tile_size as u64 / 2).max(1);

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

    /// Encode batched normalize dispatch.
    fn encode_batch_normalize(
        &self,
        cmd: &CommandBufferRef,
        retain: &mut Vec<Buffer>,
        buf_data: &Buffer,
        n: usize,
        batch_size: usize,
        inv_n: BabyBear,
    ) -> Result<(), NttError> {
        let params: Vec<u32> = vec![n as u32, inv_n.0, batch_size as u32];
        let buf_p = self.ctx.buffer_from_slice(&params)?;

        let max_tpg =
            MetalContext::max_threads_per_threadgroup(&self.batch_normalize_pipeline) as u64;
        let tpg_x = max_tpg.min(n as u64).max(1);
        let tg_x = (n as u64).div_ceil(tpg_x);

        MetalContext::encode_dispatch(
            cmd,
            &self.batch_normalize_pipeline,
            &[buf_data, &buf_p],
            MTLSize::new(tg_x, batch_size as u64, 1),
            MTLSize::new(tpg_x, 1, 1),
        );
        retain.push(buf_p);
        Ok(())
    }
}

impl NttBackend<BabyBear> for BbMetalR2 {
    fn name(&self) -> &str {
        "bb-metal-r2"
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

    fn init() -> Option<BbMetalR2> {
        match BbMetalR2::new(&shader_dir()) {
            Ok(g) => Some(g),
            Err(NttError::DeviceNotFound) => {
                eprintln!("No Metal device — skipping");
                None
            }
            Err(e) => panic!("Failed to init BbMetalR2: {}", e),
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

        for &n in &[4, 16, 256, 1024, 8192] {
            let original = bb_test_data(n);
            let mut cpu_data = original.clone();
            cpu.forward_ntt(&mut cpu_data, &[]).unwrap();
            let mut gpu_data = original.clone();
            gpu.forward_ntt(&mut gpu_data, &[]).unwrap();
            assert_eq!(gpu_data, cpu_data, "Forward mismatch at size {}", n,);
        }
    }

    #[test]
    fn test_roundtrip() {
        let gpu = match init() {
            Some(g) => g,
            None => return,
        };

        for &n in &[4, 16, 256, 1024, 8192] {
            let original = bb_test_data(n);
            let mut data = original.clone();
            gpu.forward_ntt(&mut data, &[]).unwrap();
            assert_ne!(data, original, "Forward should change data at size {}", n);
            gpu.inverse_ntt(&mut data, &[]).unwrap();
            assert_eq!(data, original, "Round-trip failed at size {}", n);
        }
    }

    #[test]
    fn test_batch_matches_single() {
        let gpu = match init() {
            Some(g) => g,
            None => return,
        };

        let log_n = 10; // n=1024
        let n = 1 << log_n;
        let batch_size = 4;

        // Generate batch_size columns of test data
        let columns: Vec<Vec<u32>> = (0..batch_size)
            .map(|_col| bb_test_data(n).iter().map(|b| b.0).collect::<Vec<u32>>())
            .collect();

        // Run single NTT on each column
        let single_results: Vec<Vec<u32>> = columns
            .iter()
            .map(|col| gpu.forward_ntt_gpu(col, log_n).unwrap().0)
            .collect();

        // Run batched NTT
        let mut batch_input = Vec::with_capacity(n * batch_size);
        for col in &columns {
            batch_input.extend_from_slice(col);
        }
        let (batch_result, _) = gpu
            .forward_ntt_batch_gpu(&batch_input, log_n, batch_size)
            .unwrap();

        // Compare each column
        for col in 0..batch_size {
            let batch_col = &batch_result[col * n..(col + 1) * n];
            assert_eq!(
                batch_col,
                &single_results[col][..],
                "Batch column {} doesn't match single NTT",
                col
            );
        }
    }

    #[test]
    fn test_batch_roundtrip() {
        let gpu = match init() {
            Some(g) => g,
            None => return,
        };

        let log_n = 10;
        let n = 1 << log_n;
        let batch_size = 4;

        let original: Vec<u32> = (0..n * batch_size)
            .map(|i| BabyBear::to_monty((i as u32 * 7 + 3) % BabyBear::P).0)
            .collect();

        let (forward, _) = gpu
            .forward_ntt_batch_gpu(&original, log_n, batch_size)
            .unwrap();
        let (result, _) = gpu
            .inverse_ntt_batch_gpu(&forward, log_n, batch_size)
            .unwrap();

        assert_eq!(result, original, "Batch round-trip failed");
    }

    #[test]
    fn test_edge_cases() {
        let gpu = match init() {
            Some(g) => g,
            None => return,
        };

        // Size 1
        let mut data = vec![BabyBear::to_monty(42)];
        gpu.forward_ntt(&mut data, &[]).unwrap();
        assert_eq!(data[0], BabyBear::to_monty(42));

        // All zeros
        let mut data = vec![BabyBear::zero(); 64];
        gpu.forward_ntt(&mut data, &[]).unwrap();
        assert!(data.iter().all(|&x| x == BabyBear::zero()));

        // Invalid size
        let mut data = vec![BabyBear::one(); 3];
        assert!(gpu.forward_ntt(&mut data, &[]).is_err());
    }
}
