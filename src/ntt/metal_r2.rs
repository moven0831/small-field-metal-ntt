//! Generic radix-2 in-place NTT on Metal GPU (CT-DIT forward / GS-DIF inverse).
//!
//! Parameterized by `R2ShaderConfig` to support multiple fields (M31, BabyBear)
//! with identical dispatch logic but field-specific shader kernels.
//!
//! Two-phase execution:
//! - Threadgroup phase: Layers with stride <= tile_size/2 in on-chip memory.
//! - Device-memory phase: Large-stride layers, one dispatch per layer.

use crate::field::Field;
use crate::gpu::MetalContext;
use crate::ntt::shader_config::R2ShaderConfig;
use crate::ntt::twiddle_cache::TwiddleCache;
use crate::ntt::{NttBackend, NttError};
use metal::*;
use std::marker::PhantomData;
use std::path::Path;

/// Maximum elements per threadgroup tile: 8192 (= 32 KB of threadgroup memory).
const MAX_TILE_LOG: usize = 13;

/// Generic radix-2 Metal NTT backend.
///
/// Type aliases for specific fields:
/// - `MetalCtGsR2` = `MetalR2<M31R2Config>` (M31 Circle NTT)
/// - `BbMetalR2` = `MetalR2<BbR2Config>` (BabyBear standard NTT)
pub struct MetalR2<C: R2ShaderConfig> {
    ctx: MetalContext,
    forward_tg_pipeline: ComputePipelineState,
    inverse_tg_pipeline: ComputePipelineState,
    forward_device_pipeline: ComputePipelineState,
    inverse_device_pipeline: ComputePipelineState,
    normalize_pipeline: ComputePipelineState,
    twiddle_cache: TwiddleCache<C::F>,
    _phantom: PhantomData<C>,
}

impl<C: R2ShaderConfig> MetalR2<C> {
    pub fn new(shader_dir: &Path) -> Result<Self, NttError> {
        let ctx = MetalContext::new(shader_dir)?;
        let forward_tg = ctx.make_pipeline(C::FORWARD_TG)?;
        let inverse_tg = ctx.make_pipeline(C::INVERSE_TG)?;
        let forward_dev = ctx.make_pipeline(C::FORWARD_DEVICE)?;
        let inverse_dev = ctx.make_pipeline(C::INVERSE_DEVICE)?;
        let normalize = ctx.make_pipeline(C::NORMALIZE)?;

        Ok(Self {
            ctx,
            forward_tg_pipeline: forward_tg,
            inverse_tg_pipeline: inverse_tg,
            forward_device_pipeline: forward_dev,
            inverse_device_pipeline: inverse_dev,
            normalize_pipeline: normalize,
            twiddle_cache: C::make_twiddle_cache(),
            _phantom: PhantomData,
        })
    }

    pub fn ctx(&self) -> &MetalContext {
        &self.ctx
    }

    /// Run forward NTT on GPU. Returns (result_data, total_gpu_time_ns).
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

        // Device-memory stages (large strides): layers (log_n-1) down to tile_log
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

        // Threadgroup stages (small strides): layers (tile_log-1) down to 0
        if tile_log > 0 {
            self.encode_threadgroup_forward(cmd, &mut retain, &buf_data, &twiddles, n, tile_log)?;
        }

        let total_ns = MetalContext::submit_batch(cmd, &retain)?;
        let result = MetalContext::read_buffer(&buf_data, n);
        Ok((result, total_ns))
    }

    /// Run inverse NTT on GPU. Returns (result_data, total_gpu_time_ns).
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

        // Threadgroup stages: layers 0 up to (tile_log-1)
        if tile_log > 0 {
            self.encode_threadgroup_inverse(cmd, &mut retain, &buf_data, &itwiddles, n, tile_log)?;
        }

        // Device-memory stages: layers tile_log up to (log_n-1)
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

        // Normalize: multiply all elements by inv_n
        let inv_n = C::F::reduce(n as u64).inv();
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
        twiddles: &[Vec<C::F>],
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
            flat_tw.extend(twiddles[layer].iter().map(|f| f.raw()));
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
        itwiddles: &[Vec<C::F>],
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
            flat_tw.extend(itwiddles[layer].iter().map(|f| f.raw()));
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

// ── NttBackend trait implementation ─────────────────────────────────────

impl<C: R2ShaderConfig> NttBackend<C::F> for MetalR2<C> {
    fn name(&self) -> &str {
        C::NAME
    }

    fn forward_ntt(&self, data: &mut [C::F], _twiddles: &[C::F]) -> Result<(), NttError> {
        let n = data.len();
        if n == 0 || (n & (n - 1)) != 0 {
            return Err(NttError::InvalidSize(n));
        }
        if n == 1 {
            return Ok(());
        }
        let log_n = n.trailing_zeros() as usize;

        let input: Vec<u32> = data.iter().map(|f| f.raw()).collect();
        let (result, _) = self.forward_ntt_gpu(&input, log_n)?;
        for (i, val) in result.iter().enumerate() {
            data[i] = C::F::from_raw(*val);
        }
        Ok(())
    }

    fn inverse_ntt(&self, data: &mut [C::F], _twiddles: &[C::F]) -> Result<(), NttError> {
        let n = data.len();
        if n == 0 || (n & (n - 1)) != 0 {
            return Err(NttError::InvalidSize(n));
        }
        if n == 1 {
            return Ok(());
        }
        let log_n = n.trailing_zeros() as usize;

        let input: Vec<u32> = data.iter().map(|f| f.raw()).collect();
        let (result, _) = self.inverse_ntt_gpu(&input, log_n)?;
        for (i, val) in result.iter().enumerate() {
            data[i] = C::F::from_raw(*val);
        }
        Ok(())
    }

    fn pointwise_mul(&self, a: &[C::F], b: &[C::F], out: &mut [C::F]) -> Result<(), NttError> {
        if a.len() != b.len() || a.len() != out.len() {
            return Err(NttError::InvalidSize(a.len()));
        }
        for i in 0..a.len() {
            out[i] = a[i].mul(b[i]);
        }
        Ok(())
    }
}

// ── Type aliases for backward compatibility ─────────────────────────────

/// M31 Circle NTT, radix-2 in-place (CT-DIT forward / GS-DIF inverse).
pub type MetalCtGsR2 = MetalR2<crate::ntt::shader_config::M31R2Config>;

/// BabyBear standard NTT, radix-2 in-place.
pub type BbMetalR2New = MetalR2<crate::ntt::shader_config::BbR2Config>;
