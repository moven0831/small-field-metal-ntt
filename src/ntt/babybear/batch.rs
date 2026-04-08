//! BabyBear batched radix-2 NTT on Metal GPU.
//!
//! Composes with the generic `MetalR2<BbR2Config>` for single-column NTT
//! and adds batch-specific dispatch (2D grid, SoA layout).

use crate::field::babybear::BabyBear;
use crate::field::Field;
use crate::gpu::MetalContext;
use crate::ntt::metal_r2::MetalR2;
use crate::ntt::shader_config::BbR2Config;
use crate::ntt::{NttBackend, NttError};
use metal::*;
use std::path::Path;

const MAX_TILE_LOG: usize = 13;

/// BabyBear batch NTT backend: single-column via generic `MetalR2` + batch dispatch.
#[allow(clippy::too_many_arguments)]
pub struct BbBatchNtt {
    single: MetalR2<BbR2Config>,
    // Batch pipelines
    batch_forward_tg_pipeline: ComputePipelineState,
    batch_inverse_tg_pipeline: ComputePipelineState,
    batch_forward_device_pipeline: ComputePipelineState,
    batch_inverse_device_pipeline: ComputePipelineState,
    batch_normalize_pipeline: ComputePipelineState,
}

impl BbBatchNtt {
    pub fn new(shader_dir: &Path) -> Result<Self, NttError> {
        let single = MetalR2::<BbR2Config>::new(shader_dir)?;

        let batch_forward_tg = single.ctx().make_pipeline("bb_r2_batch_forward_tg")?;
        let batch_inverse_tg = single.ctx().make_pipeline("bb_r2_batch_inverse_tg")?;
        let batch_forward_dev = single.ctx().make_pipeline("bb_r2_batch_butterfly_device")?;
        let batch_inverse_dev = single.ctx().make_pipeline("bb_r2_batch_butterfly_device_inv")?;
        let batch_normalize = single.ctx().make_pipeline("bb_r2_batch_normalize")?;

        Ok(Self {
            single,
            batch_forward_tg_pipeline: batch_forward_tg,
            batch_inverse_tg_pipeline: batch_inverse_tg,
            batch_forward_device_pipeline: batch_forward_dev,
            batch_inverse_device_pipeline: batch_inverse_dev,
            batch_normalize_pipeline: batch_normalize,
        })
    }

    pub fn ctx(&self) -> &MetalContext {
        self.single.ctx()
    }

    // ── Single-column delegation ────────────────────────────────────────

    pub fn forward_ntt_gpu(
        &self,
        input: &[u32],
        log_n: usize,
    ) -> Result<(Vec<u32>, u64), NttError> {
        self.single.forward_ntt_gpu(input, log_n)
    }

    pub fn inverse_ntt_gpu(
        &self,
        input: &[u32],
        log_n: usize,
    ) -> Result<(Vec<u32>, u64), NttError> {
        self.single.inverse_ntt_gpu(input, log_n)
    }

    // ── Batched NTT (SoA layout: data[col * n + row]) ────────────────────

    /// Forward NTT on `batch_size` columns of length `n` each.
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

        let twiddles = self.single.twiddle_cache().forward(log_n as u32);
        let buf_data = self.ctx().buffer_from_slice(input)?;
        let cmd = self.ctx().begin_batch();
        let mut retain = Vec::new();

        let tile_log = log_n.min(MAX_TILE_LOG);

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

        let itwiddles = self.single.twiddle_cache().inverse(log_n as u32);
        let buf_data = self.ctx().buffer_from_slice(input)?;
        let cmd = self.ctx().begin_batch();
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

        let inv_n = BabyBear::reduce(n as u64).inv();
        self.encode_batch_normalize(cmd, &mut retain, &buf_data, n, batch_size, inv_n)?;

        let total_ns = MetalContext::submit_batch(cmd, &retain)?;
        let result = MetalContext::read_buffer(&buf_data, n * batch_size);
        Ok((result, total_ns))
    }

    // ── Private batch encoding helpers ──────────────────────────────────

    #[allow(clippy::too_many_arguments)]
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
        let tw_data: Vec<u32> = twiddles.iter().map(|m| m.raw()).collect();
        let buf_tw = self.ctx().buffer_from_slice(&tw_data)?;
        let params: Vec<u32> = vec![stride as u32, n as u32, batch_size as u32];
        let buf_p = self.ctx().buffer_from_slice(&params)?;

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

    #[allow(clippy::too_many_arguments)]
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
            flat_tw.extend(twiddles[layer].iter().map(|m| m.raw()));
        }
        let buf_tw = self.ctx().buffer_from_slice(&flat_tw)?;

        let tile_size = 1usize << tile_log;
        let num_tiles_per_col = n / tile_size;

        let mut params: Vec<u32> = vec![
            n as u32,
            tile_log as u32,
            num_tg_layers as u32,
            start_layer as u32,
            num_tiles_per_col as u32,
        ];
        params.extend(tw_offsets);
        let buf_p = self.ctx().buffer_from_slice(&params)?;

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

    #[allow(clippy::too_many_arguments)]
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
            flat_tw.extend(itwiddles[layer].iter().map(|m| m.raw()));
        }
        let buf_tw = self.ctx().buffer_from_slice(&flat_tw)?;

        let tile_size = 1usize << tile_log;
        let num_tiles_per_col = n / tile_size;

        let mut params: Vec<u32> = vec![
            n as u32,
            tile_log as u32,
            num_tg_layers as u32,
            start_layer as u32,
            num_tiles_per_col as u32,
        ];
        params.extend(tw_offsets);
        let buf_p = self.ctx().buffer_from_slice(&params)?;

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

    fn encode_batch_normalize(
        &self,
        cmd: &CommandBufferRef,
        retain: &mut Vec<Buffer>,
        buf_data: &Buffer,
        n: usize,
        batch_size: usize,
        inv_n: BabyBear,
    ) -> Result<(), NttError> {
        let params: Vec<u32> = vec![n as u32, inv_n.raw(), batch_size as u32];
        let buf_p = self.ctx().buffer_from_slice(&params)?;

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

impl NttBackend<BabyBear> for BbBatchNtt {
    fn name(&self) -> &str {
        "bb-metal-r2"
    }

    fn forward_ntt(&self, data: &mut [BabyBear], twiddles: &[BabyBear]) -> Result<(), NttError> {
        self.single.forward_ntt(data, twiddles)
    }

    fn inverse_ntt(&self, data: &mut [BabyBear], twiddles: &[BabyBear]) -> Result<(), NttError> {
        self.single.inverse_ntt(data, twiddles)
    }

    fn pointwise_mul(
        &self,
        a: &[BabyBear],
        b: &[BabyBear],
        out: &mut [BabyBear],
    ) -> Result<(), NttError> {
        self.single.pointwise_mul(a, b, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ntt::babybear::cpu_reference::BbCpuReferenceBackend;
    use crate::ntt::test_utils::shader_dir;

    fn init() -> Option<BbBatchNtt> {
        match BbBatchNtt::new(&shader_dir()) {
            Ok(g) => Some(g),
            Err(NttError::DeviceNotFound) => {
                eprintln!("No Metal device — skipping");
                None
            }
            Err(e) => panic!("Failed to init BbBatchNtt: {}", e),
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
            assert_eq!(gpu_data, cpu_data, "Forward mismatch at size {}", n);
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

        let log_n = 10;
        let n = 1 << log_n;
        let batch_size = 4;

        let columns: Vec<Vec<u32>> = (0..batch_size)
            .map(|_col| bb_test_data(n).iter().map(|b| b.raw()).collect::<Vec<u32>>())
            .collect();

        let single_results: Vec<Vec<u32>> = columns
            .iter()
            .map(|col| gpu.forward_ntt_gpu(col, log_n).unwrap().0)
            .collect();

        let mut batch_input = Vec::with_capacity(n * batch_size);
        for col in &columns {
            batch_input.extend_from_slice(col);
        }
        let (batch_result, _) = gpu
            .forward_ntt_batch_gpu(&batch_input, log_n, batch_size)
            .unwrap();

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
            .map(|i| BabyBear::to_monty((i as u32 * 7 + 3) % BabyBear::P).raw())
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
