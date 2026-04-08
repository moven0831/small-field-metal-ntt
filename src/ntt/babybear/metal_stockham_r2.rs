//! BabyBear V3: Stockham radix-2 out-of-place NTT on Metal GPU.
//!
//! Reads from buffer A, writes to buffer B, then swaps (ping-pong).
//! Tests whether out-of-place memory pattern helps on Apple UMA.
//!
//! Threadgroup tile: 4096 elements (2 arrays x 16KB = 32KB total).
//! Handles 12 stages on-chip vs V2's 13.

use crate::field::babybear::BabyBear;
use crate::field::Field;
use crate::gpu::MetalContext;
use crate::ntt::babybear::twiddles::BbTwiddleCache;
use crate::ntt::{NttBackend, NttError};
use metal::*;
use std::path::Path;

const MAX_TILE_LOG: usize = 12;

pub struct BbMetalStockhamR2 {
    ctx: MetalContext,
    forward_tg_pipeline: ComputePipelineState,
    forward_device_pipeline: ComputePipelineState,
    inverse_tg_pipeline: ComputePipelineState,
    inverse_device_pipeline: ComputePipelineState,
    normalize_pipeline: ComputePipelineState,
    twiddle_cache: BbTwiddleCache,
}

impl BbMetalStockhamR2 {
    pub fn new(shader_dir: &Path) -> Result<Self, NttError> {
        let ctx = MetalContext::new(shader_dir)?;
        let forward_tg = ctx.make_pipeline("bb_stockham_r2_forward_tg")?;
        let forward_dev = ctx.make_pipeline("bb_stockham_r2_butterfly_device")?;
        let inverse_tg = ctx.make_pipeline("bb_stockham_r2_inverse_tg")?;
        let inverse_dev = ctx.make_pipeline("bb_stockham_r2_butterfly_device_inv")?;
        let normalize = ctx.make_pipeline("bb_stockham_r2_normalize")?;

        Ok(Self {
            ctx,
            forward_tg_pipeline: forward_tg,
            forward_device_pipeline: forward_dev,
            inverse_tg_pipeline: inverse_tg,
            inverse_device_pipeline: inverse_dev,
            normalize_pipeline: normalize,
            twiddle_cache: crate::ntt::babybear::twiddles::new_bb_twiddle_cache(),
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
        let buf_a = self.ctx.buffer_from_slice(input)?;
        let buf_b = self.ctx.alloc_buffer(n * 4)?;
        let cmd = self.ctx.begin_batch();
        let mut retain = Vec::new();

        let tile_log = log_n.min(MAX_TILE_LOG);
        let mut read_from_a = true;

        // Device-memory phase: layers (log_n-1) down to tile_log
        for layer in (tile_log..log_n).rev() {
            let stride = 1usize << layer;
            let tw_data: Vec<u32> = twiddles[layer].iter().map(|m| m.0).collect();
            let buf_tw = self.ctx.buffer_from_slice(&tw_data)?;
            let params: Vec<u32> = vec![stride as u32, n as u32];
            let buf_p = self.ctx.buffer_from_slice(&params)?;

            let num_butterflies = (n / 2) as u64;
            let max_tpg =
                MetalContext::max_threads_per_threadgroup(&self.forward_device_pipeline) as u64;
            let (tg, tpg) = MetalContext::compute_grid_1d(num_butterflies, max_tpg.min(256));

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

        // Threadgroup phase: layers (tile_log-1) down to 0
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
            let buf_tw = self.ctx.buffer_from_slice(&flat_tw)?;

            let mut params: Vec<u32> =
                vec![tile_log as u32, num_tg_layers as u32, start_layer as u32];
            params.extend(tw_offsets);
            let buf_p = self.ctx.buffer_from_slice(&params)?;

            let tile_size = 1usize << tile_log;
            let num_tiles = n / tile_size;
            let max_tg_threads =
                MetalContext::max_threads_per_threadgroup(&self.forward_tg_pipeline) as u64;
            let threads = max_tg_threads.min(tile_size as u64 / 2).max(1);

            let (cur_in, cur_out) = if read_from_a {
                (&buf_a, &buf_b)
            } else {
                (&buf_b, &buf_a)
            };
            MetalContext::encode_dispatch(
                cmd,
                &self.forward_tg_pipeline,
                &[cur_in, cur_out, &buf_tw, &buf_p],
                MTLSize::new(num_tiles as u64, 1, 1),
                MTLSize::new(threads, 1, 1),
            );
            retain.push(buf_tw);
            retain.push(buf_p);
            read_from_a = !read_from_a;
        }

        let total_ns = MetalContext::submit_batch(cmd, &retain)?;
        let result_buf = if read_from_a { &buf_a } else { &buf_b };
        let result = MetalContext::read_buffer(result_buf, n);
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
        let buf_a = self.ctx.buffer_from_slice(input)?;
        let buf_b = self.ctx.alloc_buffer(n * 4)?;
        let cmd = self.ctx.begin_batch();
        let mut retain = Vec::new();

        let tile_log = log_n.min(MAX_TILE_LOG);
        let mut read_from_a = true;

        // Threadgroup phase: layers 0 up to (tile_log-1)
        if tile_log > 0 {
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

            let mut params: Vec<u32> =
                vec![tile_log as u32, num_tg_layers as u32, start_layer as u32];
            params.extend(tw_offsets);
            let buf_p = self.ctx.buffer_from_slice(&params)?;

            let tile_size = 1usize << tile_log;
            let num_tiles = n / tile_size;
            let max_tg_threads =
                MetalContext::max_threads_per_threadgroup(&self.inverse_tg_pipeline) as u64;
            let threads = max_tg_threads.min(tile_size as u64 / 2).max(1);

            let (cur_in, cur_out) = if read_from_a {
                (&buf_a, &buf_b)
            } else {
                (&buf_b, &buf_a)
            };
            MetalContext::encode_dispatch(
                cmd,
                &self.inverse_tg_pipeline,
                &[cur_in, cur_out, &buf_tw, &buf_p],
                MTLSize::new(num_tiles as u64, 1, 1),
                MTLSize::new(threads, 1, 1),
            );
            retain.push(buf_tw);
            retain.push(buf_p);
            read_from_a = !read_from_a;
        }

        // Device-memory phase: layers tile_log up to (log_n-1)
        for layer in tile_log..log_n {
            let stride = 1usize << layer;
            let tw_data: Vec<u32> = itwiddles[layer].iter().map(|m| m.0).collect();
            let buf_tw = self.ctx.buffer_from_slice(&tw_data)?;
            let params: Vec<u32> = vec![stride as u32, n as u32];
            let buf_p = self.ctx.buffer_from_slice(&params)?;

            let num_butterflies = (n / 2) as u64;
            let max_tpg =
                MetalContext::max_threads_per_threadgroup(&self.inverse_device_pipeline) as u64;
            let (tg, tpg) = MetalContext::compute_grid_1d(num_butterflies, max_tpg.min(256));

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

        // Normalize in-place on result buffer
        let result_buf = if read_from_a { &buf_a } else { &buf_b };
        let inv_n = BabyBear::reduce(n as u64).inv();
        let norm_params: Vec<u32> = vec![n as u32, inv_n.0];
        let buf_p = self.ctx.buffer_from_slice(&norm_params)?;
        let max_tpg = MetalContext::max_threads_per_threadgroup(&self.normalize_pipeline) as u64;
        let (tg, tpg) = MetalContext::compute_grid_1d(n as u64, max_tpg.min(256));
        MetalContext::encode_dispatch(
            cmd,
            &self.normalize_pipeline,
            &[result_buf, &buf_p],
            tg,
            tpg,
        );
        retain.push(buf_p);

        let total_ns = MetalContext::submit_batch(cmd, &retain)?;
        let result = MetalContext::read_buffer(result_buf, n);
        Ok((result, total_ns))
    }
}

impl NttBackend<BabyBear> for BbMetalStockhamR2 {
    fn name(&self) -> &str {
        "bb-metal-stockham-r2"
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
    use crate::ntt::babybear::cpu_reference::BbCpuReferenceBackend;
    use crate::ntt::test_utils::shader_dir;

    fn init() -> Option<BbMetalStockhamR2> {
        match BbMetalStockhamR2::new(&shader_dir()) {
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
        for &n in &[4, 16, 256, 1024, 4096, 8192] {
            let original = bb_test_data(n);
            let mut cpu_data = original.clone();
            cpu.forward_ntt(&mut cpu_data, &[]).unwrap();
            let mut gpu_data = original.clone();
            gpu.forward_ntt(&mut gpu_data, &[]).unwrap();
            assert_eq!(gpu_data, cpu_data, "V3 forward mismatch at size {}", n);
        }
    }

    #[test]
    fn test_roundtrip() {
        let gpu = match init() {
            Some(g) => g,
            None => return,
        };
        for &n in &[4, 16, 256, 1024, 4096, 8192] {
            let original = bb_test_data(n);
            let mut data = original.clone();
            gpu.forward_ntt(&mut data, &[]).unwrap();
            assert_ne!(
                data, original,
                "V3 forward should change data at size {}",
                n
            );
            gpu.inverse_ntt(&mut data, &[]).unwrap();
            assert_eq!(data, original, "V3 round-trip failed at size {}", n);
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
    fn test_roundtrip_size4096_tg_boundary() {
        let gpu = match init() {
            Some(g) => g,
            None => return,
        };
        let original = bb_test_data(4096);
        let mut data = original.clone();
        gpu.forward_ntt(&mut data, &[]).unwrap();
        gpu.inverse_ntt(&mut data, &[]).unwrap();
        assert_eq!(data, original, "V3 TG boundary round-trip failed");
    }
}
