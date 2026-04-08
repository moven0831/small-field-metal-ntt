//! BabyBear V1: CT-DIT radix-2 NTT on Metal GPU — naive baseline.
//!
//! One dispatch per butterfly stage, all device memory, no threadgroup optimization.
//! Forward only. Purpose: baseline cost of GPU NTT with zero optimization.
//!
//! Reuses the `bb_r2_butterfly_device` kernel from bb_ntt.metal (same butterfly
//! structure, same position-indexed twiddles).

use crate::field::babybear::BabyBear;
use crate::field::Field;
use crate::gpu::MetalContext;
use crate::ntt::bb_twiddles::BbTwiddleCache;
use crate::ntt::{NttBackend, NttError};
use metal::*;
use std::path::Path;

pub struct BbMetalCtDitR2 {
    ctx: MetalContext,
    butterfly_pipeline: ComputePipelineState,
    twiddle_cache: BbTwiddleCache,
}

impl BbMetalCtDitR2 {
    pub fn new(shader_dir: &Path) -> Result<Self, NttError> {
        let ctx = MetalContext::new(shader_dir)?;
        let butterfly_pipeline = ctx.make_pipeline("bb_r2_butterfly_device")?;

        Ok(Self {
            ctx,
            butterfly_pipeline,
            twiddle_cache: BbTwiddleCache::new(),
        })
    }

    pub fn forward_ntt_gpu(&self, input: &[u32], log_n: usize) -> Result<(Vec<u32>, u64), NttError> {
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

        // One dispatch per layer, high to low (CT-DIT order)
        for layer in (0..log_n).rev() {
            let stride = 1usize << layer;
            let tw_data: Vec<u32> = twiddles[layer].iter().map(|m| m.0).collect();
            let buf_tw = self.ctx.buffer_from_slice(&tw_data)?;
            let params: Vec<u32> = vec![stride as u32, n as u32];
            let buf_p = self.ctx.buffer_from_slice(&params)?;

            let num_butterflies = (n / 2) as u64;
            let max_tpg = MetalContext::max_threads_per_threadgroup(&self.butterfly_pipeline) as u64;
            let (tg, tpg) = MetalContext::compute_grid_1d(num_butterflies, max_tpg.min(256));

            MetalContext::encode_dispatch(
                cmd,
                &self.butterfly_pipeline,
                &[&buf_data, &buf_tw, &buf_p],
                tg,
                tpg,
            );
            retain.push(buf_tw);
            retain.push(buf_p);
        }

        let total_ns = MetalContext::submit_batch(cmd, &retain)?;
        let result = MetalContext::read_buffer(&buf_data, n);
        Ok((result, total_ns))
    }
}

impl NttBackend<BabyBear> for BbMetalCtDitR2 {
    fn name(&self) -> &str {
        "bb-metal-ct-dit-r2"
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

    fn inverse_ntt(&self, _data: &mut [BabyBear], _twiddles: &[BabyBear]) -> Result<(), NttError> {
        todo!("Inverse NTT not implemented for naive CT-DIT baseline")
    }

    fn pointwise_mul(&self, a: &[BabyBear], b: &[BabyBear], out: &mut [BabyBear]) -> Result<(), NttError> {
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

    fn init() -> Option<BbMetalCtDitR2> {
        match BbMetalCtDitR2::new(&shader_dir()) {
            Ok(g) => Some(g),
            Err(NttError::DeviceNotFound) => { eprintln!("No Metal device — skipping"); None }
            Err(e) => panic!("Failed to init: {}", e),
        }
    }

    fn bb_test_data(n: usize) -> Vec<BabyBear> {
        let mut seed: u64 = n as u64 * 11111 + 98765;
        (0..n).map(|_| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            BabyBear::to_monty(((seed >> 33) as u32) % BabyBear::P)
        }).collect()
    }

    #[test]
    fn test_forward_matches_cpu() {
        let gpu = match init() { Some(g) => g, None => return };
        let cpu = BbCpuReferenceBackend::new();
        for &n in &[4, 16, 256, 1024, 4096] {
            let original = bb_test_data(n);
            let mut cpu_data = original.clone();
            cpu.forward_ntt(&mut cpu_data, &[]).unwrap();
            let mut gpu_data = original.clone();
            gpu.forward_ntt(&mut gpu_data, &[]).unwrap();
            assert_eq!(gpu_data, cpu_data, "V1 forward mismatch at size {}", n);
        }
    }

    #[test]
    fn test_edge_cases() {
        let gpu = match init() { Some(g) => g, None => return };
        let mut data = vec![BabyBear::zero(); 64];
        gpu.forward_ntt(&mut data, &[]).unwrap();
        assert!(data.iter().all(|&x| x == BabyBear::zero()));

        let mut data = vec![BabyBear::to_monty(42)];
        gpu.forward_ntt(&mut data, &[]).unwrap();
        assert_eq!(data[0], BabyBear::to_monty(42));

        let mut data = vec![BabyBear::one(); 3];
        assert!(gpu.forward_ntt(&mut data, &[]).is_err());
    }
}
