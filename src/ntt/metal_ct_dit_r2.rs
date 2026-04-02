//! Variant 1: CT-DIT radix-2 NTT on Metal GPU — naive baseline.
//!
//! Uses the same algorithm as the CPU reference (layers from log_n-1 to 0),
//! but executes each butterfly stage as a separate GPU kernel dispatch.
//!
//! This is intentionally unoptimized:
//! - One dispatch per butterfly stage (no stage fusion)
//! - All data in device memory (no threadgroup optimization)
//! - New twiddle buffer allocated per stage
//!
//! Total dispatches: log_n.
//! Purpose: show the baseline cost of GPU NTT with zero optimization.

use crate::field::circle::Coset;
use crate::field::m31::M31;
use crate::field::Field;
use crate::gpu::MetalContext;
use crate::ntt::{NttBackend, NttError};
use crate::ntt::twiddles::generate_twiddles;
use metal::*;
use std::path::Path;

pub struct MetalCtDitR2 {
    ctx: MetalContext,
    butterfly_pipeline: ComputePipelineState,
}

impl MetalCtDitR2 {
    pub fn new(shader_dir: &Path) -> Result<Self, NttError> {
        let ctx = MetalContext::new(shader_dir)?;
        let butterfly_pipeline = ctx.make_pipeline("ct_dit_r2_butterfly_stage")?;

        Ok(MetalCtDitR2 {
            ctx,
            butterfly_pipeline,
        })
    }

    /// Run forward Circle NTT on GPU.
    /// Returns (result_data, total_gpu_time_ns).
    pub fn forward_ntt_gpu(&self, input: &[u32], log_n: usize) -> Result<(Vec<u32>, u64), NttError> {
        let n = input.len();
        if n != (1 << log_n) {
            return Err(NttError::InvalidSize(n));
        }
        if log_n > 30 {
            return Err(NttError::InvalidSize(n)); // n must fit in u32 for GPU params
        }

        // Generate twiddles on CPU (same as cpu_reference)
        let coset = Coset::odds(log_n as u32);
        let twiddles = generate_twiddles(&coset);

        let buf_data = self.ctx.buffer_from_slice(input)?;
        let mut total_ns: u64 = 0;

        // Query max threadgroup size for this pipeline
        let max_tpg = MetalContext::max_threads_per_threadgroup(&self.butterfly_pipeline) as u64;
        let tpg_width = max_tpg.min(256);

        // Process layers from log_n-1 down to 0 (same order as CPU reference forward)
        for layer in (0..log_n).rev() {
            let stride = 1usize << layer;
            let num_butterflies = (n / 2) as u32;

            let tw_data: Vec<u32> = twiddles[layer].iter().map(|m| m.0).collect();
            let buf_tw = self.ctx.buffer_from_slice(&tw_data)?;

            let params: Vec<u32> = vec![stride as u32, n as u32];
            let buf_p = self.ctx.buffer_from_slice(&params)?;

            let tg = MTLSize::new(((num_butterflies as u64) + tpg_width - 1) / tpg_width, 1, 1);
            let tpg = MTLSize::new(tpg_width.min(num_butterflies as u64), 1, 1);

            let ns = self.ctx.dispatch_and_wait(
                &self.butterfly_pipeline,
                &[&buf_data, &buf_tw, &buf_p],
                tg,
                tpg,
            )?;
            total_ns += ns;
        }

        let result = MetalContext::read_buffer(&buf_data, n);
        Ok((result, total_ns))
    }
}

impl NttBackend<M31> for MetalCtDitR2 {
    fn name(&self) -> &str {
        "metal-ct-dit-r2"
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
        let (result, _timing) = self.forward_ntt_gpu(&input, log_n)?;

        for (i, val) in result.iter().enumerate() {
            data[i] = M31(*val);
        }
        Ok(())
    }

    fn inverse_ntt(&self, _data: &mut [M31], _twiddles: &[M31]) -> Result<(), NttError> {
        todo!("Inverse NTT not implemented for naive CT-DIT baseline")
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
    use crate::ntt::cpu_reference::CpuReferenceBackend;
    use crate::ntt::NttBackend;
    use std::path::PathBuf;

    fn shader_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders")
    }

    fn skip_if_no_metal() -> Option<MetalCtDitR2> {
        match MetalCtDitR2::new(&shader_dir()) {
            Ok(g) => Some(g),
            Err(NttError::DeviceNotFound) => {
                eprintln!("No Metal device — skipping");
                None
            }
            Err(e) => panic!("Failed: {}", e),
        }
    }

    #[test]
    fn test_gpu_ntt_matches_cpu_size4() {
        let gpu = match skip_if_no_metal() { Some(g) => g, None => return };
        let cpu = CpuReferenceBackend;

        let original = vec![M31(1), M31(2), M31(3), M31(4)];
        let mut cpu_data = original.clone();
        cpu.forward_ntt(&mut cpu_data, &[]).unwrap();
        let mut gpu_data = original.clone();
        gpu.forward_ntt(&mut gpu_data, &[]).unwrap();

        assert_eq!(gpu_data, cpu_data, "GPU NTT mismatch at size 4");
    }

    #[test]
    fn test_gpu_ntt_matches_cpu_size16() {
        let gpu = match skip_if_no_metal() { Some(g) => g, None => return };
        let cpu = CpuReferenceBackend;

        let n = 16;
        let original: Vec<M31> = (0..n).map(|i| M31(i as u32 * 7 + 3)).collect();
        let mut cpu_data = original.clone();
        cpu.forward_ntt(&mut cpu_data, &[]).unwrap();
        let mut gpu_data = original.clone();
        gpu.forward_ntt(&mut gpu_data, &[]).unwrap();

        assert_eq!(gpu_data, cpu_data, "GPU NTT mismatch at size {}", n);
    }

    #[test]
    fn test_gpu_ntt_matches_cpu_size256() {
        let gpu = match skip_if_no_metal() { Some(g) => g, None => return };
        let cpu = CpuReferenceBackend;

        let n = 256;
        let original: Vec<M31> = (0..n).map(|i| M31((i as u32 * 13 + 7) % M31::P)).collect();
        let mut cpu_data = original.clone();
        cpu.forward_ntt(&mut cpu_data, &[]).unwrap();
        let mut gpu_data = original.clone();
        gpu.forward_ntt(&mut gpu_data, &[]).unwrap();

        assert_eq!(gpu_data, cpu_data, "GPU NTT mismatch at size {}", n);
    }

    #[test]
    fn test_gpu_ntt_matches_cpu_size1024() {
        let gpu = match skip_if_no_metal() { Some(g) => g, None => return };
        let cpu = CpuReferenceBackend;

        let n = 1024;
        let original: Vec<M31> = (0..n).map(|i| M31((i as u32 * 17 + 11) % M31::P)).collect();
        let mut cpu_data = original.clone();
        cpu.forward_ntt(&mut cpu_data, &[]).unwrap();
        let mut gpu_data = original.clone();
        gpu.forward_ntt(&mut gpu_data, &[]).unwrap();

        assert_eq!(gpu_data, cpu_data, "GPU NTT mismatch at size {}", n);
    }

    #[test]
    fn test_gpu_ntt_matches_cpu_size4096() {
        let gpu = match skip_if_no_metal() { Some(g) => g, None => return };
        let cpu = CpuReferenceBackend;

        let n = 4096;
        let mut seed: u64 = 98765;
        let original: Vec<M31> = (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                M31(((seed >> 33) as u32) % M31::P)
            })
            .collect();

        let mut cpu_data = original.clone();
        cpu.forward_ntt(&mut cpu_data, &[]).unwrap();
        let mut gpu_data = original.clone();
        gpu.forward_ntt(&mut gpu_data, &[]).unwrap();

        assert_eq!(gpu_data, cpu_data, "GPU NTT mismatch at size {}", n);
    }

    #[test]
    fn test_gpu_ntt_all_zeros() {
        let gpu = match skip_if_no_metal() { Some(g) => g, None => return };
        let mut data = vec![M31(0); 64];
        gpu.forward_ntt(&mut data, &[]).unwrap();
        assert!(data.iter().all(|&x| x == M31(0)));
    }
}
