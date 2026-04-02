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

        // Process layers from log_n-1 down to 0 (same order as CPU reference forward)
        for layer in (0..log_n).rev() {
            let stride = 1usize << layer;
            let ns = self.ctx.dispatch_butterfly_r2(
                &self.butterfly_pipeline,
                &buf_data,
                &twiddles[layer],
                stride,
                n,
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
    use crate::ntt::test_utils::*;

    fn init() -> Option<MetalCtDitR2> {
        try_init_metal(|p| MetalCtDitR2::new(p))
    }

    #[test]
    fn test_forward_matches_cpu() {
        let gpu = match init() { Some(g) => g, None => return };
        assert_forward_matches_cpu(&gpu, &[4, 16, 256, 1024, 4096]);
    }

    #[test]
    fn test_forward_edge_cases() {
        let gpu = match init() { Some(g) => g, None => return };
        // V1 is forward-only (inverse is todo!()), so only test forward edge cases
        assert_edge_cases(&gpu);
    }
}
