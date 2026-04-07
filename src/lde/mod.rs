//! Coset Low-Degree Extension (LDE) pipeline on Metal GPU.
//!
//! Pipeline: iDFT_batch → zero_pad → coset_shift → forward_DFT_batch
//!
//! Designed to match zk-autoresearch's `coset_lde_batch` benchmark for
//! direct performance comparison: BabyBear field, 2^20 rows × 256 columns.

use crate::field::babybear::BabyBear;
use crate::field::Field;
use crate::gpu::MetalContext;
use crate::ntt::bb_metal_ct_gs_r4::BbMetalCtGsR4;
use crate::ntt::NttError;
use std::path::Path;

pub struct CosetLdeBatch {
    ntt: BbMetalCtGsR4,
}

/// Result of a coset LDE execution.
pub struct LdeResult {
    /// Output data in SoA layout: `data[col * n_ext + row]`.
    pub data: Vec<u32>,
    /// Total GPU wall-clock time in nanoseconds.
    pub total_ns: u64,
}

impl CosetLdeBatch {
    pub fn new(shader_dir: &Path) -> Result<Self, NttError> {
        let ntt = BbMetalCtGsR4::new(shader_dir)?;
        Ok(Self { ntt })
    }

    /// Execute the full coset LDE pipeline on GPU.
    ///
    /// Input: `batch_size` columns of `n` elements each in SoA layout (Montgomery form).
    /// Output: `batch_size` columns of `n_ext = n << added_bits` elements each.
    ///
    /// Pipeline:
    /// 1. iDFT_batch on input (in-place, size n)
    /// 2. Zero-pad from n to n_ext
    /// 3. Coset-shift: multiply element i by shift^i
    /// 4. Forward DFT_batch on extended buffer (in-place, size n_ext)
    pub fn execute(
        &self,
        input: &[u32],
        log_n: usize,
        batch_size: usize,
        added_bits: usize,
    ) -> Result<LdeResult, NttError> {
        let n = 1usize << log_n;
        let log_n_ext = log_n + added_bits;
        let n_ext = 1usize << log_n_ext;

        if input.len() != n * batch_size {
            return Err(NttError::InvalidSize(input.len()));
        }

        // Precompute coset shift powers on CPU
        let shift = BabyBear::two_adic_generator(log_n_ext as u32 + 1);
        let shift_powers = compute_shift_powers(shift, n_ext);
        let inv_n = BabyBear::reduce(n as u64).inv();

        // Allocate GPU buffers
        let buf_input = self.ntt.ctx().buffer_from_slice(input)?;
        let buf_output = self.ntt.ctx().alloc_buffer(n_ext * batch_size * 4)?;

        let cmd = self.ntt.ctx().begin_batch();
        let mut retain = Vec::new();

        // Step 1: iDFT_batch without normalize (in-place on buf_input)
        self.ntt.encode_inverse_ntt_batch_no_normalize(
            cmd,
            &mut retain,
            &buf_input,
            log_n,
            n,
            batch_size,
        )?;

        // Step 2: Fused normalize + zero-pad + coset-shift (buf_input -> buf_output)
        self.ntt.encode_fused_norm_zeropad_shift(
            cmd,
            &mut retain,
            &buf_input,
            &buf_output,
            &shift_powers,
            n,
            n_ext,
            batch_size,
            inv_n,
        )?;

        // Step 3: Forward DFT_batch on buf_output (in-place, size n_ext)
        self.ntt.encode_forward_ntt_batch(
            cmd,
            &mut retain,
            &buf_output,
            log_n_ext,
            n_ext,
            batch_size,
        )?;

        let total_ns = MetalContext::submit_batch(cmd, &retain)?;
        let data = MetalContext::read_buffer(&buf_output, n_ext * batch_size);

        Ok(LdeResult { data, total_ns })
    }

    /// Get device info for benchmark reporting.
    pub fn device_info(&self) -> crate::gpu::DeviceInfo {
        self.ntt.ctx().device_info()
    }
}

/// Compute shift^i for i in 0..n, returned as raw u32 (Montgomery form).
fn compute_shift_powers(shift: BabyBear, n: usize) -> Vec<u32> {
    let mut powers = Vec::with_capacity(n);
    let mut current = BabyBear::one();
    for _ in 0..n {
        powers.push(current.0);
        current = current.mul(shift);
    }
    powers
}

/// CPU reference implementation for correctness verification.
pub fn coset_lde_batch_cpu(
    input: &[u32],
    log_n: usize,
    batch_size: usize,
    added_bits: usize,
) -> Vec<u32> {
    use crate::ntt::bb_cpu_reference::BbCpuReferenceBackend;

    let n = 1usize << log_n;
    let log_n_ext = log_n + added_bits;
    let n_ext = 1usize << log_n_ext;
    let cpu = BbCpuReferenceBackend::new();

    // Step 1: iDFT each column
    let mut coeffs = input.to_vec();
    for col in 0..batch_size {
        let start = col * n;
        let end = start + n;
        cpu.inverse_ntt_u32(&mut coeffs[start..end]).unwrap();
    }

    // Step 2: Zero-pad
    let mut extended = vec![0u32; n_ext * batch_size];
    for col in 0..batch_size {
        let src_start = col * n;
        let dst_start = col * n_ext;
        extended[dst_start..dst_start + n].copy_from_slice(&coeffs[src_start..src_start + n]);
        // remaining elements are already 0
    }

    // Step 3: Coset shift
    let shift = BabyBear::two_adic_generator(log_n_ext as u32 + 1);
    let mut power = BabyBear::one();
    for i in 0..n_ext {
        for col in 0..batch_size {
            let idx = col * n_ext + i;
            extended[idx] = BabyBear(extended[idx]).mul(power).0;
        }
        power = power.mul(shift);
    }

    // Step 4: Forward DFT each column
    for col in 0..batch_size {
        let start = col * n_ext;
        let end = start + n_ext;
        cpu.forward_ntt_u32(&mut extended[start..end]).unwrap();
    }

    extended
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ntt::test_utils::shader_dir;

    fn init() -> Option<CosetLdeBatch> {
        match CosetLdeBatch::new(&shader_dir()) {
            Ok(g) => Some(g),
            Err(NttError::DeviceNotFound) => {
                eprintln!("No Metal device — skipping");
                None
            }
            Err(e) => panic!("Failed to init CosetLdeBatch: {}", e),
        }
    }

    fn test_data(n: usize, batch_size: usize) -> Vec<u32> {
        let mut seed: u64 = 42;
        (0..n * batch_size)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                BabyBear::to_monty(((seed >> 33) as u32) % BabyBear::P).0
            })
            .collect()
    }

    #[test]
    fn test_coset_lde_small() {
        let gpu_lde = match init() {
            Some(g) => g,
            None => return,
        };

        let log_n = 8; // n = 256
        let batch_size = 4;
        let added_bits = 1; // 2x expansion

        let input = test_data(1 << log_n, batch_size);
        let cpu_result = coset_lde_batch_cpu(&input, log_n, batch_size, added_bits);
        let gpu_result = gpu_lde.execute(&input, log_n, batch_size, added_bits).unwrap();

        assert_eq!(
            gpu_result.data, cpu_result,
            "GPU coset LDE doesn't match CPU reference"
        );
    }

    #[test]
    fn test_coset_lde_medium() {
        let gpu_lde = match init() {
            Some(g) => g,
            None => return,
        };

        let log_n = 12; // n = 4096
        let batch_size = 8;
        let added_bits = 1;

        let input = test_data(1 << log_n, batch_size);
        let cpu_result = coset_lde_batch_cpu(&input, log_n, batch_size, added_bits);
        let gpu_result = gpu_lde.execute(&input, log_n, batch_size, added_bits).unwrap();

        assert_eq!(
            gpu_result.data, cpu_result,
            "GPU coset LDE doesn't match CPU at n=4096, batch=8"
        );
    }

    #[test]
    fn test_coset_lde_4x_expansion() {
        let gpu_lde = match init() {
            Some(g) => g,
            None => return,
        };

        let log_n = 8;
        let batch_size = 4;
        let added_bits = 2; // 4x expansion

        let input = test_data(1 << log_n, batch_size);
        let cpu_result = coset_lde_batch_cpu(&input, log_n, batch_size, added_bits);
        let gpu_result = gpu_lde.execute(&input, log_n, batch_size, added_bits).unwrap();

        assert_eq!(
            gpu_result.data, cpu_result,
            "GPU coset LDE 4x doesn't match CPU"
        );
    }
}
