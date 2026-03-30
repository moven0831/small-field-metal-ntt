use crate::field::Field;
use crate::ntt::{NttBackend, NttError};

/// CPU reference implementation of Circle NTT over M31.
///
/// Uses the standard DCCT butterfly: a' = a + w*b, b' = a - w*b (CT-DIT forward).
/// This serves as the correctness oracle for all GPU variants.
///
/// Not optimized for performance — clarity is the priority.
pub struct CpuReferenceBackend;

impl<F: Field> NttBackend<F> for CpuReferenceBackend {
    fn name(&self) -> &str {
        "cpu-reference"
    }

    fn forward_ntt(&self, _data: &mut [F], _twiddles: &[F]) -> Result<(), NttError> {
        // TODO: Implement Circle NTT (DCCT) forward transform.
        //
        // Algorithm (from Stwo / Circle STARK paper):
        //   1. For each pair (a, b) with stride 2^stage:
        //      a' = a + twiddle * b
        //      b' = a - twiddle * b
        //   2. Twiddle factors are circle-group generators:
        //      - Layer 0: y-coordinates of half-coset points
        //      - Layer 1: x-coordinates of first half
        //      - Layer k (k>=2): doubling map 2*x^2 - 1 applied recursively
        //
        // The butterfly structure is identical to standard NTT after step 1.
        todo!("Implement CPU reference Circle NTT")
    }

    fn inverse_ntt(&self, _data: &mut [F], _twiddles: &[F]) -> Result<(), NttError> {
        // TODO: Implement Circle NTT (iDCCT) inverse transform.
        //
        // Uses GS-DIF butterfly: a' = a + b, b' = (a - b) * twiddle_inv
        todo!("Implement CPU reference inverse Circle NTT")
    }

    fn pointwise_mul(&self, a: &[F], b: &[F], out: &mut [F]) -> Result<(), NttError> {
        if a.len() != b.len() || a.len() != out.len() {
            return Err(NttError::InvalidSize(a.len()));
        }
        for i in 0..a.len() {
            out[i] = a[i].mul(b[i]);
        }
        Ok(())
    }
}
