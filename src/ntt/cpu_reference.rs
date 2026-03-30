use crate::field::m31::M31;
use crate::field::circle::Coset;
use crate::field::Field;
use crate::ntt::{NttBackend, NttError};

/// CPU reference implementation of the Circle FFT (CFFT / DCCT) over M31.
///
/// Algorithm (following Stwo's implementation):
///
/// ```text
/// FORWARD (evaluate): coefficients -> evaluations
///   For each layer from (log_size - 1) down to 0:
///     Apply CT-DIT butterflies: v0' = v0 + t*v1, v1' = v0 - t*v1
///   Output is in bit-reversed order.
///
/// INVERSE (interpolate): evaluations -> coefficients
///   For each layer from 0 up to (log_size - 1):
///     Apply GS-DIF butterflies: v0' = v0 + v1, v1' = (v0 - v1) * t_inv
///   Normalize by dividing all values by 2^log_size.
/// ```
///
/// Twiddle factors are x-coordinates of coset points at each layer.
/// The coset doubles after each layer (x,y) -> (2x^2-1, 2xy).
///
/// Not optimized for performance. Clarity is the priority.
pub struct CpuReferenceBackend;

impl NttBackend<M31> for CpuReferenceBackend {
    fn name(&self) -> &str {
        "cpu-reference"
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

        // Generate twiddles from the canonic coset
        let coset = Coset::odds(log_n as u32);
        let twiddles = generate_twiddles(&coset);

        // Forward FFT: process layers from last to first
        for layer in (0..log_n).rev() {
            let half = 1 << layer;
            let full = half << 1;
            let tw = &twiddles[layer];

            for block in 0..(n / full) {
                let t = tw[block];
                for j in 0..half {
                    let idx0 = block * full + j;
                    let idx1 = idx0 + half;
                    butterfly_forward(data, idx0, idx1, t);
                }
            }
        }

        Ok(())
    }

    fn inverse_ntt(&self, data: &mut [M31], _twiddles: &[M31]) -> Result<(), NttError> {
        let n = data.len();
        if n == 0 || (n & (n - 1)) != 0 {
            return Err(NttError::InvalidSize(n));
        }
        if n == 1 {
            return Ok(());
        }
        let log_n = n.trailing_zeros() as usize;

        // Generate inverse twiddles from the canonic coset
        let coset = Coset::odds(log_n as u32);
        let itwiddles = generate_itwiddles(&coset);

        // Inverse FFT: process layers from first to last
        for layer in 0..log_n {
            let half = 1 << layer;
            let full = half << 1;
            let tw = &itwiddles[layer];

            for block in 0..(n / full) {
                let t = tw[block];
                for j in 0..half {
                    let idx0 = block * full + j;
                    let idx1 = idx0 + half;
                    butterfly_inverse(data, idx0, idx1, t);
                }
            }
        }

        // Normalize: divide by n = 2^log_n
        // In M31, dividing by 2 is multiplying by (p+1)/2 = 2^30.
        // Dividing by 2^k is multiplying by 2^(30*k) mod p.
        // Simpler: inv_n = M31::reduce(n as u64).inv()
        let inv_n = M31::reduce(n as u64).inv();
        for val in data.iter_mut() {
            *val = val.mul(inv_n);
        }

        Ok(())
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

/// Apply forward butterfly (CT-DIT) at indices idx0, idx1:
///   data[idx0]' = data[idx0] + twiddle * data[idx1]
///   data[idx1]' = data[idx0] - twiddle * data[idx1]
#[inline(always)]
fn butterfly_forward(data: &mut [M31], idx0: usize, idx1: usize, twiddle: M31) {
    let tmp = data[idx1].mul(twiddle);
    let v0 = data[idx0];
    data[idx0] = v0.add(tmp);
    data[idx1] = v0.sub(tmp);
}

/// Apply inverse butterfly (GS-DIF) at indices idx0, idx1:
///   data[idx0]' = data[idx0] + data[idx1]
///   data[idx1]' = (data[idx0] - data[idx1]) * twiddle_inv
#[inline(always)]
fn butterfly_inverse(data: &mut [M31], idx0: usize, idx1: usize, twiddle_inv: M31) {
    let v0 = data[idx0];
    let v1 = data[idx1];
    data[idx0] = v0.add(v1);
    data[idx1] = v0.sub(v1).mul(twiddle_inv);
}

/// Generate forward twiddle factors for each layer.
///
/// The Circle FFT has two types of layers:
/// - Line layers (0..log_n-1): use x-coordinates of coset points
/// - Circle layer (last, index log_n-1): use y-coordinates
///
/// The coset doubles after each layer. At the deepest layer, the doubled
/// coset has x=0 (since 2*(2^15)^2 - 1 = 0 mod p for M31), so we must
/// use y-coordinates instead.
fn generate_twiddles(coset: &Coset) -> Vec<Vec<M31>> {
    let mut result = Vec::new();
    let mut current = coset.clone();
    let log_n = coset.log_size;

    for layer in 0..log_n {
        let half_size = current.size() / 2;
        let is_last_layer = layer == log_n - 1;

        let layer_twiddles: Vec<M31> = (0..half_size)
            .map(|i| {
                let p = current.at(bit_reverse(i, current.log_size - 1));
                if is_last_layer { p.y } else { p.x }
            })
            .collect();
        result.push(layer_twiddles);
        current = current.double();
    }

    result
}

/// Generate inverse twiddle factors.
/// Same structure as forward but with inverted twiddle values.
fn generate_itwiddles(coset: &Coset) -> Vec<Vec<M31>> {
    let mut result = Vec::new();
    let mut current = coset.clone();
    let log_n = coset.log_size;

    for layer in 0..log_n {
        let half_size = current.size() / 2;
        let is_last_layer = layer == log_n - 1;

        let layer_twiddles: Vec<M31> = (0..half_size)
            .map(|i| {
                let p = current.at(bit_reverse(i, current.log_size - 1));
                let t = if is_last_layer { p.y } else { p.x };
                t.inv()
            })
            .collect();
        result.push(layer_twiddles);
        current = current.double();
    }

    result
}

fn bit_reverse(index: usize, log_size: u32) -> usize {
    reverse_bits(index as u32, log_size) as usize
}

fn reverse_bits(mut val: u32, width: u32) -> u32 {
    let mut result = 0u32;
    for _ in 0..width {
        result = (result << 1) | (val & 1);
        val >>= 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::m31::M31;

    #[test]
    fn test_forward_inverse_roundtrip_small() {
        let backend = CpuReferenceBackend;
        let original: Vec<M31> = (0..4).map(|i| M31(i + 1)).collect();
        let mut data = original.clone();

        backend.forward_ntt(&mut data, &[]).unwrap();
        // After forward, data should be different from original
        assert_ne!(data, original, "Forward NTT should change the data");

        backend.inverse_ntt(&mut data, &[]).unwrap();
        // After inverse, should recover original
        assert_eq!(data, original, "Round-trip should recover original");
    }

    #[test]
    fn test_forward_inverse_roundtrip_medium() {
        let backend = CpuReferenceBackend;
        // Use 2^8 = 256 elements
        let original: Vec<M31> = (0..256).map(|i| M31(i * 7 + 3)).collect();
        let mut data = original.clone();

        backend.forward_ntt(&mut data, &[]).unwrap();
        assert_ne!(data, original);

        backend.inverse_ntt(&mut data, &[]).unwrap();
        assert_eq!(data, original, "Round-trip failed at size 256");
    }

    #[test]
    fn test_forward_inverse_roundtrip_1024() {
        let backend = CpuReferenceBackend;
        let original: Vec<M31> = (0..1024).map(|i| M31((i * 13 + 7) % M31::P)).collect();
        let mut data = original.clone();

        backend.forward_ntt(&mut data, &[]).unwrap();
        backend.inverse_ntt(&mut data, &[]).unwrap();
        assert_eq!(data, original, "Round-trip failed at size 1024");
    }

    #[test]
    fn test_all_zeros() {
        let backend = CpuReferenceBackend;
        let mut data = vec![M31(0); 16];
        backend.forward_ntt(&mut data, &[]).unwrap();
        assert!(data.iter().all(|&x| x == M31(0)), "NTT of zeros should be zeros");
    }

    #[test]
    fn test_size_one() {
        let backend = CpuReferenceBackend;
        let mut data = vec![M31(42)];
        backend.forward_ntt(&mut data, &[]).unwrap();
        assert_eq!(data[0], M31(42));
    }

    #[test]
    fn test_invalid_size() {
        let backend = CpuReferenceBackend;
        let mut data = vec![M31(1); 3]; // Not power of 2
        assert!(backend.forward_ntt(&mut data, &[]).is_err());
    }

    #[test]
    fn test_invalid_size_zero() {
        let backend = CpuReferenceBackend;
        let mut data: Vec<M31> = vec![];
        assert!(backend.forward_ntt(&mut data, &[]).is_err());
    }

    #[test]
    fn test_pointwise_mul() {
        let backend = CpuReferenceBackend;
        let a = vec![M31(2), M31(3), M31(4), M31(5)];
        let b = vec![M31(10), M31(20), M31(30), M31(40)];
        let mut out = vec![M31(0); 4];
        backend.pointwise_mul(&a, &b, &mut out).unwrap();
        assert_eq!(out, vec![M31(20), M31(60), M31(120), M31(200)]);
    }

    #[test]
    fn test_linearity() {
        // NTT should be linear: NTT(a + b) = NTT(a) + NTT(b)
        let backend = CpuReferenceBackend;
        let n = 16;
        let a: Vec<M31> = (0..n).map(|i| M31(i as u32 + 1)).collect();
        let b: Vec<M31> = (0..n).map(|i| M31(i as u32 * 3 + 2)).collect();

        // Compute NTT(a), NTT(b)
        let mut ntt_a = a.clone();
        let mut ntt_b = b.clone();
        backend.forward_ntt(&mut ntt_a, &[]).unwrap();
        backend.forward_ntt(&mut ntt_b, &[]).unwrap();

        // Compute NTT(a + b)
        let mut a_plus_b: Vec<M31> = a.iter().zip(b.iter()).map(|(&x, &y)| x.add(y)).collect();
        backend.forward_ntt(&mut a_plus_b, &[]).unwrap();

        // NTT(a) + NTT(b)
        let sum: Vec<M31> = ntt_a.iter().zip(ntt_b.iter()).map(|(&x, &y)| x.add(y)).collect();

        assert_eq!(a_plus_b, sum, "NTT should be linear");
    }

    #[test]
    fn test_roundtrip_random_large() {
        let backend = CpuReferenceBackend;
        // Pseudo-random data using a simple LCG
        let n = 4096;
        let mut seed: u64 = 12345;
        let original: Vec<M31> = (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                M31(((seed >> 33) as u32) % M31::P)
            })
            .collect();

        let mut data = original.clone();
        backend.forward_ntt(&mut data, &[]).unwrap();
        backend.inverse_ntt(&mut data, &[]).unwrap();
        assert_eq!(data, original, "Round-trip failed at size 4096");
    }
}
