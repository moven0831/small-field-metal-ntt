//! CPU reference implementation of standard radix-2 NTT over BabyBear.
//!
//! Uses GS-DIF for forward (natural input → bit-reversed output) and
//! CT-DIT for inverse (bit-reversed input → natural output, normalized by 1/n).
//! Same algorithm as the Metal GPU variants. Not optimized; clarity is the priority.

use crate::field::babybear::BabyBear;
use crate::field::Field;
use crate::ntt::bb_twiddles::BbTwiddleCache;
use crate::ntt::{NttBackend, NttError};

pub struct BbCpuReferenceBackend {
    twiddle_cache: BbTwiddleCache,
}

impl Default for BbCpuReferenceBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl BbCpuReferenceBackend {
    pub fn new() -> Self {
        Self {
            twiddle_cache: crate::ntt::bb_twiddles::new_bb_twiddle_cache(),
        }
    }

    /// Forward NTT on raw u32 data (Montgomery form).
    ///
    /// Uses GS-DIF: natural-order input → bit-reversed output.
    /// Layers processed from large stride to small stride.
    #[allow(clippy::needless_range_loop)]
    pub fn forward_ntt_u32(&self, data: &mut [u32]) -> Result<(), NttError> {
        let n = data.len();
        if n == 0 || (n & (n - 1)) != 0 {
            return Err(NttError::InvalidSize(n));
        }
        if n == 1 {
            return Ok(());
        }
        let log_n = n.trailing_zeros() as usize;
        let twiddles = self.twiddle_cache.forward(log_n as u32);

        for layer in (0..log_n).rev() {
            let half = 1 << layer;
            let full = half << 1;
            let tw = &twiddles[layer];

            for block in 0..(n / full) {
                for j in 0..half {
                    let idx0 = block * full + j;
                    let idx1 = idx0 + half;
                    let v0 = BabyBear(data[idx0]);
                    let v1 = BabyBear(data[idx1]);
                    data[idx0] = v0.add(v1).0;
                    data[idx1] = v0.sub(v1).mul(tw[j]).0;
                }
            }
        }
        Ok(())
    }

    /// Inverse NTT on raw u32 data (Montgomery form).
    ///
    /// Uses CT-DIT: bit-reversed input → natural-order output.
    /// Layers processed from small stride to large stride.
    #[allow(clippy::needless_range_loop)]
    pub fn inverse_ntt_u32(&self, data: &mut [u32]) -> Result<(), NttError> {
        let n = data.len();
        if n == 0 || (n & (n - 1)) != 0 {
            return Err(NttError::InvalidSize(n));
        }
        if n == 1 {
            return Ok(());
        }
        let log_n = n.trailing_zeros() as usize;
        let itwiddles = self.twiddle_cache.inverse(log_n as u32);

        for layer in 0..log_n {
            let half = 1 << layer;
            let full = half << 1;
            let tw = &itwiddles[layer];

            for block in 0..(n / full) {
                for j in 0..half {
                    let idx0 = block * full + j;
                    let idx1 = idx0 + half;
                    let tmp = BabyBear(data[idx1]).mul(tw[j]);
                    let v0 = BabyBear(data[idx0]);
                    data[idx0] = v0.add(tmp).0;
                    data[idx1] = v0.sub(tmp).0;
                }
            }
        }

        // Normalize by inv_n
        let inv_n = BabyBear::reduce(n as u64).inv();
        for val in data.iter_mut() {
            *val = BabyBear(*val).mul(inv_n).0;
        }
        Ok(())
    }
}

impl NttBackend<BabyBear> for BbCpuReferenceBackend {
    fn name(&self) -> &str {
        "bb-cpu-reference"
    }

    #[allow(clippy::needless_range_loop)]
    fn forward_ntt(&self, data: &mut [BabyBear], _twiddles: &[BabyBear]) -> Result<(), NttError> {
        let n = data.len();
        if n == 0 || (n & (n - 1)) != 0 {
            return Err(NttError::InvalidSize(n));
        }
        if n == 1 {
            return Ok(());
        }
        let log_n = n.trailing_zeros() as usize;
        let twiddles = self.twiddle_cache.forward(log_n as u32);

        // GS-DIF: natural input → bit-reversed output
        for layer in (0..log_n).rev() {
            let half = 1 << layer;
            let full = half << 1;
            let tw = &twiddles[layer];

            for block in 0..(n / full) {
                for j in 0..half {
                    let idx0 = block * full + j;
                    let idx1 = idx0 + half;
                    let v0 = data[idx0];
                    let v1 = data[idx1];
                    data[idx0] = v0.add(v1);
                    data[idx1] = v0.sub(v1).mul(tw[j]);
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::needless_range_loop)]
    fn inverse_ntt(&self, data: &mut [BabyBear], _twiddles: &[BabyBear]) -> Result<(), NttError> {
        let n = data.len();
        if n == 0 || (n & (n - 1)) != 0 {
            return Err(NttError::InvalidSize(n));
        }
        if n == 1 {
            return Ok(());
        }
        let log_n = n.trailing_zeros() as usize;
        let itwiddles = self.twiddle_cache.inverse(log_n as u32);

        // CT-DIT: bit-reversed input → natural output
        for layer in 0..log_n {
            let half = 1 << layer;
            let full = half << 1;
            let tw = &itwiddles[layer];

            for block in 0..(n / full) {
                for j in 0..half {
                    let idx0 = block * full + j;
                    let idx1 = idx0 + half;
                    let tmp = data[idx1].mul(tw[j]);
                    let v0 = data[idx0];
                    data[idx0] = v0.add(tmp);
                    data[idx1] = v0.sub(tmp);
                }
            }
        }

        let inv_n = BabyBear::reduce(n as u64).inv();
        for val in data.iter_mut() {
            *val = val.mul(inv_n);
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

    #[test]
    fn test_roundtrip_small() {
        let backend = BbCpuReferenceBackend::new();
        let original: Vec<BabyBear> = (1..=4).map(BabyBear::to_monty).collect();
        let mut data = original.clone();

        backend.forward_ntt(&mut data, &[]).unwrap();
        assert_ne!(data, original);
        backend.inverse_ntt(&mut data, &[]).unwrap();
        assert_eq!(data, original);
    }

    #[test]
    fn test_roundtrip_256() {
        let backend = BbCpuReferenceBackend::new();
        let original: Vec<BabyBear> = (0..256)
            .map(|i| BabyBear::to_monty((i * 7 + 3) % BabyBear::P))
            .collect();
        let mut data = original.clone();

        backend.forward_ntt(&mut data, &[]).unwrap();
        backend.inverse_ntt(&mut data, &[]).unwrap();
        assert_eq!(data, original);
    }

    #[test]
    fn test_roundtrip_1024() {
        let backend = BbCpuReferenceBackend::new();
        let original: Vec<BabyBear> = (0..1024)
            .map(|i| BabyBear::to_monty((i * 13 + 7) % BabyBear::P))
            .collect();
        let mut data = original.clone();

        backend.forward_ntt(&mut data, &[]).unwrap();
        backend.inverse_ntt(&mut data, &[]).unwrap();
        assert_eq!(data, original);
    }

    #[test]
    fn test_u32_roundtrip() {
        let backend = BbCpuReferenceBackend::new();
        let original: Vec<u32> = (1..=16).map(|i| BabyBear::to_monty(i).0).collect();
        let mut data = original.clone();

        backend.forward_ntt_u32(&mut data).unwrap();
        assert_ne!(data, original);
        backend.inverse_ntt_u32(&mut data).unwrap();
        assert_eq!(data, original);
    }

    #[test]
    fn test_linearity() {
        let backend = BbCpuReferenceBackend::new();
        let n = 16;
        let a: Vec<BabyBear> = (0..n).map(|i| BabyBear::to_monty(i as u32 + 1)).collect();
        let b: Vec<BabyBear> = (0..n)
            .map(|i| BabyBear::to_monty(i as u32 * 3 + 2))
            .collect();

        let mut ntt_a = a.clone();
        let mut ntt_b = b.clone();
        backend.forward_ntt(&mut ntt_a, &[]).unwrap();
        backend.forward_ntt(&mut ntt_b, &[]).unwrap();

        let mut a_plus_b: Vec<BabyBear> = a.iter().zip(b.iter()).map(|(&x, &y)| x.add(y)).collect();
        backend.forward_ntt(&mut a_plus_b, &[]).unwrap();

        let sum: Vec<BabyBear> = ntt_a
            .iter()
            .zip(ntt_b.iter())
            .map(|(&x, &y)| x.add(y))
            .collect();
        assert_eq!(a_plus_b, sum, "NTT should be linear");
    }

    #[test]
    fn test_all_zeros() {
        let backend = BbCpuReferenceBackend::new();
        let mut data = vec![BabyBear::zero(); 16];
        backend.forward_ntt(&mut data, &[]).unwrap();
        assert!(data.iter().all(|&x| x == BabyBear::zero()));
    }

    #[test]
    fn test_size_one() {
        let backend = BbCpuReferenceBackend::new();
        let mut data = vec![BabyBear::to_monty(42)];
        backend.forward_ntt(&mut data, &[]).unwrap();
        assert_eq!(data[0], BabyBear::to_monty(42));
    }

    #[test]
    fn test_invalid_size() {
        let backend = BbCpuReferenceBackend::new();
        let mut data = vec![BabyBear::one(); 3];
        assert!(backend.forward_ntt(&mut data, &[]).is_err());
    }
}
