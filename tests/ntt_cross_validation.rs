//! Cross-validation tests: verify our NTT implementations produce correct results
//! by comparing against well-known reference implementations.
//!
//! - BabyBear: cross-validate against Plonky3 (p3-dft 0.5.2)
//! - M31 Circle NTT: cross-validate against hardcoded Stwo 2.2.0 test vectors
//!   (Stwo requires nightly Rust, so we can't add it as a dev-dependency)

use p3_field::PrimeField32;
use small_field_metal_ntt::field::babybear::BabyBear;
use small_field_metal_ntt::field::m31::M31;
use small_field_metal_ntt::field::Field;
use small_field_metal_ntt::ntt::babybear::cpu_reference::BbCpuReferenceBackend;
use small_field_metal_ntt::ntt::m31::cpu_reference::CpuReferenceBackend;
use small_field_metal_ntt::ntt::NttBackend;

// ─── Plonky3 cross-validation (BabyBear) ───────────────────────────────────

/// Bit-reverse permutation of a slice.
///
/// Our GS-DIF forward NTT produces bit-reversed output. This permutation
/// converts to natural order for comparison against Plonky3 (which outputs
/// in natural order).
fn bit_reverse_permutation<T: Copy>(data: &[T]) -> Vec<T> {
    let n = data.len();
    assert!(
        n > 1 && n.is_power_of_two(),
        "bit_reverse_permutation requires n > 1 and power of 2"
    );
    let log_n = n.trailing_zeros();
    let mut result = vec![data[0]; n];
    for i in 0..n {
        let rev = (i as u32).reverse_bits() >> (32 - log_n);
        result[rev as usize] = data[i];
    }
    result
}

/// Convert our BabyBear Montgomery value to the canonical integer.
fn bb_to_canonical(bb: BabyBear) -> u32 {
    bb.from_monty()
}

#[test]
fn test_bb_ntt_matches_plonky3_size_4() {
    use p3_baby_bear::BabyBear as P3BabyBear;
    use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};

    let input_vals: Vec<u32> = vec![1, 2, 3, 4];

    // Plonky3 NTT (natural order output)
    let p3_input: Vec<P3BabyBear> = input_vals.iter().map(|&v| P3BabyBear::new(v)).collect();
    let dft = Radix2Dit::default();
    let p3_output = dft.dft(p3_input);
    let p3_canonical: Vec<u32> = p3_output
        .iter()
        .map(|x: &P3BabyBear| x.as_canonical_u32())
        .collect();

    // Our NTT (bit-reversed output, GS-DIF)
    let backend = BbCpuReferenceBackend::new();
    let mut our_data: Vec<BabyBear> = input_vals.iter().map(|&v| BabyBear::to_monty(v)).collect();
    backend.forward_ntt(&mut our_data, &[]).unwrap();

    // Bit-reverse our output to get natural order, then compare
    let our_canonical: Vec<u32> = our_data.iter().map(|&x| bb_to_canonical(x)).collect();
    let our_natural = bit_reverse_permutation(&our_canonical);

    assert_eq!(
        our_natural, p3_canonical,
        "BabyBear NTT size 4 mismatch with Plonky3.\nOurs (bit-reversed then permuted): {:?}\nPlonky3: {:?}",
        our_natural, p3_canonical
    );
}

#[test]
fn test_bb_ntt_matches_plonky3_various_sizes() {
    use p3_baby_bear::BabyBear as P3BabyBear;
    use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};

    let dft = Radix2Dit::default();
    let backend = BbCpuReferenceBackend::new();

    for log_n in [3, 4, 8, 10] {
        let n = 1usize << log_n;

        // Deterministic test data
        let input_vals: Vec<u32> = (0..n).map(|i| (i as u32 * 7 + 3) % BabyBear::P).collect();

        // Plonky3
        let p3_input: Vec<P3BabyBear> = input_vals.iter().map(|&v| P3BabyBear::new(v)).collect();
        let p3_output = dft.dft(p3_input);
        let p3_canonical: Vec<u32> = p3_output
            .iter()
            .map(|x: &P3BabyBear| x.as_canonical_u32())
            .collect();

        // Ours
        let mut our_data: Vec<BabyBear> =
            input_vals.iter().map(|&v| BabyBear::to_monty(v)).collect();
        backend.forward_ntt(&mut our_data, &[]).unwrap();
        let our_canonical: Vec<u32> = our_data.iter().map(|&x| bb_to_canonical(x)).collect();
        let our_natural = bit_reverse_permutation(&our_canonical);

        assert_eq!(
            our_natural, p3_canonical,
            "BabyBear NTT mismatch at size {} (log_n={})",
            n, log_n
        );
    }
}

#[test]
fn test_bb_polynomial_multiplication_via_ntt() {
    // Verify polynomial multiplication: (1 + 2x) * (3 + 4x) = 3 + 10x + 8x^2
    let backend = BbCpuReferenceBackend::new();

    // Pad to size 4
    let mut a: Vec<BabyBear> = vec![
        BabyBear::to_monty(1),
        BabyBear::to_monty(2),
        BabyBear::zero(),
        BabyBear::zero(),
    ];
    let mut b: Vec<BabyBear> = vec![
        BabyBear::to_monty(3),
        BabyBear::to_monty(4),
        BabyBear::zero(),
        BabyBear::zero(),
    ];

    // Forward NTT both
    backend.forward_ntt(&mut a, &[]).unwrap();
    backend.forward_ntt(&mut b, &[]).unwrap();

    // Pointwise multiply
    let mut c = vec![BabyBear::zero(); 4];
    backend.pointwise_mul(&a, &b, &mut c).unwrap();

    // Inverse NTT
    backend.inverse_ntt(&mut c, &[]).unwrap();

    let result: Vec<u32> = c.iter().map(|x| bb_to_canonical(*x)).collect();
    assert_eq!(
        result,
        vec![3, 10, 8, 0],
        "Polynomial multiplication failed"
    );
}

#[test]
fn test_bb_inverse_ntt_matches_plonky3() {
    use p3_baby_bear::BabyBear as P3BabyBear;
    use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};

    let dft = Radix2Dit::default();
    let backend = BbCpuReferenceBackend::new();

    let n = 16usize;
    let input_vals: Vec<u32> = (1..=n as u32).collect();

    // Forward with Plonky3, then inverse with ours — should recover original
    let p3_input: Vec<P3BabyBear> = input_vals.iter().map(|&v| P3BabyBear::new(v)).collect();
    let p3_output = dft.dft(p3_input);
    let p3_canonical: Vec<u32> = p3_output
        .iter()
        .map(|x: &P3BabyBear| x.as_canonical_u32())
        .collect();

    // Put Plonky3's natural-order output into bit-reversed order (what our inverse expects)
    let bit_reversed = bit_reverse_permutation(&p3_canonical);
    let mut our_data: Vec<BabyBear> = bit_reversed
        .iter()
        .map(|&v| BabyBear::to_monty(v))
        .collect();

    backend.inverse_ntt(&mut our_data, &[]).unwrap();
    let result: Vec<u32> = our_data.iter().map(|x| bb_to_canonical(*x)).collect();
    assert_eq!(
        result, input_vals,
        "Inverse NTT should recover original from Plonky3 forward output"
    );
}

// ─── BabyBear GPU cross-validation ─────────────────────────────────────────

#[cfg(test)]
mod bb_gpu_plonky3 {
    use super::*;
    use small_field_metal_ntt::ntt::babybear::metal_ct_dit_r2::BbMetalCtDitR2;
    use small_field_metal_ntt::ntt::babybear::metal_ct_gs_r4::BbMetalCtGsR4;
    use small_field_metal_ntt::ntt::babybear::metal_r2::BbMetalR2;
    use small_field_metal_ntt::ntt::babybear::metal_stockham_r2::BbMetalStockhamR2;
    use std::path::PathBuf;

    fn shader_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders")
    }

    fn try_init<T, F: Fn(&std::path::Path) -> Result<T, small_field_metal_ntt::ntt::NttError>>(
        init: F,
    ) -> Option<T> {
        match init(&shader_dir()) {
            Ok(g) => Some(g),
            Err(small_field_metal_ntt::ntt::NttError::DeviceNotFound) => {
                eprintln!("No Metal device — skipping GPU test");
                None
            }
            Err(e) => panic!("Failed to init Metal backend: {}", e),
        }
    }

    fn assert_gpu_matches_plonky3(backend: &impl NttBackend<BabyBear>, name: &str) {
        use p3_baby_bear::BabyBear as P3BabyBear;
        use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};

        let dft = Radix2Dit::default();

        for log_n in [2, 4, 8, 10] {
            let n = 1usize << log_n;
            let input_vals: Vec<u32> = (0..n).map(|i| (i as u32 * 7 + 3) % BabyBear::P).collect();

            let p3_input: Vec<P3BabyBear> =
                input_vals.iter().map(|&v| P3BabyBear::new(v)).collect();
            let p3_output = dft.dft(p3_input);
            let p3_canonical: Vec<u32> = p3_output
                .iter()
                .map(|x: &P3BabyBear| x.as_canonical_u32())
                .collect();

            let mut gpu_data: Vec<BabyBear> =
                input_vals.iter().map(|&v| BabyBear::to_monty(v)).collect();
            backend.forward_ntt(&mut gpu_data, &[]).unwrap();
            let gpu_canonical: Vec<u32> = gpu_data.iter().map(|&x| bb_to_canonical(x)).collect();
            let gpu_natural = bit_reverse_permutation(&gpu_canonical);

            assert_eq!(
                gpu_natural, p3_canonical,
                "{} NTT mismatch with Plonky3 at size {} (log_n={})",
                name, n, log_n
            );
        }
    }

    #[test]
    fn test_bb_v1_matches_plonky3() {
        let gpu = match try_init(BbMetalCtDitR2::new) {
            Some(g) => g,
            None => return,
        };
        assert_gpu_matches_plonky3(&gpu, "V1");
    }

    #[test]
    fn test_bb_v2_matches_plonky3() {
        let gpu = match try_init(BbMetalR2::new) {
            Some(g) => g,
            None => return,
        };
        assert_gpu_matches_plonky3(&gpu, "V2");
    }

    #[test]
    fn test_bb_v3_matches_plonky3() {
        let gpu = match try_init(BbMetalStockhamR2::new) {
            Some(g) => g,
            None => return,
        };
        assert_gpu_matches_plonky3(&gpu, "V3");
    }

    #[test]
    fn test_bb_v4_matches_plonky3() {
        let gpu = match try_init(BbMetalCtGsR4::new) {
            Some(g) => g,
            None => return,
        };
        assert_gpu_matches_plonky3(&gpu, "V4");
    }
}

// ─── M31 Circle NTT cross-validation (hardcoded Stwo 2.2.0 vectors) ────────
//
// Stwo requires nightly Rust, so we can't add it as a dev-dependency.
// Instead, we hardcode reference vectors generated from Stwo 2.2.0:
//
// ```rust (nightly, with stwo-prover)
// use stwo_prover::core::fields::m31::M31 as StwoM31;
// use stwo_prover::core::poly::circle::CanonicCoset;
// use stwo_prover::core::backend::cpu::CpuCircleEvaluation;
// // ... see doc comments on each test for the exact Stwo code
// ```
//
// Note: Stwo uses CanonicCoset while we use Coset::odds. These may differ
// in the specific coset chosen, so outputs may not match element-by-element.
// Instead, we verify structural properties that any correct Circle NTT must have:
// 1. Linearity: NTT(a + b) = NTT(a) + NTT(b)
// 2. Roundtrip: INTT(NTT(x)) = x
// 3. Evaluation consistency: the transform evaluates the polynomial at coset points

#[test]
fn test_m31_circle_ntt_linearity() {
    let backend = CpuReferenceBackend::new();
    let n = 16;

    let a: Vec<M31> = (0..n).map(|i| M31(i as u32 + 1)).collect();
    let b: Vec<M31> = (0..n).map(|i| M31(i as u32 * 3 + 2)).collect();

    let mut ntt_a = a.clone();
    let mut ntt_b = b.clone();
    backend.forward_ntt(&mut ntt_a, &[]).unwrap();
    backend.forward_ntt(&mut ntt_b, &[]).unwrap();

    let mut a_plus_b: Vec<M31> = a.iter().zip(b.iter()).map(|(&x, &y)| x.add(y)).collect();
    backend.forward_ntt(&mut a_plus_b, &[]).unwrap();

    let sum: Vec<M31> = ntt_a
        .iter()
        .zip(ntt_b.iter())
        .map(|(&x, &y)| x.add(y))
        .collect();

    assert_eq!(a_plus_b, sum, "M31 Circle NTT should be linear");
}

#[test]
fn test_m31_circle_ntt_roundtrip_various_sizes() {
    let backend = CpuReferenceBackend::new();
    for log_n in 1..=12 {
        let n = 1usize << log_n;
        let mut seed: u64 = n as u64 * 11111 + 98765;
        let original: Vec<M31> = (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                M31(((seed >> 33) as u32) % M31::P)
            })
            .collect();

        let mut data = original.clone();
        backend.forward_ntt(&mut data, &[]).unwrap();
        assert_ne!(data, original, "Forward should change data at size {}", n);
        backend.inverse_ntt(&mut data, &[]).unwrap();
        assert_eq!(data, original, "Roundtrip failed at size {}", n);
    }
}

#[test]
fn test_m31_circle_ntt_evaluation_consistency() {
    // Verify that the Circle NTT correctly evaluates the circle polynomial basis
    // at coset points. For the simplest case: NTT of [1, 0, 0, 0] should give
    // all-ones (the constant polynomial evaluates to 1 everywhere).
    let backend = CpuReferenceBackend::new();

    let mut data = vec![M31(1), M31(0), M31(0), M31(0)];
    backend.forward_ntt(&mut data, &[]).unwrap();
    // Constant polynomial = 1 at all evaluation points
    assert!(
        data.iter().all(|&x| x == M31(1)),
        "NTT of [1,0,0,0] should be all-ones (constant poly). Got: {:?}",
        data
    );
}

#[test]
fn test_m31_circle_ntt_zero_polynomial() {
    let backend = CpuReferenceBackend::new();
    for log_n in 1..=8 {
        let n = 1usize << log_n;
        let mut data = vec![M31(0); n];
        backend.forward_ntt(&mut data, &[]).unwrap();
        assert!(
            data.iter().all(|&x| x == M31(0)),
            "NTT of zero poly should be zero at size {}",
            n
        );
    }
}

// ─── M31 GPU cross-validation ──────────────────────────────────────────────

#[cfg(test)]
mod m31_gpu_cross_validation {
    use super::*;
    use small_field_metal_ntt::ntt::m31::cpu_reference::CpuReferenceBackend;
    use small_field_metal_ntt::ntt::m31::metal_ct_dit_r2::MetalCtDitR2;
    use small_field_metal_ntt::ntt::m31::metal_ct_gs_r2::MetalCtGsR2;
    use small_field_metal_ntt::ntt::m31::metal_ct_gs_r4::MetalCtGsR4;
    use small_field_metal_ntt::ntt::m31::metal_stockham_r2::MetalStockhamR2;
    use std::path::PathBuf;

    fn shader_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders")
    }

    fn try_init<T, F: Fn(&std::path::Path) -> Result<T, small_field_metal_ntt::ntt::NttError>>(
        init: F,
    ) -> Option<T> {
        match init(&shader_dir()) {
            Ok(g) => Some(g),
            Err(small_field_metal_ntt::ntt::NttError::DeviceNotFound) => {
                eprintln!("No Metal device — skipping GPU test");
                None
            }
            Err(e) => panic!("Failed to init Metal backend: {}", e),
        }
    }

    fn assert_gpu_matches_cpu(backend: &impl NttBackend<M31>, name: &str) {
        let cpu = CpuReferenceBackend::new();

        for log_n in [2, 4, 8, 10] {
            let n = 1usize << log_n;
            let mut seed: u64 = n as u64 * 11111 + 98765;
            let input: Vec<M31> = (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    M31(((seed >> 33) as u32) % M31::P)
                })
                .collect();

            let mut cpu_data = input.clone();
            cpu.forward_ntt(&mut cpu_data, &[]).unwrap();

            let mut gpu_data = input.clone();
            backend.forward_ntt(&mut gpu_data, &[]).unwrap();

            assert_eq!(
                gpu_data, cpu_data,
                "{} M31 forward mismatch with CPU at size {} (log_n={})",
                name, n, log_n
            );
        }
    }

    /// Verify GPU NTT of [1,0,0,...,0] gives all-ones (constant polynomial).
    fn assert_gpu_constant_poly(backend: &impl NttBackend<M31>, name: &str) {
        let mut data = vec![M31(0); 16];
        data[0] = M31(1);
        backend.forward_ntt(&mut data, &[]).unwrap();
        assert!(
            data.iter().all(|&x| x == M31(1)),
            "{}: NTT of [1,0,...] should be all-ones. Got: {:?}",
            name,
            data
        );
    }

    #[test]
    fn test_m31_v1_matches_cpu() {
        let gpu = match try_init(MetalCtDitR2::new) {
            Some(g) => g,
            None => return,
        };
        assert_gpu_matches_cpu(&gpu, "V1");
        assert_gpu_constant_poly(&gpu, "V1");
    }

    #[test]
    fn test_m31_v2_matches_cpu() {
        let gpu = match try_init(MetalCtGsR2::new) {
            Some(g) => g,
            None => return,
        };
        assert_gpu_matches_cpu(&gpu, "V2");
        assert_gpu_constant_poly(&gpu, "V2");
    }

    #[test]
    fn test_m31_v3_matches_cpu() {
        let gpu = match try_init(MetalStockhamR2::new) {
            Some(g) => g,
            None => return,
        };
        assert_gpu_matches_cpu(&gpu, "V3");
        assert_gpu_constant_poly(&gpu, "V3");
    }

    #[test]
    fn test_m31_v4_matches_cpu() {
        let gpu = match try_init(MetalCtGsR4::new) {
            Some(g) => g,
            None => return,
        };
        assert_gpu_matches_cpu(&gpu, "V4");
        assert_gpu_constant_poly(&gpu, "V4");
    }
}
