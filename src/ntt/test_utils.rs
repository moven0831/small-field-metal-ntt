//! Shared test utilities for NTT backend variants.
//!
//! Provides parameterized correctness tests that work with any `NttBackend<M31>`.
//! Each GPU variant's test module delegates to these shared functions, keeping
//! only variant-specific tests (e.g., V4's odd-log radix-2 fallback) local.

use crate::field::m31::M31;
use crate::ntt::cpu_reference::CpuReferenceBackend;
use crate::ntt::{NttBackend, NttError};
use std::path::PathBuf;

/// Path to the shaders directory.
pub fn shader_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders")
}

/// Try to initialize a Metal backend, skipping the test if no GPU is available.
///
/// Usage:
/// ```ignore
/// let gpu = match try_init_metal(|p| MetalCtGsR2::new(p)) {
///     Some(g) => g,
///     None => return,
/// };
/// ```
pub fn try_init_metal<T, F: Fn(&std::path::Path) -> Result<T, NttError>>(init: F) -> Option<T> {
    match init(&shader_dir()) {
        Ok(g) => Some(g),
        Err(NttError::DeviceNotFound) => {
            eprintln!("No Metal device — skipping");
            None
        }
        Err(e) => panic!("Failed to init Metal backend: {}", e),
    }
}

/// Generate deterministic pseudo-random M31 test data using a simple LCG.
pub fn lcg_data(n: usize, seed: u64) -> Vec<M31> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            M31(((s >> 33) as u32) % M31::P)
        })
        .collect()
}

/// Generate deterministic data for a given size, choosing the generator
/// based on size to match existing tests.
fn test_data(n: usize) -> Vec<M31> {
    match n {
        4 => vec![M31(1), M31(2), M31(3), M31(4)],
        16 => (0..n).map(|i| M31(i as u32 * 7 + 3)).collect(),
        256 => (0..n).map(|i| M31((i as u32 * 13 + 7) % M31::P)).collect(),
        1024 => (0..n).map(|i| M31((i as u32 * 17 + 11) % M31::P)).collect(),
        _ => lcg_data(n, n as u64 * 11111 + 98765),
    }
}

/// Assert that the backend's forward NTT matches the CPU reference at the given sizes.
pub fn assert_forward_matches_cpu(backend: &impl NttBackend<M31>, sizes: &[usize]) {
    let cpu = CpuReferenceBackend::new();
    for &n in sizes {
        let original = test_data(n);
        let mut cpu_data = original.clone();
        cpu.forward_ntt(&mut cpu_data, &[]).unwrap();
        let mut gpu_data = original.clone();
        backend.forward_ntt(&mut gpu_data, &[]).unwrap();
        assert_eq!(
            gpu_data, cpu_data,
            "{}: Forward mismatch at size {}",
            backend.name(),
            n,
        );
    }
}

/// Assert that forward + inverse NTT is the identity at the given sizes.
pub fn assert_roundtrip(backend: &impl NttBackend<M31>, sizes: &[usize]) {
    for &n in sizes {
        let original = test_data(n);
        let mut data = original.clone();
        backend.forward_ntt(&mut data, &[]).unwrap();
        assert_ne!(
            data, original,
            "{}: Forward should change data at size {}",
            backend.name(),
            n,
        );
        backend.inverse_ntt(&mut data, &[]).unwrap();
        assert_eq!(
            data, original,
            "{}: Round-trip failed at size {}",
            backend.name(),
            n,
        );
    }
}

/// Assert basic edge cases: zeros, size 1, invalid sizes.
pub fn assert_edge_cases(backend: &impl NttBackend<M31>) {
    // All zeros
    let mut data = vec![M31(0); 64];
    backend.forward_ntt(&mut data, &[]).unwrap();
    assert!(data.iter().all(|&x| x == M31(0)), "NTT of zeros should be zeros");

    // Size 1 identity
    let mut data = vec![M31(42)];
    backend.forward_ntt(&mut data, &[]).unwrap();
    assert_eq!(data[0], M31(42), "Size 1 should be identity");

    // Invalid size
    let mut data = vec![M31(1); 3];
    assert!(backend.forward_ntt(&mut data, &[]).is_err(), "Size 3 should fail");

    // Empty
    let mut data: Vec<M31> = vec![];
    assert!(backend.forward_ntt(&mut data, &[]).is_err(), "Empty should fail");
}

/// Assert edge cases for inverse NTT.
pub fn assert_inverse_edge_cases(backend: &impl NttBackend<M31>) {
    // Size 1 identity
    let mut data = vec![M31(42)];
    backend.inverse_ntt(&mut data, &[]).unwrap();
    assert_eq!(data[0], M31(42), "Inverse size 1 should be identity");

    // Invalid size
    let mut data = vec![M31(1); 3];
    assert!(backend.inverse_ntt(&mut data, &[]).is_err(), "Inverse size 3 should fail");

    // Empty
    let mut data: Vec<M31> = vec![];
    assert!(backend.inverse_ntt(&mut data, &[]).is_err(), "Inverse empty should fail");
}

/// Assert pointwise multiplication works correctly.
pub fn assert_pointwise_mul(backend: &impl NttBackend<M31>) {
    let a = vec![M31(2), M31(3), M31(4), M31(5)];
    let b = vec![M31(10), M31(20), M31(30), M31(40)];
    let mut out = vec![M31(0); 4];
    backend.pointwise_mul(&a, &b, &mut out).unwrap();
    assert_eq!(out, vec![M31(20), M31(60), M31(120), M31(200)]);

    // Size mismatch
    let a = vec![M31(1); 4];
    let b = vec![M31(1); 3];
    let mut out = vec![M31(0); 4];
    assert!(backend.pointwise_mul(&a, &b, &mut out).is_err());
}
