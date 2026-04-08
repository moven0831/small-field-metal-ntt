//! Shared twiddle factor generation for Circle NTT.
//!
//! The Circle FFT has two types of layers:
//! - Line layers (0..log_n-1): use x-coordinates of coset points
//! - Circle layer (last, index log_n-1): use y-coordinates
//!
//! The coset doubles after each layer. At the deepest layer, the doubled
//! coset has x=0 (since 2*(2^15)^2 - 1 = 0 mod p for M31), so we must
//! use y-coordinates instead.

use crate::field::circle::Coset;
use crate::field::m31::M31;
use crate::field::Field;

/// Generate forward twiddle factors for each layer.
///
/// Returns a `Vec<Vec<M31>>` where `result[layer]` contains the twiddle
/// factors for that layer. Layer 0 has n/2 twiddles, layer 1 has n/4, etc.
pub fn generate_twiddles(coset: &Coset) -> Vec<Vec<M31>> {
    let mut result = Vec::new();
    let mut current = coset.clone();
    let log_n = coset.log_size;

    for layer in 0..log_n {
        let half_size = current.size() / 2;
        let is_last_layer = layer == log_n - 1;

        let layer_twiddles: Vec<M31> = (0..half_size)
            .map(|i| {
                let p = current.at(bit_reverse_idx(i, current.log_size - 1));
                if is_last_layer {
                    p.y
                } else {
                    p.x
                }
            })
            .collect();
        result.push(layer_twiddles);
        current = current.double();
    }

    result
}

/// Generate inverse twiddle factors.
/// Same structure as forward but with inverted twiddle values.
pub fn generate_itwiddles(coset: &Coset) -> Vec<Vec<M31>> {
    let mut result = Vec::new();
    let mut current = coset.clone();
    let log_n = coset.log_size;

    for layer in 0..log_n {
        let half_size = current.size() / 2;
        let is_last_layer = layer == log_n - 1;

        let layer_twiddles: Vec<M31> = (0..half_size)
            .map(|i| {
                let p = current.at(bit_reverse_idx(i, current.log_size - 1));
                let t = if is_last_layer { p.y } else { p.x };
                t.inv()
            })
            .collect();
        result.push(layer_twiddles);
        current = current.double();
    }

    result
}

/// Reverse the bits of `index` within a `log_size`-bit field.
pub fn bit_reverse_idx(index: usize, log_size: u32) -> usize {
    let mut val = index as u32;
    let mut result = 0u32;
    for _ in 0..log_size {
        result = (result << 1) | (val & 1);
        val >>= 1;
    }
    result as usize
}

/// M31 twiddle cache using the generic `TwiddleCache<M31>`.
///
/// Wraps circle-group coset construction inside the generation functions.
pub type TwiddleCache = super::twiddle_cache::TwiddleCache<M31>;

/// Create a new M31 twiddle cache with circle-group generators.
pub fn new_m31_twiddle_cache() -> TwiddleCache {
    super::twiddle_cache::TwiddleCache::new(m31_gen_forward, m31_gen_inverse)
}

fn m31_gen_forward(log_n: u32) -> Vec<Vec<M31>> {
    generate_twiddles(&Coset::odds(log_n))
}

fn m31_gen_inverse(log_n: u32) -> Vec<Vec<M31>> {
    generate_itwiddles(&Coset::odds(log_n))
}
