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
use std::cell::RefCell;
use std::rc::Rc;

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

/// Cached twiddle factor storage to avoid recomputing per NTT call.
///
/// Caches forward and inverse twiddles for the most recently used `log_n`.
/// Uses `Rc` to avoid cloning the full twiddle vectors on access.
/// Not `Send`/`Sync` — single-threaded only (uses `RefCell` + `Rc`).
pub struct TwiddleCache {
    inner: RefCell<Option<CachedEntry>>,
}

struct CachedEntry {
    log_n: u32,
    forward: Rc<Vec<Vec<M31>>>,
    inverse: Rc<Vec<Vec<M31>>>,
}

impl TwiddleCache {
    pub fn new() -> Self {
        Self {
            inner: RefCell::new(None),
        }
    }

    /// Get forward twiddles for the given log_n, computing and caching if needed.
    /// Returns an `Rc` — cheap reference count bump, no data copy.
    pub fn forward(&self, log_n: u32) -> Rc<Vec<Vec<M31>>> {
        self.ensure_cached(log_n);
        Rc::clone(
            &self
                .inner
                .borrow()
                .as_ref()
                .expect("ensure_cached must populate cache")
                .forward,
        )
    }

    /// Get inverse twiddles for the given log_n, computing and caching if needed.
    /// Returns an `Rc` — cheap reference count bump, no data copy.
    pub fn inverse(&self, log_n: u32) -> Rc<Vec<Vec<M31>>> {
        self.ensure_cached(log_n);
        Rc::clone(
            &self
                .inner
                .borrow()
                .as_ref()
                .expect("ensure_cached must populate cache")
                .inverse,
        )
    }

    fn ensure_cached(&self, log_n: u32) {
        let needs_recompute = self
            .inner
            .borrow()
            .as_ref()
            .map_or(true, |e| e.log_n != log_n);

        if needs_recompute {
            let coset = Coset::odds(log_n);
            let forward = Rc::new(generate_twiddles(&coset));
            let inverse = Rc::new(generate_itwiddles(&coset));
            *self.inner.borrow_mut() = Some(CachedEntry {
                log_n,
                forward,
                inverse,
            });
        }
    }
}
