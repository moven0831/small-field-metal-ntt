//! BabyBear twiddle factor generation and caching for standard NTT.
//!
//! Unlike the M31 Circle NTT twiddles (which use coset x/y coordinates),
//! BabyBear twiddles are powers of roots of unity in the multiplicative group.
//! BabyBear has two-adicity 27, supporting standard radix-2 NTT up to 2^27.

use crate::field::babybear::BabyBear;
use crate::field::Field;
use std::cell::RefCell;
use std::rc::Rc;

/// Generate inverse twiddle factors for BabyBear NTT of size 2^log_n.
///
/// Same structure as `BabyBear::generate_twiddles()` but each element is inverted.
pub fn generate_inverse_twiddles(log_n: u32) -> Vec<Vec<BabyBear>> {
    let forward = BabyBear::generate_twiddles(log_n);
    forward
        .into_iter()
        .map(|layer| layer.into_iter().map(|w| w.inv()).collect())
        .collect()
}

/// Cached twiddle factor storage for BabyBear NTT.
///
/// Caches forward and inverse twiddles for the most recently used `log_n`.
/// Not `Send`/`Sync` — single-threaded only (uses `RefCell` + `Rc`).
pub struct BbTwiddleCache {
    inner: RefCell<Option<CachedEntry>>,
}

struct CachedEntry {
    log_n: u32,
    forward: Rc<Vec<Vec<BabyBear>>>,
    inverse: Rc<Vec<Vec<BabyBear>>>,
}

impl BbTwiddleCache {
    pub fn new() -> Self {
        Self {
            inner: RefCell::new(None),
        }
    }

    pub fn forward(&self, log_n: u32) -> Rc<Vec<Vec<BabyBear>>> {
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

    pub fn inverse(&self, log_n: u32) -> Rc<Vec<Vec<BabyBear>>> {
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
            let forward = Rc::new(BabyBear::generate_twiddles(log_n));
            let inverse = Rc::new(generate_inverse_twiddles(log_n));
            *self.inner.borrow_mut() = Some(CachedEntry {
                log_n,
                forward,
                inverse,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inverse_twiddles_are_inverses() {
        let log_n = 4u32;
        let fwd = BabyBear::generate_twiddles(log_n);
        let inv = generate_inverse_twiddles(log_n);

        for (layer_idx, (f_layer, i_layer)) in fwd.iter().zip(inv.iter()).enumerate() {
            for (j, (&f, &i)) in f_layer.iter().zip(i_layer.iter()).enumerate() {
                let product = f.mul(i);
                assert_eq!(
                    product,
                    BabyBear::one(),
                    "layer {} index {}: forward * inverse should be 1",
                    layer_idx,
                    j
                );
            }
        }
    }

    #[test]
    fn test_cache_returns_consistent_results() {
        let cache = BbTwiddleCache::new();
        let a = cache.forward(10);
        let b = cache.forward(10);
        assert_eq!(a.len(), b.len());
        for (la, lb) in a.iter().zip(b.iter()) {
            assert_eq!(la, lb);
        }
    }

    #[test]
    fn test_cache_invalidates_on_size_change() {
        let cache = BbTwiddleCache::new();
        let a = cache.forward(8);
        let b = cache.forward(10);
        assert_eq!(a.len(), 8);
        assert_eq!(b.len(), 10);
    }
}
