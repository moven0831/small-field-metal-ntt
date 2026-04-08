//! Generic twiddle factor cache for NTT backends.
//!
//! Caches forward and inverse twiddles for the most recently used `log_n`.
//! The generation functions are field-specific (circle group for M31,
//! roots of unity for BabyBear) but the caching logic is identical.
//!
//! Not `Send`/`Sync` — single-threaded only (uses `RefCell` + `Rc`).

use crate::field::Field;
use std::cell::RefCell;
use std::rc::Rc;

/// Cached twiddle factor storage, generic over the field type.
///
/// Uses function pointers for generation to accommodate different
/// mathematical structures (circle group vs multiplicative group).
pub struct TwiddleCache<F: Field> {
    inner: RefCell<Option<CachedEntry<F>>>,
    gen_forward: fn(u32) -> Vec<Vec<F>>,
    gen_inverse: fn(u32) -> Vec<Vec<F>>,
}

struct CachedEntry<F: Field> {
    log_n: u32,
    forward: Rc<Vec<Vec<F>>>,
    inverse: Rc<Vec<Vec<F>>>,
}

impl<F: Field> TwiddleCache<F> {
    /// Create a new cache with the given generation functions.
    pub fn new(
        gen_forward: fn(u32) -> Vec<Vec<F>>,
        gen_inverse: fn(u32) -> Vec<Vec<F>>,
    ) -> Self {
        Self {
            inner: RefCell::new(None),
            gen_forward,
            gen_inverse,
        }
    }

    /// Get forward twiddles for the given log_n, computing and caching if needed.
    pub fn forward(&self, log_n: u32) -> Rc<Vec<Vec<F>>> {
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
    pub fn inverse(&self, log_n: u32) -> Rc<Vec<Vec<F>>> {
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
            .is_none_or(|e| e.log_n != log_n);

        if needs_recompute {
            let forward = Rc::new((self.gen_forward)(log_n));
            let inverse = Rc::new((self.gen_inverse)(log_n));
            *self.inner.borrow_mut() = Some(CachedEntry {
                log_n,
                forward,
                inverse,
            });
        }
    }
}
