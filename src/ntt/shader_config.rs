//! Shader configuration traits for generic Metal NTT backends.
//!
//! Maps field types to concrete Metal shader function names at compile time.
//! Each field (M31, BabyBear) has a different set of shader kernels but
//! identical dispatch logic. The traits capture this mapping.

use crate::field::Field;
use crate::ntt::twiddle_cache::TwiddleCache;

/// Configuration for radix-2 in-place NTT (CT-DIT forward / GS-DIF inverse).
///
/// Maps a field type to the shader function names for single-column NTT.
pub trait R2ShaderConfig: 'static {
    type F: Field;

    /// Algorithm variant name for reporting (e.g., "metal-ct-gs-r2").
    const NAME: &'static str;

    /// Shader function names for single-column pipelines.
    const FORWARD_TG: &'static str;
    const INVERSE_TG: &'static str;
    const FORWARD_DEVICE: &'static str;
    const INVERSE_DEVICE: &'static str;
    const NORMALIZE: &'static str;

    /// Create a twiddle cache appropriate for this field.
    fn make_twiddle_cache() -> TwiddleCache<Self::F>;
}

/// Extension for backends that support batched NTT (e.g., BabyBear for Plonky3 LDE).
///
/// M31 does not implement this because Circle NTT batch shaders don't exist yet.
pub trait BatchR2ShaderConfig: R2ShaderConfig {
    const BATCH_FORWARD_TG: &'static str;
    const BATCH_INVERSE_TG: &'static str;
    const BATCH_FORWARD_DEVICE: &'static str;
    const BATCH_INVERSE_DEVICE: &'static str;
    const BATCH_NORMALIZE: &'static str;
}

// ── M31 Configuration ──────────────────────────────────────────────────

pub struct M31R2Config;

impl R2ShaderConfig for M31R2Config {
    type F = crate::field::m31::M31;
    const NAME: &'static str = "metal-ct-gs-r2";
    const FORWARD_TG: &'static str = "ct_gs_r2_forward_tg";
    const INVERSE_TG: &'static str = "ct_gs_r2_inverse_tg";
    const FORWARD_DEVICE: &'static str = "ct_gs_r2_butterfly_device";
    const INVERSE_DEVICE: &'static str = "ct_gs_r2_butterfly_device_inv";
    const NORMALIZE: &'static str = "ct_gs_r2_normalize";

    fn make_twiddle_cache() -> TwiddleCache<Self::F> {
        crate::ntt::m31::twiddles::new_m31_twiddle_cache()
    }
}

// ── BabyBear Configuration ─────────────────────────────────────────────

pub struct BbR2Config;

impl R2ShaderConfig for BbR2Config {
    type F = crate::field::babybear::BabyBear;
    const NAME: &'static str = "metal-bb-r2";
    const FORWARD_TG: &'static str = "bb_r2_forward_tg";
    const INVERSE_TG: &'static str = "bb_r2_inverse_tg";
    const FORWARD_DEVICE: &'static str = "bb_r2_butterfly_device";
    const INVERSE_DEVICE: &'static str = "bb_r2_butterfly_device_inv";
    const NORMALIZE: &'static str = "bb_r2_normalize";

    fn make_twiddle_cache() -> TwiddleCache<Self::F> {
        crate::ntt::babybear::twiddles::new_bb_twiddle_cache()
    }
}

impl BatchR2ShaderConfig for BbR2Config {
    const BATCH_FORWARD_TG: &'static str = "bb_r2_batch_forward_tg";
    const BATCH_INVERSE_TG: &'static str = "bb_r2_batch_inverse_tg";
    const BATCH_FORWARD_DEVICE: &'static str = "bb_r2_batch_butterfly_device";
    const BATCH_INVERSE_DEVICE: &'static str = "bb_r2_batch_butterfly_device_inv";
    const BATCH_NORMALIZE: &'static str = "bb_r2_batch_normalize";
}
