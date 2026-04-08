pub mod babybear;
pub mod circle;
pub mod m31;

/// Trait for finite field arithmetic used by NTT backends.
/// Implementations must be consistent between Rust (CPU) and Metal (GPU) versions.
pub trait Field: Copy + Clone + PartialEq + Eq + std::fmt::Debug {
    /// The prime modulus of the field.
    const MODULUS: u32;

    /// Additive identity.
    fn zero() -> Self;

    /// Multiplicative identity.
    fn one() -> Self;

    /// Field addition: (a + b) mod p
    fn add(self, other: Self) -> Self;

    /// Field subtraction: (a - b) mod p
    fn sub(self, other: Self) -> Self;

    /// Field multiplication: (a * b) mod p
    fn mul(self, other: Self) -> Self;

    /// Reduce a u64 value into the field's internal representation.
    ///
    /// For standard fields (M31), internal form = standard value.
    /// For Montgomery fields (BabyBear), internal form = Montgomery encoding.
    /// Callers should use field methods (add, mul, etc.) for arithmetic
    /// and field-specific conversion (e.g., from_monty) for output.
    fn reduce(val: u64) -> Self;

    /// Modular inverse via Fermat's little theorem: a^(p-2) mod p
    fn inv(self) -> Self;

    /// Extract the raw internal u32 representation for GPU buffer serialization.
    ///
    /// For M31, this is the standard value. For BabyBear, this is the Montgomery-encoded value.
    /// Use `from_raw` to reconstruct from GPU buffer readback.
    fn raw(self) -> u32;

    /// Construct a field element from its raw internal u32 representation.
    ///
    /// The value must already be in the field's internal form (Montgomery for BabyBear,
    /// standard for M31). This is the inverse of `raw()`: `F::from_raw(f.raw()) == f`.
    fn from_raw(val: u32) -> Self;
}
