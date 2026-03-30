pub mod m31;
pub mod circle;

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

    /// Modular reduction from u64 intermediate.
    fn reduce(val: u64) -> Self;

    /// Modular inverse via Fermat's little theorem: a^(p-2) mod p
    fn inv(self) -> Self;
}
