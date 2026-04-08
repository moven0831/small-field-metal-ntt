use super::Field;

/// Mersenne-31 field element: p = 2^31 - 1
///
/// Reduction is exceptionally fast: reduce(a) = (a >> 31) + (a & 0x7FFF_FFFF)
/// with a conditional subtract if result >= p.
///
/// Used by Stwo (Circle STARKs). M31 has 2-adicity of only 1, so standard NTT
/// is impossible. Instead we use Circle NTT (DCCT) which exploits the circle
/// group C(F_p) = {(x,y): x^2+y^2=1} with order p+1 = 2^31.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct M31(pub u32);

impl M31 {
    pub const P: u32 = (1u32 << 31) - 1;
}

impl Field for M31 {
    const MODULUS: u32 = Self::P;

    #[inline(always)]
    fn zero() -> Self {
        M31(0)
    }

    #[inline(always)]
    fn one() -> Self {
        M31(1)
    }

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        let sum = self.0 as u64 + other.0 as u64;
        Self::reduce(sum)
    }

    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        let val = if self.0 >= other.0 {
            self.0 - other.0
        } else {
            Self::P - (other.0 - self.0)
        };
        M31(val)
    }

    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        let product = self.0 as u64 * other.0 as u64;
        Self::reduce(product)
    }

    #[inline(always)]
    fn reduce(val: u64) -> Self {
        // Mersenne reduction: val mod (2^31 - 1)
        // = (val >> 31) + (val & 0x7FFF_FFFF)
        // May need one more reduction if result >= p.
        let lo = (val & 0x7FFF_FFFF) as u32;
        let hi = (val >> 31) as u32;
        let sum = lo + hi;
        // If sum >= P, subtract P. Since sum < 2*P, one subtract suffices.
        if sum >= Self::P {
            M31(sum - Self::P)
        } else {
            M31(sum)
        }
    }

    fn inv(self) -> Self {
        // Fermat's little theorem: a^(p-2) mod p
        // p-2 = 2^31 - 3
        let mut result = Self::one();
        let mut base = self;
        let mut exp = Self::P - 2;
        while exp > 0 {
            if exp & 1 == 1 {
                result = result.mul(base);
            }
            base = base.mul(base);
            exp >>= 1;
        }
        result
    }

    #[inline(always)]
    fn raw(self) -> u32 {
        self.0
    }

    #[inline(always)]
    fn from_raw(val: u32) -> Self {
        M31(val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_basic() {
        assert_eq!(M31(5).add(M31(3)), M31(8));
    }

    #[test]
    fn test_add_wrap() {
        let p_minus_1 = M31(M31::P - 1);
        assert_eq!(p_minus_1.add(M31(1)), M31(0));
        assert_eq!(p_minus_1.add(p_minus_1), M31(M31::P - 2));
    }

    #[test]
    fn test_sub_basic() {
        assert_eq!(M31(8).sub(M31(3)), M31(5));
    }

    #[test]
    fn test_sub_underflow() {
        assert_eq!(M31(0).sub(M31(1)), M31(M31::P - 1));
    }

    #[test]
    fn test_mul_basic() {
        assert_eq!(M31(3).mul(M31(7)), M31(21));
    }

    #[test]
    fn test_mul_identity() {
        assert_eq!(M31(42).mul(M31(1)), M31(42));
        assert_eq!(M31(42).mul(M31(0)), M31(0));
    }

    #[test]
    fn test_mul_large() {
        // (p-1) * (p-1) = (p-1)^2 mod p = 1
        let p_minus_1 = M31(M31::P - 1);
        assert_eq!(p_minus_1.mul(p_minus_1), M31(1));
    }

    #[test]
    fn test_reduce_zero() {
        assert_eq!(M31::reduce(0), M31(0));
    }

    #[test]
    fn test_reduce_p() {
        assert_eq!(M31::reduce(M31::P as u64), M31(0));
    }

    #[test]
    fn test_inv() {
        let a = M31(42);
        let a_inv = a.inv();
        assert_eq!(a.mul(a_inv), M31(1));
    }
}
