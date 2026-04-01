use super::Field;

/// BabyBear field element: p = 2^31 - 2^27 + 1 = 2013265921
///
/// Values are stored in Montgomery form: the internal u32 represents `a * R mod p`
/// where R = 2^32. This makes multiplication fast via Montgomery reduction, at the
/// cost of converting to/from Montgomery form on field entry/exit.
///
/// Used by Plonky3. BabyBear has 2-adicity of 27, meaning the multiplicative group
/// has a subgroup of order 2^27, making it ideal for standard radix-2 NTT.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct BabyBear(pub u32);

/// The prime modulus: p = 2^31 - 2^27 + 1 = 2013265921
const P: u32 = 2013265921; // 0x78000001

/// R = 2^32 mod p — this is the Montgomery form of 1.
const R_MOD_P: u32 = 268435454; // 0x0FFFFFFE

/// R^2 mod p — used to convert a standard integer into Montgomery form.
const R2_MOD_P: u32 = 1172168163; // 0x45DDDDE3

/// Montgomery constant: -p^{-1} mod 2^32.
/// Used in Montgomery reduction: m = (T_lo * MONTY_CONST) mod R.
const MONTY_CONST: u32 = 2013265919; // 0x77FFFFFF

/// Two-adicity: the largest k such that 2^k divides p-1.
/// p - 1 = 2013265920 = 2^27 * 15.
pub const TWO_ADICITY: u32 = 27;

/// Primitive 2^27-th root of unity (in standard form: 440564289).
/// Stored here in Montgomery form for direct use in field arithmetic.
const TWO_ADIC_ROOT_MONTY: u32 = 1476048622; // 0x57FAB6EE

impl BabyBear {
    pub const P: u32 = P;

    /// Montgomery reduction of a u64 value.
    /// Given T (a u64), computes T * R^{-1} mod p.
    #[inline(always)]
    fn monty_reduce(t: u64) -> u32 {
        let t_lo = t as u32;
        let m = t_lo.wrapping_mul(MONTY_CONST);
        let mp = m as u64 * P as u64;
        let result = ((t + mp) >> 32) as u32;
        if result >= P {
            result - P
        } else {
            result
        }
    }

    /// Convert a standard integer to Montgomery form.
    pub fn to_monty(val: u32) -> BabyBear {
        // a_monty = a * R mod p = monty_reduce(a * R^2 mod p ... wait, just monty_reduce(a * R^2))
        // monty_reduce(a * R^2) = a * R^2 * R^{-1} = a * R mod p
        let t = val as u64 * R2_MOD_P as u64;
        BabyBear(Self::monty_reduce(t))
    }

    /// Convert from Montgomery form back to a standard integer.
    pub fn from_monty(self) -> u32 {
        // monty_reduce(a_monty * 1) = a_monty * R^{-1} = a * R * R^{-1} = a mod p
        Self::monty_reduce(self.0 as u64)
    }

    /// Generator of the 2^log_n multiplicative subgroup.
    /// Obtained by squaring the primitive 2^27-th root of unity (27 - log_n) times.
    pub fn two_adic_generator(log_n: u32) -> BabyBear {
        assert!(log_n <= TWO_ADICITY, "log_n exceeds two-adicity");
        let mut g = BabyBear(TWO_ADIC_ROOT_MONTY);
        for _ in 0..(TWO_ADICITY - log_n) {
            g = g.mul(g);
        }
        g
    }

    /// Generate twiddle factors for an NTT of size 2^log_n.
    ///
    /// Returns a vector of layers, where layer `s` (0-indexed) contains the
    /// twiddle factors for the s-th butterfly stage. Layer s has 2^s entries,
    /// each being a power of the 2^(s+1)-th root of unity.
    pub fn generate_twiddles(log_n: u32) -> Vec<Vec<BabyBear>> {
        let mut twiddles = Vec::with_capacity(log_n as usize);
        for s in 0..log_n {
            let half_m = 1usize << s;
            let g = Self::two_adic_generator(s + 1);
            let mut layer = Vec::with_capacity(half_m);
            let mut w = BabyBear::one();
            for _ in 0..half_m {
                layer.push(w);
                w = w.mul(g);
            }
            twiddles.push(layer);
        }
        twiddles
    }
}

impl Field for BabyBear {
    const MODULUS: u32 = P;

    #[inline(always)]
    fn zero() -> Self {
        BabyBear(0)
    }

    #[inline(always)]
    fn one() -> Self {
        BabyBear(R_MOD_P)
    }

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        let sum = self.0 + other.0;
        // Since both operands are < P, sum < 2P, so one conditional subtract suffices.
        if sum >= P {
            BabyBear(sum - P)
        } else {
            BabyBear(sum)
        }
    }

    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        if self.0 >= other.0 {
            BabyBear(self.0 - other.0)
        } else {
            BabyBear(P - (other.0 - self.0))
        }
    }

    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        let product = self.0 as u64 * other.0 as u64;
        BabyBear(Self::monty_reduce(product))
    }

    #[inline(always)]
    fn reduce(val: u64) -> Self {
        // Reduce a u64 into Montgomery form.
        // First reduce to [0, p), then convert to Montgomery form.
        let reduced = (val % P as u64) as u32;
        Self::to_monty(reduced)
    }

    fn inv(self) -> Self {
        // Fermat's little theorem: a^{-1} = a^{p-2} mod p
        // Since values are in Montgomery form, repeated Montgomery multiplication works directly.
        let mut result = Self::one();
        let mut base = self;
        let mut exp = P - 2;
        while exp > 0 {
            if exp & 1 == 1 {
                result = result.mul(base);
            }
            base = base.mul(base);
            exp >>= 1;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_basic() {
        let a = BabyBear::to_monty(3);
        let b = BabyBear::to_monty(7);
        let c = a.add(b);
        assert_eq!(c.from_monty(), 10);
    }

    #[test]
    fn test_add_wrap() {
        let a = BabyBear::to_monty(P - 1);
        let b = BabyBear::to_monty(1);
        let c = a.add(b);
        assert_eq!(c.from_monty(), 0);
    }

    #[test]
    fn test_sub_basic() {
        let a = BabyBear::to_monty(10);
        let b = BabyBear::to_monty(3);
        let c = a.sub(b);
        assert_eq!(c.from_monty(), 7);
    }

    #[test]
    fn test_sub_underflow() {
        let a = BabyBear::to_monty(0);
        let b = BabyBear::to_monty(1);
        let c = a.sub(b);
        assert_eq!(c.from_monty(), P - 1);
    }

    #[test]
    fn test_mul_basic() {
        let a = BabyBear::to_monty(3);
        let b = BabyBear::to_monty(7);
        let c = a.mul(b);
        assert_eq!(c.from_monty(), 21);
    }

    #[test]
    fn test_mul_identity() {
        let a = BabyBear::to_monty(42);
        let one = BabyBear::one();
        assert_eq!(a.mul(one).from_monty(), 42);
    }

    #[test]
    fn test_mul_zero() {
        let a = BabyBear::to_monty(42);
        let zero = BabyBear::zero();
        assert_eq!(a.mul(zero).from_monty(), 0);
    }

    #[test]
    fn test_mul_large() {
        // (p-1)^2 mod p = 1
        let a = BabyBear::to_monty(P - 1);
        assert_eq!(a.mul(a).from_monty(), 1);
    }

    #[test]
    fn test_one_monty() {
        // Montgomery form of 1 is R_MOD_P
        let one = BabyBear::one();
        assert_eq!(one.0, R_MOD_P);
        assert_eq!(one.from_monty(), 1);
    }

    #[test]
    fn test_zero_monty() {
        let zero = BabyBear::zero();
        assert_eq!(zero.0, 0);
        assert_eq!(zero.from_monty(), 0);
    }

    #[test]
    fn test_monty_roundtrip() {
        for &val in &[0u32, 1, 2, 42, 1000, P - 1, P - 2] {
            let m = BabyBear::to_monty(val);
            assert_eq!(m.from_monty(), val, "roundtrip failed for {}", val);
        }
    }

    #[test]
    fn test_inv() {
        let a = BabyBear::to_monty(42);
        let a_inv = a.inv();
        let product = a.mul(a_inv);
        assert_eq!(product.from_monty(), 1);
    }

    #[test]
    fn test_inv_various() {
        for &val in &[1u32, 2, 7, 100, 999999, P - 1] {
            let a = BabyBear::to_monty(val);
            let a_inv = a.inv();
            assert_eq!(
                a.mul(a_inv).from_monty(),
                1,
                "inv failed for {}",
                val
            );
        }
    }

    #[test]
    fn test_reduce() {
        // reduce should take a u64 and return the value in Montgomery form
        let val = 21u64;
        let r = BabyBear::reduce(val);
        assert_eq!(r.from_monty(), 21);
    }

    #[test]
    fn test_reduce_large() {
        let val = P as u64 + 5;
        let r = BabyBear::reduce(val);
        assert_eq!(r.from_monty(), 5);
    }

    #[test]
    fn test_monty_reduce_one_squared() {
        // monty_reduce(R_MOD_P * R_MOD_P) should give R_MOD_P (1 * 1 = 1 in monty)
        let t = R_MOD_P as u64 * R_MOD_P as u64;
        assert_eq!(BabyBear::monty_reduce(t), R_MOD_P);
    }

    #[test]
    fn test_two_adic_generator_order() {
        // g^(2^27) should equal 1 (order divides 2^27).
        let g = BabyBear::two_adic_generator(TWO_ADICITY);
        let mut val = g;
        for _ in 0..TWO_ADICITY {
            val = val.mul(val);
        }
        assert_eq!(val, BabyBear::one(), "g^(2^27) should be 1");
    }

    #[test]
    fn test_two_adic_generator_primitive() {
        // g^(2^26) should NOT be 1 (primitive root check)
        let g = BabyBear::two_adic_generator(TWO_ADICITY);
        let mut val = g;
        for _ in 0..(TWO_ADICITY - 1) {
            val = val.mul(val);
        }
        assert_ne!(val, BabyBear::one(), "g^(2^26) should NOT be 1");
    }

    #[test]
    fn test_two_adic_generator_subgroup() {
        // Generator for 2^4 subgroup should have order exactly 16
        let g = BabyBear::two_adic_generator(4);
        let mut val = BabyBear::one();
        for i in 1..=16 {
            val = val.mul(g);
            if i < 16 {
                assert_ne!(val, BabyBear::one(), "order too small at {}", i);
            }
        }
        assert_eq!(val, BabyBear::one(), "g^16 should be 1");
    }

    #[test]
    fn test_generate_twiddles() {
        let log_n = 3u32;
        let twiddles = BabyBear::generate_twiddles(log_n);
        assert_eq!(twiddles.len(), 3);
        // Layer 0 has 1 entry (the identity)
        assert_eq!(twiddles[0].len(), 1);
        assert_eq!(twiddles[0][0], BabyBear::one());
        // Layer 1 has 2 entries
        assert_eq!(twiddles[1].len(), 2);
        // Layer 2 has 4 entries
        assert_eq!(twiddles[2].len(), 4);

        // First twiddle of every layer should be 1
        for layer in &twiddles {
            assert_eq!(layer[0], BabyBear::one());
        }
    }

    #[test]
    fn test_constants_consistency() {
        // Verify R_MOD_P is the Montgomery form of 1
        assert_eq!(BabyBear::to_monty(1).0, R_MOD_P);

        // Verify MONTY_CONST: (P * MONTY_CONST + 1) mod 2^32 == 0
        let check = (P as u64 * MONTY_CONST as u64 + 1) % (1u64 << 32);
        assert_eq!(check, 0, "MONTY_CONST verification failed");
    }
}
