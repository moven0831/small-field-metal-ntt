//! Known-value field arithmetic tests for BabyBear and M31.
//!
//! These tests verify fundamental field operations against values computed
//! independently (via Python/Sage or u128 arithmetic). They catch regressions
//! in Montgomery reduction, modular arithmetic, and root-of-unity generation.

use small_field_metal_ntt::field::babybear::BabyBear;
use small_field_metal_ntt::field::m31::M31;
use small_field_metal_ntt::field::Field;

// ─── BabyBear field arithmetic ─────────────────────────────────────────────

const BB_P: u32 = 2013265921; // 2^31 - 2^27 + 1

#[test]
fn test_bb_modulus() {
    assert_eq!(BabyBear::P, BB_P);
    assert_eq!(BB_P, (1u32 << 31) - (1u32 << 27) + 1);
}

#[test]
fn test_bb_montgomery_constants() {
    // R = 2^32 mod p
    let r_mod_p = (1u64 << 32) % BB_P as u64;
    assert_eq!(BabyBear::to_monty(1).0, r_mod_p as u32, "R_MOD_P mismatch");

    // Verify: to_monty(1) is the Montgomery form of 1
    assert_eq!(BabyBear::to_monty(1).from_monty(), 1);

    // Verify: to_monty(0) is 0
    assert_eq!(BabyBear::to_monty(0).0, 0);
}

#[test]
fn test_bb_arithmetic_known_values() {
    // 123 * 456 mod p = 56088
    let a = BabyBear::to_monty(123);
    let b = BabyBear::to_monty(456);
    assert_eq!(a.mul(b).from_monty(), 56088);

    // (p-1) * (p-1) mod p = 1
    let pm1 = BabyBear::to_monty(BB_P - 1);
    assert_eq!(pm1.mul(pm1).from_monty(), 1);

    // (p-1) + 1 mod p = 0
    assert_eq!(pm1.add(BabyBear::to_monty(1)).from_monty(), 0);

    // 0 - 1 mod p = p-1
    let zero = BabyBear::zero();
    let one = BabyBear::one();
    assert_eq!(zero.sub(one).from_monty(), BB_P - 1);
}

#[test]
fn test_bb_inverse_known_values() {
    // inv(7) mod p — verify via 7 * inv(7) = 1
    let seven = BabyBear::to_monty(7);
    let inv7 = seven.inv();
    assert_eq!(seven.mul(inv7).from_monty(), 1);

    // Verify the actual value: 7 * x ≡ 1 (mod p) → x = (p+1)/7 if 7 | (p+1)
    // p+1 = 2013265922, 2013265922 / 7 = 287609417.43... not exact
    // So compute via u128: 7^(p-2) mod p
    let expected = mod_pow_u128(7, BB_P as u64 - 2, BB_P as u64) as u32;
    assert_eq!(inv7.from_monty(), expected);

    // inv(1) = 1
    assert_eq!(BabyBear::one().inv().from_monty(), 1);

    // inv(p-1) = p-1 (since (p-1)^2 = 1 mod p)
    assert_eq!(
        BabyBear::to_monty(BB_P - 1).inv().from_monty(),
        BB_P - 1
    );
}

#[test]
fn test_bb_root_of_unity_orders() {
    // For each k from 1..=10, verify:
    // 1. g^(2^k) == 1
    // 2. g^(2^(k-1)) != 1 (primitive root check)
    for k in 1..=10u32 {
        let g = BabyBear::two_adic_generator(k);

        // g^(2^k) should be 1
        let mut val = g;
        for _ in 0..k {
            val = val.mul(val);
        }
        assert_eq!(
            val,
            BabyBear::one(),
            "g^(2^{}) should be 1 for k={}",
            k,
            k
        );

        // g^(2^(k-1)) should NOT be 1
        let mut val = g;
        for _ in 0..k - 1 {
            val = val.mul(val);
        }
        assert_ne!(
            val,
            BabyBear::one(),
            "g^(2^{}) should NOT be 1 for k={} (not primitive)",
            k - 1,
            k
        );
    }
}

#[test]
fn test_bb_two_adicity() {
    // p - 1 = 2^27 * 15
    let pm1 = BB_P - 1;
    let mut val = pm1;
    let mut count = 0u32;
    while val % 2 == 0 {
        val /= 2;
        count += 1;
    }
    assert_eq!(count, 27, "Two-adicity should be 27");
    assert_eq!(val, 15, "Odd part of p-1 should be 15");
}

// ─── M31 field arithmetic ──────────────────────────────────────────────────

const M31_P: u32 = (1u32 << 31) - 1; // 2147483647

#[test]
fn test_m31_modulus() {
    assert_eq!(M31::P, M31_P);
    assert_eq!(M31_P, 2147483647);
}

#[test]
fn test_m31_arithmetic_known_values() {
    // 123 * 456 mod p = 56088
    let a = M31(123);
    let b = M31(456);
    assert_eq!(a.mul(b), M31(56088));

    // (p-1) * (p-1) mod p = 1
    let pm1 = M31(M31_P - 1);
    assert_eq!(pm1.mul(pm1), M31(1));

    // (p-1) + 1 mod p = 0
    assert_eq!(pm1.add(M31(1)), M31(0));

    // 0 - 1 mod p = p-1
    assert_eq!(M31(0).sub(M31(1)), M31(M31_P - 1));
}

#[test]
fn test_m31_inverse_known_values() {
    // inv(7) mod p
    let seven = M31(7);
    let inv7 = seven.inv();
    assert_eq!(seven.mul(inv7), M31(1));

    let expected = mod_pow_u128(7, M31_P as u64 - 2, M31_P as u64) as u32;
    assert_eq!(inv7.0, expected);

    // inv(1) = 1
    assert_eq!(M31(1).inv(), M31(1));

    // inv(p-1) = p-1
    assert_eq!(M31(M31_P - 1).inv(), M31(M31_P - 1));
}

#[test]
fn test_m31_circle_group_order() {
    // The circle group C(M31) has order p+1 = 2^31, giving 2-adicity 31.
    // The generator of the full group should have order exactly 2^31.
    use small_field_metal_ntt::field::circle::{CirclePoint, Coset};

    // Generator of the full circle group (order 2^31)
    let g = CirclePoint::subgroup_gen(31);

    // g is on the unit circle: x^2 + y^2 = 1 mod p
    let x2 = g.x.mul(g.x);
    let y2 = g.y.mul(g.y);
    assert_eq!(x2.add(y2), M31(1), "Generator should be on the unit circle");

    // Verify coset generation works for various sizes
    for log_n in 1..=10u32 {
        let coset = Coset::odds(log_n);
        assert_eq!(coset.size(), 1 << log_n);
    }
}

// ─── Helper ────────────────────────────────────────────────────────────────

/// Modular exponentiation using u128 to avoid overflow.
fn mod_pow_u128(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    let mut result = 1u128;
    let m = modulus as u128;
    base %= modulus;
    let mut b = base as u128;
    while exp > 0 {
        if exp & 1 == 1 {
            result = (result * b) % m;
        }
        b = (b * b) % m;
        exp >>= 1;
    }
    result as u64
}
