use super::m31::M31;

/// A point on the circle x^2 + y^2 = 1 over M31.
///
/// The circle group has order p+1 = 2^31. Group operation is:
///   (x0, y0) + (x1, y1) = (x0*x1 - y0*y1, x0*y1 + y0*x1)
///
/// This is equivalent to complex multiplication restricted to the unit circle.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CirclePoint {
    pub x: M31,
    pub y: M31,
}

/// Generator of the circle group of order 2^31 over M31.
/// Verified against Stwo: (2, 1268011823).
pub const CIRCLE_GEN: CirclePoint = CirclePoint {
    x: M31(2),
    y: M31(1268011823),
};

/// Log of the circle group order.
pub const CIRCLE_LOG_ORDER: u32 = 31;

/// Identity element of the circle group: (1, 0).
pub const CIRCLE_IDENTITY: CirclePoint = CirclePoint {
    x: M31(1),
    y: M31(0),
};

impl CirclePoint {
    /// Circle group addition (complex multiplication on the unit circle).
    pub fn add(self, other: CirclePoint) -> CirclePoint {
        use super::Field;
        CirclePoint {
            x: self.x.mul(other.x).sub(self.y.mul(other.y)),
            y: self.x.mul(other.y).add(self.y.mul(other.x)),
        }
    }

    /// Circle group negation: -(x, y) = (x, -y).
    pub fn neg(self) -> CirclePoint {
        use super::Field;
        CirclePoint {
            x: self.x,
            y: M31(0).sub(self.y),
        }
    }

    /// Point doubling on the circle: double(x, y) = (2x^2 - 1, 2xy).
    pub fn double(self) -> CirclePoint {
        use super::Field;
        let two = M31(2);
        CirclePoint {
            x: two.mul(self.x.mul(self.x)).sub(M31(1)),
            y: two.mul(self.x.mul(self.y)),
        }
    }

    /// Scalar multiplication: self * n (repeated doubling).
    pub fn mul_scalar(self, mut n: u32) -> CirclePoint {
        let mut result = CIRCLE_IDENTITY;
        let mut base = self;
        while n > 0 {
            if n & 1 == 1 {
                result = result.add(base);
            }
            base = base.add(base);
            n >>= 1;
        }
        result
    }

    /// Generator of the subgroup of order 2^log_size.
    /// This is CIRCLE_GEN^(2^(31 - log_size)).
    pub fn subgroup_gen(log_size: u32) -> CirclePoint {
        assert!(log_size <= CIRCLE_LOG_ORDER);
        let mut p = CIRCLE_GEN;
        for _ in 0..(CIRCLE_LOG_ORDER - log_size) {
            p = p.double();
        }
        p
    }
}

/// A coset of a circle subgroup: initial + i * step, for i in 0..2^log_size.
#[derive(Clone, Debug)]
pub struct Coset {
    pub initial: CirclePoint,
    pub step: CirclePoint,
    pub log_size: u32,
}

impl Coset {
    /// The subgroup of order 2^log_size (initial = identity).
    pub fn subgroup(log_size: u32) -> Coset {
        Coset {
            initial: CIRCLE_IDENTITY,
            step: CirclePoint::subgroup_gen(log_size),
            log_size,
        }
    }

    /// The "odds" coset used as the canonic evaluation domain.
    /// odds(n) = G_{2n} + <G_n>, where G_k is the subgroup generator of order k.
    pub fn odds(log_size: u32) -> Coset {
        let step = CirclePoint::subgroup_gen(log_size);
        // initial = subgroup_gen(log_size + 1) = step of the 2x-size subgroup
        let initial = CirclePoint::subgroup_gen(log_size + 1);
        Coset {
            initial,
            step,
            log_size,
        }
    }

    /// The "half odds" coset.
    /// half_odds(n) = G_{4n} + <G_n>
    pub fn half_odds(log_size: u32) -> Coset {
        let step = CirclePoint::subgroup_gen(log_size);
        let initial = CirclePoint::subgroup_gen(log_size + 2);
        Coset {
            initial,
            step,
            log_size,
        }
    }

    pub fn size(&self) -> usize {
        1 << self.log_size
    }

    /// Get the i-th point of the coset.
    pub fn at(&self, i: usize) -> CirclePoint {
        self.initial.add(self.step.mul_scalar(i as u32))
    }

    /// Double the coset: maps each point (x,y) -> (2x^2-1, 2xy).
    pub fn double(&self) -> Coset {
        Coset {
            initial: self.initial.double(),
            step: self.step.double(),
            log_size: self.log_size - 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::Field;

    #[test]
    fn test_generator_on_circle() {
        // Verify (2, 1268011823) is on the circle: x^2 + y^2 = 1 mod p
        let x2 = CIRCLE_GEN.x.mul(CIRCLE_GEN.x);
        let y2 = CIRCLE_GEN.y.mul(CIRCLE_GEN.y);
        assert_eq!(x2.add(y2), M31(1));
    }

    #[test]
    fn test_identity() {
        let p = CIRCLE_GEN;
        let sum = p.add(CIRCLE_IDENTITY);
        assert_eq!(sum, p);
    }

    #[test]
    fn test_negation() {
        let p = CIRCLE_GEN;
        let neg = p.neg();
        let sum = p.add(neg);
        assert_eq!(sum, CIRCLE_IDENTITY);
    }

    #[test]
    fn test_generator_order() {
        // Generator has order 2^31. Doubling 31 times should give identity.
        let mut p = CIRCLE_GEN;
        for _ in 0..CIRCLE_LOG_ORDER {
            p = p.double();
        }
        assert_eq!(p, CIRCLE_IDENTITY);
    }

    #[test]
    fn test_generator_not_smaller_order() {
        // After 30 doublings, should NOT be identity (order is exactly 2^31, not 2^30).
        let mut p = CIRCLE_GEN;
        for _ in 0..(CIRCLE_LOG_ORDER - 1) {
            p = p.double();
        }
        assert_ne!(p, CIRCLE_IDENTITY);
    }

    #[test]
    fn test_subgroup_gen() {
        // subgroup_gen(4) should have order 2^4 = 16
        let g = CirclePoint::subgroup_gen(4);
        let mut p = g;
        for _ in 0..3 {
            p = p.double();
        }
        // After 3 doublings of a 2^4-order generator, we get a point of order 2.
        // One more doubling gives identity.
        let p2 = p.double();
        assert_eq!(p2, CIRCLE_IDENTITY);
        // But p itself is not identity
        assert_ne!(p, CIRCLE_IDENTITY);
    }

    #[test]
    fn test_double_formula() {
        // double(x,y) should equal (x,y) + (x,y) using the group operation
        let p = CIRCLE_GEN;
        let doubled = p.double();
        let added = p.add(p);
        assert_eq!(doubled, added);
    }

    #[test]
    fn test_coset_points_on_circle() {
        let coset = Coset::odds(4);
        for i in 0..coset.size() {
            let p = coset.at(i);
            let x2 = p.x.mul(p.x);
            let y2 = p.y.mul(p.y);
            assert_eq!(x2.add(y2), M31(1), "Point {} not on circle", i);
        }
    }
}
