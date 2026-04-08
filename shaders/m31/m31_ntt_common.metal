// Common NTT butterfly operations shared across all algorithm variants.
//
// The DCCT (Circle NTT) butterfly is identical to standard NTT butterfly:
//   Forward (CT-DIT): a' = a + w*b,  b' = a - w*b
//   Inverse (GS-DIF): a' = a + b,    b' = (a - b) * w_inv
//
// The circle-group structure is absorbed entirely into twiddle factor
// precomputation (y-coords for layer 0, x-coords for layer 1, doubling-map
// for deeper layers). Runtime butterfly logic is indistinguishable from
// standard radix-2 NTT.
//
// Verified against Stwo and Lambdaworks source code.

#ifndef NTT_COMMON_H
#define NTT_COMMON_H

#include "../fields/m31_field.metal"

// Forward butterfly (CT-DIT): in-place
//   a' = a + w*b
//   b' = a - w*b
inline void butterfly_forward(thread uint &a, thread uint &b, uint twiddle) {
    uint t = m31_mul(b, twiddle);
    b = m31_sub(a, t);
    a = m31_add(a, t);
}

// Inverse butterfly (GS-DIF): in-place
//   a' = a + b
//   b' = (a - b) * w_inv
inline void butterfly_inverse(thread uint &a, thread uint &b, uint twiddle_inv) {
    uint sum = m31_add(a, b);
    uint diff = m31_sub(a, b);
    a = sum;
    b = m31_mul(diff, twiddle_inv);
}

// Radix-4 forward butterfly: 4 inputs, 3 twiddle multiplications.
// Halves the number of barrier synchronizations compared to radix-2.
inline void butterfly_radix4_forward(
    thread uint &a0, thread uint &a1,
    thread uint &a2, thread uint &a3,
    uint w1, uint w2, uint w3
) {
    // Stage 1: two radix-2 butterflies
    uint t0 = m31_mul(a2, w2);
    uint t1 = m31_mul(a3, w2);
    uint b0 = m31_add(a0, t0);
    uint b1 = m31_add(a1, t1);
    uint b2 = m31_sub(a0, t0);
    uint b3 = m31_sub(a1, t1);

    // Stage 2: two more butterflies with different twiddles
    uint t2 = m31_mul(b1, w1);
    uint t3 = m31_mul(b3, w3);
    a0 = m31_add(b0, t2);
    a1 = m31_sub(b0, t2);
    a2 = m31_add(b2, t3);
    a3 = m31_sub(b2, t3);
}

#endif // NTT_COMMON_H
