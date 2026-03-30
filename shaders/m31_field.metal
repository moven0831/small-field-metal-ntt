// M31 field arithmetic for Metal compute shaders.
// p = 2^31 - 1 (Mersenne prime)
//
// Reduction: reduce(a) = (a >> 31) + (a & 0x7FFFFFFF), conditional subtract if >= p.
// This is the fastest reduction of any cryptographic prime.
//
// All functions operate on uint (32-bit unsigned). Intermediate products use ulong (64-bit).
// Included by all NTT variant shaders via #include.

#ifndef M31_FIELD_H
#define M31_FIELD_H

#include <metal_stdlib>
using namespace metal;

constant uint M31_P = 0x7FFFFFFFu; // 2^31 - 1

inline uint m31_reduce(ulong val) {
    uint lo = uint(val & 0x7FFFFFFF);
    uint hi = uint(val >> 31);
    uint sum = lo + hi;
    return (sum >= M31_P) ? (sum - M31_P) : sum;
}

inline uint m31_add(uint a, uint b) {
    ulong sum = ulong(a) + ulong(b);
    return m31_reduce(sum);
}

inline uint m31_sub(uint a, uint b) {
    return (a >= b) ? (a - b) : (M31_P - (b - a));
}

inline uint m31_mul(uint a, uint b) {
    ulong product = ulong(a) * ulong(b);
    return m31_reduce(product);
}

#endif // M31_FIELD_H
