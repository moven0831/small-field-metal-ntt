// BabyBear field arithmetic for Metal compute shaders.
// p = 2^31 - 2^27 + 1 = 2013265921
//
// Uses Montgomery multiplication for fast modular multiply on GPU.
// Values are stored in Montgomery form: val_monty = val * R mod p, where R = 2^32.
// Addition and subtraction operate directly on Montgomery-form values.
//
// All functions operate on uint (32-bit unsigned). Intermediate products use ulong (64-bit).
// Included by NTT variant shaders via #include.

#ifndef BABYBEAR_FIELD_H
#define BABYBEAR_FIELD_H

#include <metal_stdlib>
using namespace metal;

constant uint BB_P = 0x78000001u;           // 2013265921
constant uint BB_MONTY_CONST = 0x77FFFFFFu; // -p^{-1} mod 2^32 = 2013265919
constant uint BB_R_MOD_P = 0x0FFFFFFEu;     // R mod p = 268435454 (Montgomery form of 1)

// Montgomery reduction: given T (a ulong), compute T * R^{-1} mod p.
inline uint bb_monty_reduce(ulong t) {
    uint t_lo = uint(t & 0xFFFFFFFF);
    uint m = t_lo * BB_MONTY_CONST;    // mod 2^32 implicit
    ulong mp = ulong(m) * ulong(BB_P);
    uint result = uint((t + mp) >> 32);
    return (result >= BB_P) ? (result - BB_P) : result;
}

inline uint bb_add(uint a, uint b) {
    uint sum = a + b;
    return (sum >= BB_P) ? (sum - BB_P) : sum;
}

inline uint bb_sub(uint a, uint b) {
    return (a >= b) ? (a - b) : (BB_P - (b - a));
}

inline uint bb_mul(uint a, uint b) {
    ulong product = ulong(a) * ulong(b);
    return bb_monty_reduce(product);
}

#endif // BABYBEAR_FIELD_H
