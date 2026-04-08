// Variant 1: Cooley-Tukey DIT radix-2 NTT with explicit bit-reversal.
//
// This is the NAIVE BASELINE — intentionally unoptimized:
// - Separate bit-reversal permutation kernel
// - One dispatch per butterfly stage (no stage fusion)
// - All data in device memory (no threadgroup optimization)
//
// Purpose: show the cost of zero optimization on Metal GPU.
// All other variants should beat this.

#ifndef NTT_CT_DIT_R2_H
#define NTT_CT_DIT_R2_H

#include "../fields/m31_field.metal"

/// Bit-reversal permutation kernel.
/// Reorders data[] so that data[i] goes to data[bit_reverse(i, log_n)].
/// Uses an output buffer to avoid in-place conflicts.
///
/// params[0] = n (number of elements)
/// params[1] = log_n
kernel void bitrev_permute(
    device const uint* input  [[buffer(0)]],
    device uint* output       [[buffer(1)]],
    device const uint* params [[buffer(2)]],
    uint tid                  [[thread_position_in_grid]]
) {
    uint n = params[0];
    uint log_n = params[1];
    if (tid >= n) return;

    // Compute bit-reversed index
    uint rev = 0;
    uint val = tid;
    for (uint i = 0; i < log_n; i++) {
        rev = (rev << 1) | (val & 1);
        val >>= 1;
    }

    output[rev] = input[tid];
}

/// Single-stage CT-DIT butterfly kernel (device memory, no threadgroup).
///
/// Each thread handles one butterfly pair at the given stage.
/// stride = 2^stage, so butterfly pairs are (i, i+stride) within blocks of 2*stride.
///
/// params[0] = stride (2^stage)
/// params[1] = n (total elements)
kernel void ct_dit_r2_butterfly_stage(
    device uint* data            [[buffer(0)]],
    device const uint* twiddles  [[buffer(1)]],
    device const uint* params    [[buffer(2)]],
    uint tid                     [[thread_position_in_grid]]
) {
    uint stride = params[0];
    uint n = params[1];
    uint full = stride << 1;

    // Map thread to butterfly pair
    uint block = tid / stride;
    uint j = tid % stride;
    uint idx0 = block * full + j;
    uint idx1 = idx0 + stride;

    if (idx1 >= n) return;

    uint tw = twiddles[block];
    uint a = data[idx0];
    uint b = data[idx1];

    uint t = m31_mul(b, tw);
    data[idx0] = m31_add(a, t);
    data[idx1] = m31_sub(a, t);
}

#endif // NTT_CT_DIT_R2_H
