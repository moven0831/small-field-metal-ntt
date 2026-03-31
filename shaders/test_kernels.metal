// Test kernels for validating the Metal compute pipeline.
// These are used by integration tests, not by the NTT benchmark.

#include "m31_field.metal"

/// Simple kernel: element-wise M31 addition of two buffers.
/// out[i] = m31_add(a[i], b[i])
kernel void m31_add_kernel(
    device const uint* a [[buffer(0)]],
    device const uint* b [[buffer(1)]],
    device uint* out     [[buffer(2)]],
    uint tid             [[thread_position_in_grid]]
) {
    out[tid] = m31_add(a[tid], b[tid]);
}

/// Simple kernel: element-wise M31 multiplication.
/// out[i] = m31_mul(a[i], b[i])
kernel void m31_mul_kernel(
    device const uint* a [[buffer(0)]],
    device const uint* b [[buffer(1)]],
    device uint* out     [[buffer(2)]],
    uint tid             [[thread_position_in_grid]]
) {
    out[tid] = m31_mul(a[tid], b[tid]);
}

/// Single-stage forward butterfly kernel for testing.
/// Applies butterfly_forward to pairs at distance `stride`:
///   data[i] and data[i + stride] for i in [block_start..block_start + stride)
///
/// Parameters packed in params buffer:
///   params[0] = stride
///   params[1] = n (total elements)
kernel void butterfly_forward_kernel(
    device uint* data            [[buffer(0)]],
    device const uint* twiddles  [[buffer(1)]],
    device const uint* params    [[buffer(2)]],
    uint tid                     [[thread_position_in_grid]]
) {
    uint stride = params[0];
    uint n = params[1];
    uint full = stride << 1;

    // Each thread handles one butterfly pair
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
