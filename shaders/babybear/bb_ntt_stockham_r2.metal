// BabyBear Stockham radix-2 out-of-place NTT (ping-pong buffers) for Metal GPU.

#include "bb_common.metal"

#define BB_STOCKHAM_TILE_SIZE 4096

// Stockham forward GS-DIF (threadgroup, ping-pong)
// params: [tile_log, num_layers, start_layer, tw_offsets...]
kernel void bb_stockham_r2_forward_tg(
    device const uint* input       [[buffer(0)]],
    device uint* output            [[buffer(1)]],
    device const uint* twiddles    [[buffer(2)]],
    device const uint* params      [[buffer(3)]],
    uint tg_id                     [[threadgroup_position_in_grid]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    threadgroup uint tile_a[BB_STOCKHAM_TILE_SIZE];
    threadgroup uint tile_b[BB_STOCKHAM_TILE_SIZE];

    uint tile_log   = params[0];
    uint num_layers = params[1];
    uint start_layer = params[2];
    uint tile_size  = 1u << tile_log;
    uint tile_off   = tg_id * tile_size;

    for (uint i = tid; i < tile_size; i += tg_size) {
        tile_a[i] = input[tile_off + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint half_tile = tile_size >> 1;
    bool read_from_a = true;

    for (uint li = 0; li < num_layers; li++) {
        uint layer  = start_layer - li;
        uint stride = 1u << layer;
        uint full   = stride << 1;
        uint tw_off = params[3 + li];

        for (uint i = tid; i < half_tile; i += tg_size) {
            uint block = i / stride;
            uint j     = i % stride;
            uint idx0  = block * full + j;
            uint idx1  = idx0 + stride;

            uint tw = twiddles[tw_off + j];

            uint a, b;
            if (read_from_a) { a = tile_a[idx0]; b = tile_a[idx1]; }
            else             { a = tile_b[idx0]; b = tile_b[idx1]; }

            uint out0 = bb_add(a, b);
            uint out1 = bb_mul(bb_sub(a, b), tw);

            if (read_from_a) { tile_b[idx0] = out0; tile_b[idx1] = out1; }
            else             { tile_a[idx0] = out0; tile_a[idx1] = out1; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        read_from_a = !read_from_a;
    }

    for (uint i = tid; i < tile_size; i += tg_size) {
        output[tile_off + i] = read_from_a ? tile_a[i] : tile_b[i];
    }
}

// Stockham forward GS-DIF (device memory, out-of-place)
// params[0] = stride, params[1] = n
kernel void bb_stockham_r2_butterfly_device(
    device const uint* input     [[buffer(0)]],
    device uint* output          [[buffer(1)]],
    device const uint* twiddles  [[buffer(2)]],
    device const uint* params    [[buffer(3)]],
    uint tid                     [[thread_position_in_grid]]
) {
    uint stride = params[0];
    uint n      = params[1];
    uint full   = stride << 1;

    uint block = tid / stride;
    uint j     = tid % stride;
    uint idx0  = block * full + j;
    uint idx1  = idx0 + stride;
    if (idx1 >= n) return;

    uint tw = twiddles[j];
    uint a = input[idx0];
    uint b = input[idx1];
    output[idx0] = bb_add(a, b);
    output[idx1] = bb_mul(bb_sub(a, b), tw);
}

// Stockham inverse CT-DIT (threadgroup, ping-pong)
// params: [tile_log, num_layers, start_layer, tw_offsets...]
kernel void bb_stockham_r2_inverse_tg(
    device const uint* input       [[buffer(0)]],
    device uint* output            [[buffer(1)]],
    device const uint* twiddles    [[buffer(2)]],
    device const uint* params      [[buffer(3)]],
    uint tg_id                     [[threadgroup_position_in_grid]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    threadgroup uint tile_a[BB_STOCKHAM_TILE_SIZE];
    threadgroup uint tile_b[BB_STOCKHAM_TILE_SIZE];

    uint tile_log   = params[0];
    uint num_layers = params[1];
    uint start_layer = params[2];
    uint tile_size  = 1u << tile_log;
    uint tile_off   = tg_id * tile_size;

    for (uint i = tid; i < tile_size; i += tg_size) {
        tile_a[i] = input[tile_off + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint half_tile = tile_size >> 1;
    bool read_from_a = true;

    for (uint li = 0; li < num_layers; li++) {
        uint layer  = start_layer + li;
        uint stride = 1u << layer;
        uint full   = stride << 1;
        uint tw_off = params[3 + li];

        for (uint i = tid; i < half_tile; i += tg_size) {
            uint block = i / stride;
            uint j     = i % stride;
            uint idx0  = block * full + j;
            uint idx1  = idx0 + stride;

            uint tw = twiddles[tw_off + j];

            uint a, b;
            if (read_from_a) { a = tile_a[idx0]; b = tile_a[idx1]; }
            else             { a = tile_b[idx0]; b = tile_b[idx1]; }

            uint t = bb_mul(b, tw);
            uint out0 = bb_add(a, t);
            uint out1 = bb_sub(a, t);

            if (read_from_a) { tile_b[idx0] = out0; tile_b[idx1] = out1; }
            else             { tile_a[idx0] = out0; tile_a[idx1] = out1; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        read_from_a = !read_from_a;
    }

    for (uint i = tid; i < tile_size; i += tg_size) {
        output[tile_off + i] = read_from_a ? tile_a[i] : tile_b[i];
    }
}

// Stockham inverse CT-DIT (device memory, out-of-place)
// params[0] = stride, params[1] = n
kernel void bb_stockham_r2_butterfly_device_inv(
    device const uint* input     [[buffer(0)]],
    device uint* output          [[buffer(1)]],
    device const uint* twiddles  [[buffer(2)]],
    device const uint* params    [[buffer(3)]],
    uint tid                     [[thread_position_in_grid]]
) {
    uint stride = params[0];
    uint n      = params[1];
    uint full   = stride << 1;

    uint block = tid / stride;
    uint j     = tid % stride;
    uint idx0  = block * full + j;
    uint idx1  = idx0 + stride;
    if (idx1 >= n) return;

    uint tw = twiddles[j];
    uint a = input[idx0];
    uint b = input[idx1];
    uint t = bb_mul(b, tw);
    output[idx0] = bb_add(a, t);
    output[idx1] = bb_sub(a, t);
}

// Stockham normalize (in-place)
// params[0] = n, params[1] = scalar
kernel void bb_stockham_r2_normalize(
    device uint* data            [[buffer(0)]],
    device const uint* params    [[buffer(1)]],
    uint tid                     [[thread_position_in_grid]]
) {
    uint n      = params[0];
    uint scalar = params[1];
    if (tid >= n) return;
    data[tid] = bb_mul(data[tid], scalar);
}
