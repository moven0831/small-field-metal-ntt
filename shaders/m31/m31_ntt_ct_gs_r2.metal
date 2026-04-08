// Variant 2: CT-DIT forward / GS-DIF inverse, radix-2, in-place.
//
// Two-phase execution strategy:
//   Phase 1 (threadgroup): Small-stride layers processed in on-chip memory.
//     One dispatch covers all small-stride layers. Each threadgroup loads a
//     tile of elements, runs multiple butterfly stages with threadgroup barriers,
//     writes back once. Cost: 1 device read + 1 device write.
//
//   Phase 2 (device memory): Large-stride layers where butterfly pairs span
//     beyond a single tile. One dispatch per remaining layer.
//
// Key property: NO bit-reversal permutation between forward and inverse.
// Forward (CT-DIT) produces bit-reversed output; inverse (GS-DIF) expects it.

#ifndef NTT_CT_GS_R2_H
#define NTT_CT_GS_R2_H

#include "../fields/m31_field.metal"

// Maximum threadgroup tile: 8192 elements = 32 KB.
// Handles up to 13 butterfly stages on-chip.
#define CT_GS_R2_MAX_TILE 8192

// ─── Forward CT-DIT butterfly stages (threadgroup memory) ────────────────
//
// Each threadgroup loads one tile, runs multiple butterfly stages on-chip,
// then writes back. Dramatically reduces device memory traffic compared to
// per-stage dispatches.
//
// params layout:
//   [0] = n             (total elements)
//   [1] = tile_log      (log2 of tile size)
//   [2] = num_layers    (layers to process in threadgroup)
//   [3] = start_layer   (highest layer index; processes start_layer down)
//   [4..4+num_layers]   = per-layer twiddle offset into flat twiddles buffer
kernel void ct_gs_r2_forward_tg(
    device uint* data              [[buffer(0)]],
    device const uint* twiddles    [[buffer(1)]],
    device const uint* params      [[buffer(2)]],
    uint tg_id                     [[threadgroup_position_in_grid]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    threadgroup uint tile[CT_GS_R2_MAX_TILE];

    uint tile_log   = params[1];
    uint num_layers = params[2];
    uint start_layer = params[3];
    uint tile_size  = 1u << tile_log;
    uint tile_off   = tg_id * tile_size;

    // Load tile from device memory
    for (uint i = tid; i < tile_size; i += tg_size) {
        tile[i] = data[tile_off + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Forward CT-DIT: layers from start_layer down
    uint half_tile = tile_size >> 1;
    for (uint li = 0; li < num_layers; li++) {
        uint layer  = start_layer - li;
        uint stride = 1u << layer;
        uint full   = stride << 1;
        uint blocks_per_tile = tile_size / full;
        uint tw_off = params[4 + li];
        uint global_block_base = tg_id * blocks_per_tile;

        for (uint i = tid; i < half_tile; i += tg_size) {
            uint block = i / stride;
            uint j     = i % stride;
            uint idx0  = block * full + j;
            uint idx1  = idx0 + stride;

            uint tw = twiddles[tw_off + global_block_base + block];
            uint a = tile[idx0];
            uint b = tile[idx1];
            uint t = m31_mul(b, tw);
            tile[idx0] = m31_add(a, t);
            tile[idx1] = m31_sub(a, t);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write tile back to device memory
    for (uint i = tid; i < tile_size; i += tg_size) {
        data[tile_off + i] = tile[i];
    }
}

// ─── Inverse GS-DIF butterfly stages (threadgroup memory) ────────────────
//
// params layout: same structure as forward, but:
//   start_layer = lowest layer index (processes start_layer upward)
kernel void ct_gs_r2_inverse_tg(
    device uint* data              [[buffer(0)]],
    device const uint* twiddles    [[buffer(1)]],
    device const uint* params      [[buffer(2)]],
    uint tg_id                     [[threadgroup_position_in_grid]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    threadgroup uint tile[CT_GS_R2_MAX_TILE];

    uint tile_log   = params[1];
    uint num_layers = params[2];
    uint start_layer = params[3];
    uint tile_size  = 1u << tile_log;
    uint tile_off   = tg_id * tile_size;

    // Load tile
    for (uint i = tid; i < tile_size; i += tg_size) {
        tile[i] = data[tile_off + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Inverse GS-DIF: layers from start_layer upward
    uint half_tile = tile_size >> 1;
    for (uint li = 0; li < num_layers; li++) {
        uint layer  = start_layer + li;
        uint stride = 1u << layer;
        uint full   = stride << 1;
        uint blocks_per_tile = tile_size / full;
        uint tw_off = params[4 + li];
        uint global_block_base = tg_id * blocks_per_tile;

        for (uint i = tid; i < half_tile; i += tg_size) {
            uint block = i / stride;
            uint j     = i % stride;
            uint idx0  = block * full + j;
            uint idx1  = idx0 + stride;

            uint tw = twiddles[tw_off + global_block_base + block];
            uint a = tile[idx0];
            uint b = tile[idx1];
            tile[idx0] = m31_add(a, b);
            tile[idx1] = m31_mul(m31_sub(a, b), tw);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write tile back
    for (uint i = tid; i < tile_size; i += tg_size) {
        data[tile_off + i] = tile[i];
    }
}

// ─── Forward CT-DIT butterfly (device memory, single stage) ──────────────
//
// For large-stride layers that exceed threadgroup tile capacity.
// params[0] = stride, params[1] = n
kernel void ct_gs_r2_butterfly_device(
    device uint* data            [[buffer(0)]],
    device const uint* twiddles  [[buffer(1)]],
    device const uint* params    [[buffer(2)]],
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

    uint tw = twiddles[block];
    uint a = data[idx0];
    uint b = data[idx1];
    uint t = m31_mul(b, tw);
    data[idx0] = m31_add(a, t);
    data[idx1] = m31_sub(a, t);
}

// ─── Inverse GS-DIF butterfly (device memory, single stage) ─────────────
//
// params[0] = stride, params[1] = n
kernel void ct_gs_r2_butterfly_device_inv(
    device uint* data            [[buffer(0)]],
    device const uint* twiddles  [[buffer(1)]],
    device const uint* params    [[buffer(2)]],
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

    uint tw = twiddles[block];
    uint a = data[idx0];
    uint b = data[idx1];
    data[idx0] = m31_add(a, b);
    data[idx1] = m31_mul(m31_sub(a, b), tw);
}

// ─── Element-wise multiply by scalar (inverse NTT normalization) ─────────
//
// params[0] = n, params[1] = scalar (inv_n in M31)
kernel void ct_gs_r2_normalize(
    device uint* data            [[buffer(0)]],
    device const uint* params    [[buffer(1)]],
    uint tid                     [[thread_position_in_grid]]
) {
    uint n      = params[0];
    uint scalar = params[1];
    if (tid >= n) return;
    data[tid] = m31_mul(data[tid], scalar);
}

#endif // NTT_CT_GS_R2_H
