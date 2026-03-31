// Variant 3: Stockham radix-2 out-of-place NTT with ping-pong buffers.
//
// Key difference from Variants 1 & 2: reads from buffer A, writes to buffer B,
// then swaps. This avoids read-after-write hazards within a dispatch and tests
// whether the out-of-place memory pattern helps on Apple UMA.
//
// Two-phase execution:
//   Phase 1 (device memory): Large-stride layers, one dispatch per layer.
//     Each dispatch reads from one device buffer and writes to the other.
//   Phase 2 (threadgroup): Small-stride layers. Each threadgroup loads a tile,
//     ping-pongs between two threadgroup arrays (2 x 4096 = 32 KB), writes
//     the result to the output device buffer.
//
// Uses the same CT-DIT butterfly and twiddle factors as Variants 1 & 2.
// Output is bit-reversed (same as CPU reference forward NTT).

#ifndef NTT_STOCKHAM_R2_H
#define NTT_STOCKHAM_R2_H

#include "m31_field.metal"

// Tile size for ping-pong: 4096 elements per array x 4 bytes = 16 KB each.
// Two arrays (tile_a + tile_b) = 32 KB total (Apple Silicon threadgroup budget).
#define STOCKHAM_TILE_SIZE 4096

// ─── Forward CT-DIT butterfly stages (threadgroup, ping-pong) ────────────
//
// Each threadgroup loads one tile into tile_a, then ping-pongs between
// tile_a and tile_b for each layer. Writes the final result to the output
// device buffer.
//
// params layout:
//   [0] = tile_log      (log2 of tile size, max 12)
//   [1] = num_layers    (layers processed in threadgroup)
//   [2] = start_layer   (highest layer index, counting down)
//   [3..3+num_layers]   = per-layer twiddle offset
kernel void stockham_r2_forward_tg(
    device const uint* input       [[buffer(0)]],
    device uint* output            [[buffer(1)]],
    device const uint* twiddles    [[buffer(2)]],
    device const uint* params      [[buffer(3)]],
    uint tg_id                     [[threadgroup_position_in_grid]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    threadgroup uint tile_a[STOCKHAM_TILE_SIZE];
    threadgroup uint tile_b[STOCKHAM_TILE_SIZE];

    uint tile_log   = params[0];
    uint num_layers = params[1];
    uint start_layer = params[2];
    uint tile_size  = 1u << tile_log;
    uint tile_off   = tg_id * tile_size;

    // Load tile from input device buffer into tile_a
    for (uint i = tid; i < tile_size; i += tg_size) {
        tile_a[i] = input[tile_off + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Ping-pong through layers
    uint half_tile = tile_size >> 1;
    bool read_from_a = true;

    for (uint li = 0; li < num_layers; li++) {
        uint layer  = start_layer - li;
        uint stride = 1u << layer;
        uint full   = stride << 1;
        uint blocks_per_tile = tile_size / full;
        uint tw_off = params[3 + li];
        uint global_block_base = tg_id * blocks_per_tile;

        for (uint i = tid; i < half_tile; i += tg_size) {
            uint block = i / stride;
            uint j     = i % stride;
            uint idx0  = block * full + j;
            uint idx1  = idx0 + stride;

            uint tw = twiddles[tw_off + global_block_base + block];

            uint a, b;
            if (read_from_a) {
                a = tile_a[idx0]; b = tile_a[idx1];
            } else {
                a = tile_b[idx0]; b = tile_b[idx1];
            }

            uint t = m31_mul(b, tw);
            uint out0 = m31_add(a, t);
            uint out1 = m31_sub(a, t);

            if (read_from_a) {
                tile_b[idx0] = out0; tile_b[idx1] = out1;
            } else {
                tile_a[idx0] = out0; tile_a[idx1] = out1;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        read_from_a = !read_from_a;
    }

    // Write result (from whichever array is current) to output device buffer
    for (uint i = tid; i < tile_size; i += tg_size) {
        output[tile_off + i] = read_from_a ? tile_a[i] : tile_b[i];
    }
}

// ─── Forward CT-DIT butterfly (device memory, out-of-place) ──────────────
//
// Reads from input buffer, writes to output buffer. Caller swaps buffer
// roles after each dispatch.
//
// params[0] = stride, params[1] = n
kernel void stockham_r2_butterfly_device(
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

    uint tw = twiddles[block];
    uint a = input[idx0];
    uint b = input[idx1];
    uint t = m31_mul(b, tw);
    output[idx0] = m31_add(a, t);
    output[idx1] = m31_sub(a, t);
}

#endif // NTT_STOCKHAM_R2_H
