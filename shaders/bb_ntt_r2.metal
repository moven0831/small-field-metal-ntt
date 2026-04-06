// BabyBear radix-2 NTT kernels for Metal GPU.
//
// CT-DIT forward / GS-DIF inverse, in-place with threadgroup memory.
// Uses BabyBear Montgomery field arithmetic (not Circle NTT).
//
// Key difference from M31 Circle NTT twiddle layout:
//   Standard NTT: twiddle at position j within stride, same for all blocks.
//   Circle NTT:   twiddle per block (different for each block).
//
// Phase 1 (threadgroup): Small-stride layers in on-chip memory.
// Phase 2 (device memory): Large-stride layers, one dispatch per layer.

#ifndef BB_NTT_R2_H
#define BB_NTT_R2_H

#include "babybear_field.metal"

#define BB_R2_MAX_TILE 8192

// ─── Forward CT-DIT butterfly stages (threadgroup memory) ────────────────
//
// Standard NTT twiddle layout: layer l has 2^l entries, indexed by position j.
// Flat twiddles buffer: [layer start_layer entries, layer start_layer-1 entries, ...]
//
// params layout:
//   [0] = n             (total elements)
//   [1] = tile_log      (log2 of tile size)
//   [2] = num_layers    (layers to process in threadgroup)
//   [3] = start_layer   (highest layer index; processes start_layer down)
//   [4..4+num_layers]   = per-layer twiddle offset into flat twiddles buffer
kernel void bb_r2_forward_tg(
    device uint* data              [[buffer(0)]],
    device const uint* twiddles    [[buffer(1)]],
    device const uint* params      [[buffer(2)]],
    uint tg_id                     [[threadgroup_position_in_grid]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    threadgroup uint tile[BB_R2_MAX_TILE];

    uint tile_log   = params[1];
    uint num_layers = params[2];
    uint start_layer = params[3];
    uint tile_size  = 1u << tile_log;
    uint tile_off   = tg_id * tile_size;

    for (uint i = tid; i < tile_size; i += tg_size) {
        tile[i] = data[tile_off + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint half_tile = tile_size >> 1;
    for (uint li = 0; li < num_layers; li++) {
        uint layer  = start_layer - li;
        uint stride = 1u << layer;
        uint full   = stride << 1;
        uint tw_off = params[4 + li];

        for (uint i = tid; i < half_tile; i += tg_size) {
            uint block = i / stride;
            uint j     = i % stride;
            uint idx0  = block * full + j;
            uint idx1  = idx0 + stride;

            // Standard NTT: twiddle indexed by j (position within stride)
            uint tw = twiddles[tw_off + j];
            uint a = tile[idx0];
            uint b = tile[idx1];
            uint t = bb_mul(b, tw);
            tile[idx0] = bb_add(a, t);
            tile[idx1] = bb_sub(a, t);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = tid; i < tile_size; i += tg_size) {
        data[tile_off + i] = tile[i];
    }
}

// ─── Inverse GS-DIF butterfly stages (threadgroup memory) ────────────────
//
// start_layer = lowest layer index (processes start_layer upward)
kernel void bb_r2_inverse_tg(
    device uint* data              [[buffer(0)]],
    device const uint* twiddles    [[buffer(1)]],
    device const uint* params      [[buffer(2)]],
    uint tg_id                     [[threadgroup_position_in_grid]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    threadgroup uint tile[BB_R2_MAX_TILE];

    uint tile_log   = params[1];
    uint num_layers = params[2];
    uint start_layer = params[3];
    uint tile_size  = 1u << tile_log;
    uint tile_off   = tg_id * tile_size;

    for (uint i = tid; i < tile_size; i += tg_size) {
        tile[i] = data[tile_off + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint half_tile = tile_size >> 1;
    for (uint li = 0; li < num_layers; li++) {
        uint layer  = start_layer + li;
        uint stride = 1u << layer;
        uint full   = stride << 1;
        uint tw_off = params[4 + li];

        for (uint i = tid; i < half_tile; i += tg_size) {
            uint block = i / stride;
            uint j     = i % stride;
            uint idx0  = block * full + j;
            uint idx1  = idx0 + stride;

            uint tw = twiddles[tw_off + j];
            uint a = tile[idx0];
            uint b = tile[idx1];
            tile[idx0] = bb_add(a, b);
            tile[idx1] = bb_mul(bb_sub(a, b), tw);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = tid; i < tile_size; i += tg_size) {
        data[tile_off + i] = tile[i];
    }
}

// ─── Forward CT-DIT butterfly (device memory, single stage) ──────────────
//
// Standard NTT: twiddle indexed by j (position within stride).
// params[0] = stride, params[1] = n
kernel void bb_r2_butterfly_device(
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

    uint tw = twiddles[j];
    uint a = data[idx0];
    uint b = data[idx1];
    uint t = bb_mul(b, tw);
    data[idx0] = bb_add(a, t);
    data[idx1] = bb_sub(a, t);
}

// ─── Inverse GS-DIF butterfly (device memory, single stage) ─────────────
//
// params[0] = stride, params[1] = n
kernel void bb_r2_butterfly_device_inv(
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

    uint tw = twiddles[j];
    uint a = data[idx0];
    uint b = data[idx1];
    data[idx0] = bb_add(a, b);
    data[idx1] = bb_mul(bb_sub(a, b), tw);
}

// ─── Element-wise multiply by scalar (inverse NTT normalization) ─────────
//
// params[0] = n, params[1] = scalar (inv_n in BabyBear Montgomery form)
kernel void bb_r2_normalize(
    device uint* data            [[buffer(0)]],
    device const uint* params    [[buffer(1)]],
    uint tid                     [[thread_position_in_grid]]
) {
    uint n      = params[0];
    uint scalar = params[1];
    if (tid >= n) return;
    data[tid] = bb_mul(data[tid], scalar);
}

// ─── Batched kernels ────────────────────────────────────────────────────
// SoA layout: data[col * n + row]. Batch dimension = grid y (for 2D kernels)
// or flattened into 1D grid (for threadgroup kernels).

// Batched forward CT-DIT butterfly (device memory)
// params[0] = stride, params[1] = n, params[2] = batch_size
kernel void bb_r2_batch_butterfly_device(
    device uint* data            [[buffer(0)]],
    device const uint* twiddles  [[buffer(1)]],
    device const uint* params    [[buffer(2)]],
    uint2 tid                    [[thread_position_in_grid]]
) {
    uint stride     = params[0];
    uint n          = params[1];
    uint full       = stride << 1;
    uint col        = tid.y;
    uint col_offset = col * n;

    uint block = tid.x / stride;
    uint j     = tid.x % stride;
    uint idx0  = col_offset + block * full + j;
    uint idx1  = idx0 + stride;
    if (block * full + j + stride >= n) return;

    uint tw = twiddles[j];
    uint a = data[idx0];
    uint b = data[idx1];
    uint t = bb_mul(b, tw);
    data[idx0] = bb_add(a, t);
    data[idx1] = bb_sub(a, t);
}

// Batched inverse GS-DIF butterfly (device memory)
// params[0] = stride, params[1] = n, params[2] = batch_size
kernel void bb_r2_batch_butterfly_device_inv(
    device uint* data            [[buffer(0)]],
    device const uint* twiddles  [[buffer(1)]],
    device const uint* params    [[buffer(2)]],
    uint2 tid                    [[thread_position_in_grid]]
) {
    uint stride     = params[0];
    uint n          = params[1];
    uint full       = stride << 1;
    uint col        = tid.y;
    uint col_offset = col * n;

    uint block = tid.x / stride;
    uint j     = tid.x % stride;
    uint idx0  = col_offset + block * full + j;
    uint idx1  = idx0 + stride;
    if (block * full + j + stride >= n) return;

    uint tw = twiddles[j];
    uint a = data[idx0];
    uint b = data[idx1];
    data[idx0] = bb_add(a, b);
    data[idx1] = bb_mul(bb_sub(a, b), tw);
}

// Batched forward CT-DIT threadgroup kernel (1D grid, col computed from tg_id)
// params: [n, tile_log, num_layers, start_layer, num_tiles_per_col, tw_offsets...]
kernel void bb_r2_batch_forward_tg(
    device uint* data              [[buffer(0)]],
    device const uint* twiddles    [[buffer(1)]],
    device const uint* params      [[buffer(2)]],
    uint tg_id                     [[threadgroup_position_in_grid]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    threadgroup uint tile[BB_R2_MAX_TILE];

    uint n              = params[0];
    uint tile_log       = params[1];
    uint num_layers     = params[2];
    uint start_layer    = params[3];
    uint num_tiles_per_col = params[4];
    uint tile_size      = 1u << tile_log;

    uint col            = tg_id / num_tiles_per_col;
    uint tile_in_col    = tg_id % num_tiles_per_col;
    uint col_offset     = col * n;
    uint tile_off       = tile_in_col * tile_size;

    for (uint i = tid; i < tile_size; i += tg_size) {
        tile[i] = data[col_offset + tile_off + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint half_tile = tile_size >> 1;
    for (uint li = 0; li < num_layers; li++) {
        uint layer  = start_layer - li;
        uint stride = 1u << layer;
        uint full   = stride << 1;
        uint tw_off = params[5 + li];

        for (uint i = tid; i < half_tile; i += tg_size) {
            uint block = i / stride;
            uint j     = i % stride;
            uint idx0  = block * full + j;
            uint idx1  = idx0 + stride;

            uint tw = twiddles[tw_off + j];
            uint a = tile[idx0];
            uint b = tile[idx1];
            uint t = bb_mul(b, tw);
            tile[idx0] = bb_add(a, t);
            tile[idx1] = bb_sub(a, t);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = tid; i < tile_size; i += tg_size) {
        data[col_offset + tile_off + i] = tile[i];
    }
}

// Batched inverse GS-DIF threadgroup kernel (1D grid)
// params: [n, tile_log, num_layers, start_layer, num_tiles_per_col, tw_offsets...]
kernel void bb_r2_batch_inverse_tg(
    device uint* data              [[buffer(0)]],
    device const uint* twiddles    [[buffer(1)]],
    device const uint* params      [[buffer(2)]],
    uint tg_id                     [[threadgroup_position_in_grid]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    threadgroup uint tile[BB_R2_MAX_TILE];

    uint n              = params[0];
    uint tile_log       = params[1];
    uint num_layers     = params[2];
    uint start_layer    = params[3];
    uint num_tiles_per_col = params[4];
    uint tile_size      = 1u << tile_log;

    uint col            = tg_id / num_tiles_per_col;
    uint tile_in_col    = tg_id % num_tiles_per_col;
    uint col_offset     = col * n;
    uint tile_off       = tile_in_col * tile_size;

    for (uint i = tid; i < tile_size; i += tg_size) {
        tile[i] = data[col_offset + tile_off + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint half_tile = tile_size >> 1;
    for (uint li = 0; li < num_layers; li++) {
        uint layer  = start_layer + li;
        uint stride = 1u << layer;
        uint full   = stride << 1;
        uint tw_off = params[5 + li];

        for (uint i = tid; i < half_tile; i += tg_size) {
            uint block = i / stride;
            uint j     = i % stride;
            uint idx0  = block * full + j;
            uint idx1  = idx0 + stride;

            uint tw = twiddles[tw_off + j];
            uint a = tile[idx0];
            uint b = tile[idx1];
            tile[idx0] = bb_add(a, b);
            tile[idx1] = bb_mul(bb_sub(a, b), tw);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = tid; i < tile_size; i += tg_size) {
        data[col_offset + tile_off + i] = tile[i];
    }
}

// Batched normalize (multiply all elements by scalar)
// params[0] = n, params[1] = scalar, params[2] = batch_size
kernel void bb_r2_batch_normalize(
    device uint* data            [[buffer(0)]],
    device const uint* params    [[buffer(1)]],
    uint2 tid                    [[thread_position_in_grid]]
) {
    uint n      = params[0];
    uint scalar = params[1];
    uint col    = tid.y;
    uint row    = tid.x;
    if (row >= n) return;
    uint idx = col * n + row;
    data[idx] = bb_mul(data[idx], scalar);
}

// ─── LDE helper kernels ────────────────────────────────────────────────

// Zero-pad: copy from input buffer to output buffer, zero the rest.
// SoA layout. params[0] = n_orig, params[1] = n_ext, params[2] = batch_size
kernel void bb_zero_pad_batch(
    device const uint* input     [[buffer(0)]],
    device uint* output          [[buffer(1)]],
    device const uint* params    [[buffer(2)]],
    uint2 tid                    [[thread_position_in_grid]]
) {
    uint n_orig = params[0];
    uint n_ext  = params[1];
    uint col    = tid.y;
    uint row    = tid.x;
    if (row >= n_ext) return;

    uint out_idx = col * n_ext + row;
    if (row < n_orig) {
        uint in_idx = col * n_orig + row;
        output[out_idx] = input[in_idx];
    } else {
        output[out_idx] = 0;
    }
}

// Coset shift: multiply each element by shift_powers[row].
// params[0] = n_ext, params[1] = batch_size
kernel void bb_coset_shift_batch(
    device uint* data                [[buffer(0)]],
    device const uint* shift_powers  [[buffer(1)]],
    device const uint* params        [[buffer(2)]],
    uint2 tid                        [[thread_position_in_grid]]
) {
    uint n_ext = params[0];
    uint col   = tid.y;
    uint row   = tid.x;
    if (row >= n_ext) return;

    uint idx = col * n_ext + row;
    data[idx] = bb_mul(data[idx], shift_powers[row]);
}

// ═══════════════════════════════════════════════════════════════════════
// V3: Stockham radix-2 out-of-place NTT (ping-pong buffers)
// ═══════════════════════════════════════════════════════════════════════

#define BB_STOCKHAM_TILE_SIZE 4096

// Stockham forward CT-DIT (threadgroup, ping-pong)
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

// Stockham forward CT-DIT (device memory, out-of-place)
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
    uint t = bb_mul(b, tw);
    output[idx0] = bb_add(a, t);
    output[idx1] = bb_sub(a, t);
}

// Stockham inverse GS-DIF (threadgroup, ping-pong)
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

// Stockham inverse GS-DIF (device memory, out-of-place)
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
    output[idx0] = bb_add(a, b);
    output[idx1] = bb_mul(bb_sub(a, b), tw);
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

// ═══════════════════════════════════════════════════════════════════════
// V4: CT-GS Radix-4 NTT (two layers per dispatch)
// ═══════════════════════════════════════════════════════════════════════

// Radix-4 device forward butterfly (two layers k, k-1)
// Standard NTT: two distinct outer twiddles + one shared inner twiddle
// Buffers: [data, twiddles_outer (layer k), twiddles_inner (layer k-1), params]
// params: [outer, inner, n]
kernel void bb_r4_butterfly_device(
    device uint* data                [[buffer(0)]],
    device const uint* twiddles_outer [[buffer(1)]],
    device const uint* twiddles_inner [[buffer(2)]],
    device const uint* params        [[buffer(3)]],
    uint tid                         [[thread_position_in_grid]]
) {
    uint outer = params[0];
    uint inner = params[1];
    uint n     = params[2];
    uint quad  = outer << 1;

    uint q = tid / inner;
    uint j = tid % inner;
    uint base = q * quad + j;
    if (base + outer + inner >= n) return;

    uint idx0 = base;
    uint idx1 = base + inner;
    uint idx2 = base + outer;
    uint idx3 = base + outer + inner;

    uint a0 = data[idx0];
    uint a1 = data[idx1];
    uint a2 = data[idx2];
    uint a3 = data[idx3];

    // Two distinct outer twiddles (layer k)
    uint w2a = twiddles_outer[j];
    uint w2b = twiddles_outer[j + inner];
    // One shared inner twiddle (layer k-1)
    uint w1 = twiddles_inner[j];

    // Stage 1: outer butterfly
    uint t0 = bb_mul(a2, w2a);
    uint t1 = bb_mul(a3, w2b);
    uint b0 = bb_add(a0, t0);
    uint b1 = bb_add(a1, t1);
    uint b2 = bb_sub(a0, t0);
    uint b3 = bb_sub(a1, t1);

    // Stage 2: inner butterfly
    uint t2 = bb_mul(b1, w1);
    uint t3 = bb_mul(b3, w1);
    data[idx0] = bb_add(b0, t2);
    data[idx1] = bb_sub(b0, t2);
    data[idx2] = bb_add(b2, t3);
    data[idx3] = bb_sub(b2, t3);
}

// Radix-4 device inverse butterfly (GS-DIF, two layers k-1, k)
// params: [outer, inner, n]
kernel void bb_r4_butterfly_device_inv(
    device uint* data                [[buffer(0)]],
    device const uint* twiddles_outer [[buffer(1)]],
    device const uint* twiddles_inner [[buffer(2)]],
    device const uint* params        [[buffer(3)]],
    uint tid                         [[thread_position_in_grid]]
) {
    uint outer = params[0];
    uint inner = params[1];
    uint n     = params[2];
    uint quad  = outer << 1;

    uint q = tid / inner;
    uint j = tid % inner;
    uint base = q * quad + j;
    if (base + outer + inner >= n) return;

    uint idx0 = base;
    uint idx1 = base + inner;
    uint idx2 = base + outer;
    uint idx3 = base + outer + inner;

    uint a0 = data[idx0];
    uint a1 = data[idx1];
    uint a2 = data[idx2];
    uint a3 = data[idx3];

    // Inverse of standard NTT radix-4
    // Two distinct outer inverse twiddles + one shared inner inverse twiddle
    uint w2a = twiddles_outer[j];
    uint w2b = twiddles_outer[j + inner];
    uint w1 = twiddles_inner[j];

    // Stage 1: undo inner (GS-DIF)
    uint s0 = bb_add(a0, a1);
    uint s1 = bb_mul(bb_sub(a0, a1), w1);
    uint s2 = bb_add(a2, a3);
    uint s3 = bb_mul(bb_sub(a2, a3), w1);

    // Stage 2: undo outer (GS-DIF)
    data[idx0] = bb_add(s0, s2);
    data[idx1] = bb_add(s1, s3);
    data[idx2] = bb_mul(bb_sub(s0, s2), w2a);
    data[idx3] = bb_mul(bb_sub(s1, s3), w2b);
}

// Radix-2 fallback for odd log_n (forward, device memory)
// Same as bb_r2_butterfly_device but with a distinct kernel name for pipeline
// params[0] = stride, params[1] = n
kernel void bb_r4_butterfly_device_r2(
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
    uint tw = twiddles[j];
    uint a = data[idx0];
    uint b = data[idx1];
    uint t = bb_mul(b, tw);
    data[idx0] = bb_add(a, t);
    data[idx1] = bb_sub(a, t);
}

// Radix-2 fallback (inverse, device memory)
// params[0] = stride, params[1] = n
kernel void bb_r4_r2_device_inv(
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
    uint tw = twiddles[j];
    uint a = data[idx0];
    uint b = data[idx1];
    data[idx0] = bb_add(a, b);
    data[idx1] = bb_mul(bb_sub(a, b), tw);
}

// Radix-4 forward threadgroup kernel
// Processes radix-4 pairs within tile, with optional final radix-2.
// params: [n, tile_log, num_r4, has_final_r2, tw_offsets...]
// tw_offsets come in pairs: [outer_off, inner_off] for each r4 stage,
// then one more offset for the final r2 stage if has_final_r2.
kernel void bb_r4_forward_tg(
    device uint* data              [[buffer(0)]],
    device const uint* twiddles    [[buffer(1)]],
    device const uint* params      [[buffer(2)]],
    uint tg_id                     [[threadgroup_position_in_grid]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    threadgroup uint tile[BB_R2_MAX_TILE];

    uint n              = params[0];
    uint tile_log       = params[1];
    uint num_r4         = params[2];
    uint has_final_r2   = params[3];
    uint tile_size      = 1u << tile_log;
    uint tile_off       = tg_id * tile_size;

    for (uint i = tid; i < tile_size; i += tg_size) {
        tile[i] = data[tile_off + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint quarter_tile = tile_size >> 2;
    uint half_tile    = tile_size >> 1;

    // Radix-4 stages (pairs of layers, high to low)
    uint layer = tile_log - 1;
    for (uint s = 0; s < num_r4; s++) {
        uint k = layer;
        uint outer = 1u << k;
        uint inner = outer >> 1;
        uint quad  = outer << 1;
        uint tw_outer_off = params[4 + 2 * s];
        uint tw_inner_off = params[4 + 2 * s + 1];

        for (uint i = tid; i < quarter_tile; i += tg_size) {
            uint q = i / inner;
            uint j = i % inner;
            uint idx0 = q * quad + j;
            uint idx1 = idx0 + inner;
            uint idx2 = idx0 + outer;
            uint idx3 = idx2 + inner;

            uint w2a = twiddles[tw_outer_off + j];
            uint w2b = twiddles[tw_outer_off + j + inner];
            uint w1  = twiddles[tw_inner_off + j];

            uint a0 = tile[idx0], a1 = tile[idx1], a2 = tile[idx2], a3 = tile[idx3];

            uint t0 = bb_mul(a2, w2a);
            uint t1 = bb_mul(a3, w2b);
            uint b0 = bb_add(a0, t0);
            uint b1 = bb_add(a1, t1);
            uint b2 = bb_sub(a0, t0);
            uint b3 = bb_sub(a1, t1);

            uint t2 = bb_mul(b1, w1);
            uint t3 = bb_mul(b3, w1);
            tile[idx0] = bb_add(b0, t2);
            tile[idx1] = bb_sub(b0, t2);
            tile[idx2] = bb_add(b2, t3);
            tile[idx3] = bb_sub(b2, t3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        layer -= 2;
    }

    // Optional final radix-2 stage (layer 0)
    if (has_final_r2) {
        uint tw_off = params[4 + 2 * num_r4];
        for (uint i = tid; i < half_tile; i += tg_size) {
            uint idx0 = i * 2;
            uint idx1 = idx0 + 1;
            uint tw = twiddles[tw_off]; // layer 0: stride=1, only twiddle[0]=1
            uint a = tile[idx0];
            uint b = tile[idx1];
            uint t = bb_mul(b, tw);
            tile[idx0] = bb_add(a, t);
            tile[idx1] = bb_sub(a, t);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = tid; i < tile_size; i += tg_size) {
        data[tile_off + i] = tile[i];
    }
}

// Radix-4 inverse threadgroup kernel (GS-DIF)
// params: [n, tile_log, num_r4, has_initial_r2, tw_offsets...]
kernel void bb_r4_inverse_tg(
    device uint* data              [[buffer(0)]],
    device const uint* twiddles    [[buffer(1)]],
    device const uint* params      [[buffer(2)]],
    uint tg_id                     [[threadgroup_position_in_grid]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    threadgroup uint tile[BB_R2_MAX_TILE];

    uint n              = params[0];
    uint tile_log       = params[1];
    uint num_r4         = params[2];
    uint has_initial_r2 = params[3];
    uint tile_size      = 1u << tile_log;
    uint tile_off       = tg_id * tile_size;

    for (uint i = tid; i < tile_size; i += tg_size) {
        tile[i] = data[tile_off + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint quarter_tile = tile_size >> 2;
    uint half_tile    = tile_size >> 1;
    uint tw_idx       = 4; // offset into params for tw_offsets

    // Optional initial radix-2 stage (layer 0, GS-DIF)
    if (has_initial_r2) {
        uint tw_off = params[tw_idx];
        tw_idx++;
        for (uint i = tid; i < half_tile; i += tg_size) {
            uint idx0 = i * 2;
            uint idx1 = idx0 + 1;
            uint tw = twiddles[tw_off];
            uint a = tile[idx0];
            uint b = tile[idx1];
            tile[idx0] = bb_add(a, b);
            tile[idx1] = bb_mul(bb_sub(a, b), tw);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Radix-4 stages (GS-DIF, pairs of layers, low to high)
    uint layer = has_initial_r2 ? 1u : 0u;
    for (uint s = 0; s < num_r4; s++) {
        uint k = layer + 1; // outer layer
        uint outer = 1u << k;
        uint inner = outer >> 1;
        uint quad  = outer << 1;
        uint tw_inner_off = params[tw_idx];
        uint tw_outer_off = params[tw_idx + 1];
        tw_idx += 2;

        for (uint i = tid; i < quarter_tile; i += tg_size) {
            uint q = i / inner;
            uint j = i % inner;
            uint idx0 = q * quad + j;
            uint idx1 = idx0 + inner;
            uint idx2 = idx0 + outer;
            uint idx3 = idx2 + inner;

            uint w2a = twiddles[tw_outer_off + j];
            uint w2b = twiddles[tw_outer_off + j + inner];
            uint w1  = twiddles[tw_inner_off + j];

            uint a0 = tile[idx0], a1 = tile[idx1], a2 = tile[idx2], a3 = tile[idx3];

            // GS-DIF inverse radix-4
            uint s0 = bb_add(a0, a1);
            uint s1 = bb_mul(bb_sub(a0, a1), w1);
            uint s2 = bb_add(a2, a3);
            uint s3 = bb_mul(bb_sub(a2, a3), w1);

            tile[idx0] = bb_add(s0, s2);
            tile[idx1] = bb_add(s1, s3);
            tile[idx2] = bb_mul(bb_sub(s0, s2), w2a);
            tile[idx3] = bb_mul(bb_sub(s1, s3), w2b);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        layer += 2;
    }

    for (uint i = tid; i < tile_size; i += tg_size) {
        data[tile_off + i] = tile[i];
    }
}

// Radix-4 normalize (same as r2)
kernel void bb_r4_normalize(
    device uint* data            [[buffer(0)]],
    device const uint* params    [[buffer(1)]],
    uint tid                     [[thread_position_in_grid]]
) {
    uint n      = params[0];
    uint scalar = params[1];
    if (tid >= n) return;
    data[tid] = bb_mul(data[tid], scalar);
}

#endif // BB_NTT_R2_H
