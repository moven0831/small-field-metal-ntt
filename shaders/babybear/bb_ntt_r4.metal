// BabyBear radix-4 NTT kernels (CT-GS) for Metal GPU.
//
// Single-column + batched variants. Two layers per dispatch (halves barrier count).

#include "bb_common.metal"

// Radix-4 device forward butterfly (GS-DIF, two layers k, k-1)
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

    // Stage 1: outer butterfly (GS-DIF, layer k)
    uint s0 = bb_add(a0, a2);
    uint s2 = bb_mul(bb_sub(a0, a2), w2a);
    uint s1 = bb_add(a1, a3);
    uint s3 = bb_mul(bb_sub(a1, a3), w2b);

    // Stage 2: inner butterfly (GS-DIF, layer k-1)
    data[idx0] = bb_add(s0, s1);
    data[idx1] = bb_mul(bb_sub(s0, s1), w1);
    data[idx2] = bb_add(s2, s3);
    data[idx3] = bb_mul(bb_sub(s2, s3), w1);
}

// Radix-4 device inverse butterfly (CT-DIT, two layers k-1, k)
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

    // Stage 1: inner butterfly (CT-DIT, layer k-1)
    uint t0 = bb_mul(a1, w1);
    uint t1 = bb_mul(a3, w1);
    uint b0 = bb_add(a0, t0);
    uint b1 = bb_sub(a0, t0);
    uint b2 = bb_add(a2, t1);
    uint b3 = bb_sub(a2, t1);

    // Stage 2: outer butterfly (CT-DIT, layer k)
    uint t2 = bb_mul(b2, w2a);
    uint t3 = bb_mul(b3, w2b);
    data[idx0] = bb_add(b0, t2);
    data[idx1] = bb_add(b1, t3);
    data[idx2] = bb_sub(b0, t2);
    data[idx3] = bb_sub(b1, t3);
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
    data[idx0] = bb_add(a, b);
    data[idx1] = bb_mul(bb_sub(a, b), tw);
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
    uint t = bb_mul(b, tw);
    data[idx0] = bb_add(a, t);
    data[idx1] = bb_sub(a, t);
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

            // GS-DIF radix-4: outer (layer k) then inner (layer k-1)
            uint s0 = bb_add(a0, a2);
            uint s2 = bb_mul(bb_sub(a0, a2), w2a);
            uint s1 = bb_add(a1, a3);
            uint s3 = bb_mul(bb_sub(a1, a3), w2b);

            tile[idx0] = bb_add(s0, s1);
            tile[idx1] = bb_mul(bb_sub(s0, s1), w1);
            tile[idx2] = bb_add(s2, s3);
            tile[idx3] = bb_mul(bb_sub(s2, s3), w1);
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
            tile[idx0] = bb_add(a, b);
            tile[idx1] = bb_mul(bb_sub(a, b), tw);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = tid; i < tile_size; i += tg_size) {
        data[tile_off + i] = tile[i];
    }
}

// Radix-4 inverse threadgroup kernel (CT-DIT)
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

    // Optional initial radix-2 stage (layer 0, CT-DIT)
    if (has_initial_r2) {
        uint tw_off = params[tw_idx];
        tw_idx++;
        for (uint i = tid; i < half_tile; i += tg_size) {
            uint idx0 = i * 2;
            uint idx1 = idx0 + 1;
            uint tw = twiddles[tw_off];
            uint a = tile[idx0];
            uint b = tile[idx1];
            uint t = bb_mul(b, tw);
            tile[idx0] = bb_add(a, t);
            tile[idx1] = bb_sub(a, t);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Radix-4 stages (CT-DIT, pairs of layers, low to high)
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

            // CT-DIT inverse radix-4: inner (layer k-1) then outer (layer k)
            uint t0 = bb_mul(a1, w1);
            uint t1 = bb_mul(a3, w1);
            uint b0 = bb_add(a0, t0);
            uint b1 = bb_sub(a0, t0);
            uint b2 = bb_add(a2, t1);
            uint b3 = bb_sub(a2, t1);

            uint t2 = bb_mul(b2, w2a);
            uint t3 = bb_mul(b3, w2b);
            tile[idx0] = bb_add(b0, t2);
            tile[idx1] = bb_add(b1, t3);
            tile[idx2] = bb_sub(b0, t2);
            tile[idx3] = bb_sub(b1, t3);
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

// ═══════════════════════════════════════════════════════════════════════
// Batched Radix-4 kernels (for coset LDE pipeline)
// SoA layout: data[col * n + row]. Batch via 2D grid (x=butterflies, y=col).
// ═══════════════════════════════════════════════════════════════════════

// Batched radix-4 forward device butterfly (GS-DIF)
// params: [outer, inner, n, batch_size]
kernel void bb_r4_batch_butterfly_device(
    device uint* data                 [[buffer(0)]],
    device const uint* twiddles_outer [[buffer(1)]],
    device const uint* twiddles_inner [[buffer(2)]],
    device const uint* params         [[buffer(3)]],
    uint2 tid                         [[thread_position_in_grid]]
) {
    uint outer = params[0];
    uint inner = params[1];
    uint n     = params[2];
    uint quad  = outer << 1;
    uint col   = tid.y;
    uint col_offset = col * n;

    uint q = tid.x / inner;
    uint j = tid.x % inner;
    uint base = q * quad + j;
    if (base + outer + inner >= n) return;

    uint idx0 = col_offset + base;
    uint idx1 = col_offset + base + inner;
    uint idx2 = col_offset + base + outer;
    uint idx3 = col_offset + base + outer + inner;

    uint a0 = data[idx0], a1 = data[idx1], a2 = data[idx2], a3 = data[idx3];
    uint w2a = twiddles_outer[j];
    uint w2b = twiddles_outer[j + inner];
    uint w1  = twiddles_inner[j];

    // GS-DIF: outer (layer k) then inner (layer k-1)
    uint s0 = bb_add(a0, a2);
    uint s2 = bb_mul(bb_sub(a0, a2), w2a);
    uint s1 = bb_add(a1, a3);
    uint s3 = bb_mul(bb_sub(a1, a3), w2b);

    data[idx0] = bb_add(s0, s1);
    data[idx1] = bb_mul(bb_sub(s0, s1), w1);
    data[idx2] = bb_add(s2, s3);
    data[idx3] = bb_mul(bb_sub(s2, s3), w1);
}

// Batched radix-4 inverse device butterfly (CT-DIT)
// params: [outer, inner, n, batch_size]
kernel void bb_r4_batch_butterfly_device_inv(
    device uint* data                 [[buffer(0)]],
    device const uint* twiddles_outer [[buffer(1)]],
    device const uint* twiddles_inner [[buffer(2)]],
    device const uint* params         [[buffer(3)]],
    uint2 tid                         [[thread_position_in_grid]]
) {
    uint outer = params[0];
    uint inner = params[1];
    uint n     = params[2];
    uint quad  = outer << 1;
    uint col   = tid.y;
    uint col_offset = col * n;

    uint q = tid.x / inner;
    uint j = tid.x % inner;
    uint base = q * quad + j;
    if (base + outer + inner >= n) return;

    uint idx0 = col_offset + base;
    uint idx1 = col_offset + base + inner;
    uint idx2 = col_offset + base + outer;
    uint idx3 = col_offset + base + outer + inner;

    uint a0 = data[idx0], a1 = data[idx1], a2 = data[idx2], a3 = data[idx3];
    uint w2a = twiddles_outer[j];
    uint w2b = twiddles_outer[j + inner];
    uint w1  = twiddles_inner[j];

    // CT-DIT: inner (layer k-1) then outer (layer k)
    uint t0 = bb_mul(a1, w1);
    uint t1 = bb_mul(a3, w1);
    uint b0 = bb_add(a0, t0);
    uint b1 = bb_sub(a0, t0);
    uint b2 = bb_add(a2, t1);
    uint b3 = bb_sub(a2, t1);

    uint t2 = bb_mul(b2, w2a);
    uint t3 = bb_mul(b3, w2b);
    data[idx0] = bb_add(b0, t2);
    data[idx1] = bb_add(b1, t3);
    data[idx2] = bb_sub(b0, t2);
    data[idx3] = bb_sub(b1, t3);
}

// Batched radix-2 fallback for odd device layers (forward)
// params: [stride, n, batch_size]
kernel void bb_r4_batch_butterfly_device_r2(
    device uint* data            [[buffer(0)]],
    device const uint* twiddles  [[buffer(1)]],
    device const uint* params    [[buffer(2)]],
    uint2 tid                    [[thread_position_in_grid]]
) {
    uint stride = params[0];
    uint n      = params[1];
    uint full   = stride << 1;
    uint col    = tid.y;
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

// Batched radix-2 fallback for odd device layers (inverse)
// params: [stride, n, batch_size]
kernel void bb_r4_batch_r2_device_inv(
    device uint* data            [[buffer(0)]],
    device const uint* twiddles  [[buffer(1)]],
    device const uint* params    [[buffer(2)]],
    uint2 tid                    [[thread_position_in_grid]]
) {
    uint stride = params[0];
    uint n      = params[1];
    uint full   = stride << 1;
    uint col    = tid.y;
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

// Batched radix-4 forward threadgroup kernel (1D grid, col from tg_id)
// params: [n, tile_log, num_r4, has_final_r2, num_tiles_per_col, tw_offsets...]
kernel void bb_r4_batch_forward_tg(
    device uint* data              [[buffer(0)]],
    device const uint* twiddles    [[buffer(1)]],
    device const uint* params      [[buffer(2)]],
    uint tg_id                     [[threadgroup_position_in_grid]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    threadgroup uint tile[BB_R2_MAX_TILE];

    uint n                  = params[0];
    uint tile_log           = params[1];
    uint num_r4             = params[2];
    uint has_final_r2       = params[3];
    uint num_tiles_per_col  = params[4];
    uint tile_size          = 1u << tile_log;

    uint col          = tg_id / num_tiles_per_col;
    uint tile_in_col  = tg_id % num_tiles_per_col;
    uint col_offset   = col * n;
    uint tile_off     = tile_in_col * tile_size;

    for (uint i = tid; i < tile_size; i += tg_size) {
        tile[i] = data[col_offset + tile_off + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint quarter_tile = tile_size >> 2;
    uint half_tile    = tile_size >> 1;

    uint layer = tile_log - 1;
    for (uint s = 0; s < num_r4; s++) {
        uint k = layer;
        uint outer = 1u << k;
        uint inner = outer >> 1;
        uint quad  = outer << 1;
        uint tw_outer_off = params[5 + 2 * s];
        uint tw_inner_off = params[5 + 2 * s + 1];

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

            // GS-DIF radix-4: outer (layer k) then inner (layer k-1)
            uint s0 = bb_add(a0, a2);
            uint s2 = bb_mul(bb_sub(a0, a2), w2a);
            uint s1 = bb_add(a1, a3);
            uint s3 = bb_mul(bb_sub(a1, a3), w2b);

            tile[idx0] = bb_add(s0, s1);
            tile[idx1] = bb_mul(bb_sub(s0, s1), w1);
            tile[idx2] = bb_add(s2, s3);
            tile[idx3] = bb_mul(bb_sub(s2, s3), w1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        layer -= 2;
    }

    if (has_final_r2) {
        uint tw_off = params[5 + 2 * num_r4];
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

    for (uint i = tid; i < tile_size; i += tg_size) {
        data[col_offset + tile_off + i] = tile[i];
    }
}

// Batched radix-4 inverse threadgroup kernel (CT-DIT, 1D grid)
// params: [n, tile_log, num_r4, has_initial_r2, num_tiles_per_col, tw_offsets...]
kernel void bb_r4_batch_inverse_tg(
    device uint* data              [[buffer(0)]],
    device const uint* twiddles    [[buffer(1)]],
    device const uint* params      [[buffer(2)]],
    uint tg_id                     [[threadgroup_position_in_grid]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    threadgroup uint tile[BB_R2_MAX_TILE];

    uint n                  = params[0];
    uint tile_log           = params[1];
    uint num_r4             = params[2];
    uint has_initial_r2     = params[3];
    uint num_tiles_per_col  = params[4];
    uint tile_size          = 1u << tile_log;

    uint col          = tg_id / num_tiles_per_col;
    uint tile_in_col  = tg_id % num_tiles_per_col;
    uint col_offset   = col * n;
    uint tile_off     = tile_in_col * tile_size;

    for (uint i = tid; i < tile_size; i += tg_size) {
        tile[i] = data[col_offset + tile_off + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint quarter_tile = tile_size >> 2;
    uint half_tile    = tile_size >> 1;
    uint tw_idx       = 5;

    if (has_initial_r2) {
        uint tw_off = params[tw_idx];
        tw_idx++;
        for (uint i = tid; i < half_tile; i += tg_size) {
            uint idx0 = i * 2;
            uint idx1 = idx0 + 1;
            uint tw = twiddles[tw_off];
            uint a = tile[idx0];
            uint b = tile[idx1];
            uint t = bb_mul(b, tw);
            tile[idx0] = bb_add(a, t);
            tile[idx1] = bb_sub(a, t);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint layer = has_initial_r2 ? 1u : 0u;
    for (uint s = 0; s < num_r4; s++) {
        uint k = layer + 1;
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

            // CT-DIT inverse radix-4: inner (layer k-1) then outer (layer k)
            uint t0 = bb_mul(a1, w1);
            uint t1 = bb_mul(a3, w1);
            uint b0 = bb_add(a0, t0);
            uint b1 = bb_sub(a0, t0);
            uint b2 = bb_add(a2, t1);
            uint b3 = bb_sub(a2, t1);

            uint t2 = bb_mul(b2, w2a);
            uint t3 = bb_mul(b3, w2b);
            tile[idx0] = bb_add(b0, t2);
            tile[idx1] = bb_add(b1, t3);
            tile[idx2] = bb_sub(b0, t2);
            tile[idx3] = bb_sub(b1, t3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        layer += 2;
    }

    for (uint i = tid; i < tile_size; i += tg_size) {
        data[col_offset + tile_off + i] = tile[i];
    }
}

// Batched radix-4 normalize
// params[0] = n, params[1] = scalar, params[2] = batch_size
kernel void bb_r4_batch_normalize(
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
