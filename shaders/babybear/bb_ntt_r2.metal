// BabyBear radix-2 NTT kernels (CT-DIT/GS-DIF) for Metal GPU.
//
// Single-column + batched variants + LDE helper kernels.

#include "bb_common.metal"

// ─── Forward GS-DIF butterfly stages (threadgroup memory) ────────────────
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
            tile[idx0] = bb_add(a, b);
            tile[idx1] = bb_mul(bb_sub(a, b), tw);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = tid; i < tile_size; i += tg_size) {
        data[tile_off + i] = tile[i];
    }
}

// ─── Inverse CT-DIT butterfly stages (threadgroup memory) ────────────────
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

// ─── Forward GS-DIF butterfly (device memory, single stage) ──────────────
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
    data[idx0] = bb_add(a, b);
    data[idx1] = bb_mul(bb_sub(a, b), tw);
}

// ─── Inverse CT-DIT butterfly (device memory, single stage) ─────────────
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
    uint t = bb_mul(b, tw);
    data[idx0] = bb_add(a, t);
    data[idx1] = bb_sub(a, t);
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

// Batched forward GS-DIF butterfly (device memory)
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
    data[idx0] = bb_add(a, b);
    data[idx1] = bb_mul(bb_sub(a, b), tw);
}

// Batched inverse CT-DIT butterfly (device memory)
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
    uint t = bb_mul(b, tw);
    data[idx0] = bb_add(a, t);
    data[idx1] = bb_sub(a, t);
}

// Batched forward GS-DIF threadgroup kernel (1D grid, col computed from tg_id)
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
            tile[idx0] = bb_add(a, b);
            tile[idx1] = bb_mul(bb_sub(a, b), tw);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = tid; i < tile_size; i += tg_size) {
        data[col_offset + tile_off + i] = tile[i];
    }
}

// Batched inverse CT-DIT threadgroup kernel (1D grid)
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

// ─── Fused normalize + zero-pad + coset-shift (for LDE pipeline) ────
//
// Replaces 3 separate dispatches (normalize, zero_pad, coset_shift) with 1.
// Reads from iDFT output buffer (n_orig per col), writes to extended output
// buffer (n_ext per col) with normalization and coset shift applied.
// params: [n_orig, n_ext, batch_size, inv_n]
kernel void bb_fused_normalize_zeropad_shift(
    device const uint* input         [[buffer(0)]],
    device uint* output              [[buffer(1)]],
    device const uint* shift_powers  [[buffer(2)]],
    device const uint* params        [[buffer(3)]],
    uint2 tid                        [[thread_position_in_grid]]
) {
    uint n_orig = params[0];
    uint n_ext  = params[1];
    uint inv_n  = params[3];
    uint col    = tid.y;
    uint row    = tid.x;
    if (row >= n_ext) return;

    uint out_idx = col * n_ext + row;
    if (row < n_orig) {
        uint in_idx = col * n_orig + row;
        uint val = bb_mul(input[in_idx], inv_n);     // normalize
        output[out_idx] = bb_mul(val, shift_powers[row]); // coset shift
    } else {
        output[out_idx] = 0;  // zero-pad (shift of 0 is still 0)
    }
}
