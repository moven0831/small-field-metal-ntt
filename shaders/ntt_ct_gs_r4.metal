// Variant 4: CT-DIT/GS-DIF radix-4, in-place, with threadgroup memory.
//
// Each radix-4 stage replaces two radix-2 stages, halving the barrier count.
// For 2^20 NTT: 10 radix-4 barriers vs 20 radix-2 barriers.
//
// When log_n is odd, the final stage falls back to radix-2.
// Uses the same threadgroup tile (8192 elements, 32 KB) as Variant 2.

#ifndef NTT_CT_GS_R4_H
#define NTT_CT_GS_R4_H

#include "m31_field.metal"

#define CT_GS_R4_MAX_TILE 8192

// ─── Radix-4 + radix-2 stages in threadgroup memory ─────────────────────
//
// Processes paired layers as radix-4, with optional final radix-2 stage.
//
// params layout:
//   [0] = tile_log
//   [1] = num_r4_stages    (radix-4 paired stages)
//   [2] = has_final_r2     (0 or 1)
//   [3..3+2*num_r4_stages] = (tw_outer_off, tw_inner_off) per r4 stage
//   [3+2*num_r4_stages]    = tw_r2_off (if has_final_r2)
kernel void ct_gs_r4_forward_tg(
    device uint* data              [[buffer(0)]],
    device const uint* twiddles    [[buffer(1)]],
    device const uint* params      [[buffer(2)]],
    uint tg_id                     [[threadgroup_position_in_grid]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    threadgroup uint tile[CT_GS_R4_MAX_TILE];

    uint tile_log      = params[0];
    uint num_r4        = params[1];
    uint has_final_r2  = params[2];
    uint tile_size     = 1u << tile_log;
    uint tile_off      = tg_id * tile_size;

    // Load tile
    for (uint i = tid; i < tile_size; i += tg_size) {
        tile[i] = data[tile_off + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Start layer = tile_log - 1, processing downward in pairs
    uint start_layer = tile_log - 1;
    uint quarter_tile = tile_size >> 2;

    // ── Radix-4 stages ──────────────────────────────────────────────
    for (uint s = 0; s < num_r4; s++) {
        uint k = start_layer - 2 * s;          // outer layer
        uint outer = 1u << k;
        uint inner = outer >> 1;                // 2^(k-1)
        uint quad  = outer << 1;                // 2^(k+1)

        uint tw_outer_off = params[3 + 2 * s];
        uint tw_inner_off = params[3 + 2 * s + 1];

        uint quads_per_tile = tile_size / quad;
        uint blocks_inner_per_tile = tile_size / outer;  // = 2 * quads_per_tile
        uint global_quad_base = tg_id * quads_per_tile;
        uint global_inner_base = tg_id * blocks_inner_per_tile;

        for (uint i = tid; i < quarter_tile; i += tg_size) {
            uint q = i / inner;
            uint j = i % inner;
            uint idx0 = q * quad + j;
            uint idx1 = idx0 + inner;
            uint idx2 = idx0 + outer;
            uint idx3 = idx2 + inner;

            uint w2 = twiddles[tw_outer_off + global_quad_base + q];
            uint w1 = twiddles[tw_inner_off + global_inner_base + 2 * q];
            uint w3 = twiddles[tw_inner_off + global_inner_base + 2 * q + 1];

            uint a0 = tile[idx0], a1 = tile[idx1];
            uint a2 = tile[idx2], a3 = tile[idx3];

            // Stage 1: outer layer butterfly (w2)
            uint t0 = m31_mul(a2, w2);
            uint t1 = m31_mul(a3, w2);
            uint b0 = m31_add(a0, t0);
            uint b1 = m31_add(a1, t1);
            uint b2 = m31_sub(a0, t0);
            uint b3 = m31_sub(a1, t1);

            // Stage 2: inner layer butterfly (w1 for top, w3 for bottom)
            uint t2 = m31_mul(b1, w1);
            uint t3 = m31_mul(b3, w3);
            tile[idx0] = m31_add(b0, t2);
            tile[idx1] = m31_sub(b0, t2);
            tile[idx2] = m31_add(b2, t3);
            tile[idx3] = m31_sub(b2, t3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Final radix-2 stage (if odd number of layers) ───────────────
    if (has_final_r2) {
        uint tw_r2_off = params[3 + 2 * num_r4];
        uint half_tile = tile_size >> 1;
        // Layer 0: stride=1, full=2
        uint blocks_per_tile = half_tile;
        uint global_block_base = tg_id * blocks_per_tile;

        for (uint i = tid; i < half_tile; i += tg_size) {
            uint block = i;
            uint idx0 = block * 2;
            uint idx1 = idx0 + 1;

            uint tw = twiddles[tw_r2_off + global_block_base + block];
            uint a = tile[idx0];
            uint b = tile[idx1];
            uint t = m31_mul(b, tw);
            tile[idx0] = m31_add(a, t);
            tile[idx1] = m31_sub(a, t);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write tile back
    for (uint i = tid; i < tile_size; i += tg_size) {
        data[tile_off + i] = tile[i];
    }
}

// ─── Radix-4 butterfly (device memory, single paired stage) ─────────────
//
// Combines two layers (k, k-1) into one dispatch.
// params[0] = outer (2^k), params[1] = inner (2^(k-1)), params[2] = n
kernel void ct_gs_r4_butterfly_device(
    device uint* data              [[buffer(0)]],
    device const uint* tw_outer    [[buffer(1)]],
    device const uint* tw_inner    [[buffer(2)]],
    device const uint* params      [[buffer(3)]],
    uint tid                       [[thread_position_in_grid]]
) {
    uint outer = params[0];
    uint inner = params[1];
    uint n     = params[2];
    uint quad  = outer << 1;

    uint q = tid / inner;
    uint j = tid % inner;
    uint idx0 = q * quad + j;
    uint idx3 = idx0 + outer + inner;
    if (idx3 >= n) return;

    uint idx1 = idx0 + inner;
    uint idx2 = idx0 + outer;

    uint w2 = tw_outer[q];
    uint w1 = tw_inner[2 * q];
    uint w3 = tw_inner[2 * q + 1];

    uint a0 = data[idx0], a1 = data[idx1];
    uint a2 = data[idx2], a3 = data[idx3];

    uint t0 = m31_mul(a2, w2);
    uint t1 = m31_mul(a3, w2);
    uint b0 = m31_add(a0, t0);
    uint b1 = m31_add(a1, t1);
    uint b2 = m31_sub(a0, t0);
    uint b3 = m31_sub(a1, t1);

    uint t2 = m31_mul(b1, w1);
    uint t3 = m31_mul(b3, w3);
    data[idx0] = m31_add(b0, t2);
    data[idx1] = m31_sub(b0, t2);
    data[idx2] = m31_add(b2, t3);
    data[idx3] = m31_sub(b2, t3);
}

// ─── Radix-2 fallback (device memory, for odd layer counts) ─────────────
//
// params[0] = stride, params[1] = n
kernel void ct_gs_r4_butterfly_device_r2(
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

// ─── Inverse GS-DIF radix-4 butterfly stages (threadgroup memory) ──────────
//
// Inverse of ct_gs_r4_forward_tg. Processes layers from low to high in pairs.
// Each radix-4 butterfly undoes two CT-DIT layers by applying GS-DIF in reverse order.
//
// params layout:
//   [0] = tile_log
//   [1] = num_r4_stages
//   [2] = has_initial_r2     (0 or 1 — radix-2 layer 0 processed first if odd)
//   [3..] = twiddle offsets:
//     if has_initial_r2: [r2_off, then pairs (tw_inner_off, tw_outer_off) per r4 stage]
//     else: pairs (tw_inner_off, tw_outer_off) per r4 stage
kernel void ct_gs_r4_inverse_tg(
    device uint* data              [[buffer(0)]],
    device const uint* twiddles    [[buffer(1)]],
    device const uint* params      [[buffer(2)]],
    uint tg_id                     [[threadgroup_position_in_grid]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    threadgroup uint tile[CT_GS_R4_MAX_TILE];

    uint tile_log      = params[0];
    uint num_r4        = params[1];
    uint has_initial_r2 = params[2];
    uint tile_size     = 1u << tile_log;
    uint tile_off      = tg_id * tile_size;

    // Load tile
    for (uint i = tid; i < tile_size; i += tg_size) {
        tile[i] = data[tile_off + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint quarter_tile = tile_size >> 2;
    uint offset_idx = 3;

    // ── Initial radix-2 stage (layer 0, if odd number of layers) ───────
    if (has_initial_r2) {
        uint tw_r2_off = params[offset_idx];
        offset_idx++;
        uint half_tile = tile_size >> 1;
        // Layer 0: stride=1, full=2
        uint blocks_per_tile = half_tile;
        uint global_block_base = tg_id * blocks_per_tile;

        for (uint i = tid; i < half_tile; i += tg_size) {
            uint block = i;
            uint idx0 = block * 2;
            uint idx1 = idx0 + 1;

            uint tw = twiddles[tw_r2_off + global_block_base + block];
            uint a = tile[idx0];
            uint b = tile[idx1];
            // GS-DIF: add first, then twiddle
            tile[idx0] = m31_add(a, b);
            tile[idx1] = m31_mul(m31_sub(a, b), tw);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Radix-4 inverse stages (processing from low layers to high) ────
    // For inverse, we process layer pairs from low to high.
    // start_layer depends on whether we had an initial r2.
    uint start_layer = has_initial_r2 ? 1 : 0;

    for (uint s = 0; s < num_r4; s++) {
        uint k_inner = start_layer + 2 * s;       // inner (lower) layer
        uint k_outer = k_inner + 1;               // outer (higher) layer
        uint inner = 1u << k_inner;
        uint outer = 1u << k_outer;
        uint quad  = outer << 1;

        uint tw_inner_off = params[offset_idx];
        uint tw_outer_off = params[offset_idx + 1];
        offset_idx += 2;

        uint quads_per_tile = tile_size / quad;
        uint blocks_inner_per_tile = tile_size / outer;
        uint global_quad_base = tg_id * quads_per_tile;
        uint global_inner_base = tg_id * blocks_inner_per_tile;

        for (uint i = tid; i < quarter_tile; i += tg_size) {
            uint q = i / inner;
            uint j = i % inner;
            uint idx0 = q * quad + j;
            uint idx1 = idx0 + inner;
            uint idx2 = idx0 + outer;
            uint idx3 = idx2 + inner;

            uint w2_inv = twiddles[tw_outer_off + global_quad_base + q];
            uint w1_inv = twiddles[tw_inner_off + global_inner_base + 2 * q];
            uint w3_inv = twiddles[tw_inner_off + global_inner_base + 2 * q + 1];

            uint a0 = tile[idx0], a1 = tile[idx1];
            uint a2 = tile[idx2], a3 = tile[idx3];

            // Undo stage 2 (inner layer): GS-DIF on pairs (0,1) and (2,3)
            uint b0 = m31_add(a0, a1);
            uint b1 = m31_mul(m31_sub(a0, a1), w1_inv);
            uint b2 = m31_add(a2, a3);
            uint b3 = m31_mul(m31_sub(a2, a3), w3_inv);

            // Undo stage 1 (outer layer): GS-DIF on pairs (0,2) and (1,3)
            tile[idx0] = m31_add(b0, b2);
            tile[idx1] = m31_add(b1, b3);
            tile[idx2] = m31_mul(m31_sub(b0, b2), w2_inv);
            tile[idx3] = m31_mul(m31_sub(b1, b3), w2_inv);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write tile back
    for (uint i = tid; i < tile_size; i += tg_size) {
        data[tile_off + i] = tile[i];
    }
}

// ─── Inverse GS-DIF radix-4 butterfly (device memory, single paired stage) ──
//
// Undoes two forward CT-DIT layers (k_outer, k_inner) where k_outer = k_inner + 1.
// params[0] = outer (2^k_outer), params[1] = inner (2^k_inner), params[2] = n
kernel void ct_gs_r4_butterfly_device_inv(
    device uint* data              [[buffer(0)]],
    device const uint* tw_outer    [[buffer(1)]],
    device const uint* tw_inner    [[buffer(2)]],
    device const uint* params      [[buffer(3)]],
    uint tid                       [[thread_position_in_grid]]
) {
    uint outer = params[0];
    uint inner = params[1];
    uint n     = params[2];
    uint quad  = outer << 1;

    uint q = tid / inner;
    uint j = tid % inner;
    uint idx0 = q * quad + j;
    uint idx3 = idx0 + outer + inner;
    if (idx3 >= n) return;

    uint idx1 = idx0 + inner;
    uint idx2 = idx0 + outer;

    uint w2_inv = tw_outer[q];
    uint w1_inv = tw_inner[2 * q];
    uint w3_inv = tw_inner[2 * q + 1];

    uint a0 = data[idx0], a1 = data[idx1];
    uint a2 = data[idx2], a3 = data[idx3];

    // Undo stage 2 (inner layer): GS-DIF on pairs (0,1) and (2,3)
    uint b0 = m31_add(a0, a1);
    uint b1 = m31_mul(m31_sub(a0, a1), w1_inv);
    uint b2 = m31_add(a2, a3);
    uint b3 = m31_mul(m31_sub(a2, a3), w3_inv);

    // Undo stage 1 (outer layer): GS-DIF on pairs (0,2) and (1,3)
    data[idx0] = m31_add(b0, b2);
    data[idx1] = m31_add(b1, b3);
    data[idx2] = m31_mul(m31_sub(b0, b2), w2_inv);
    data[idx3] = m31_mul(m31_sub(b1, b3), w2_inv);
}

// ─── Inverse GS-DIF radix-2 fallback (device memory, for odd log_n) ─────────
//
// params[0] = stride, params[1] = n
kernel void ct_gs_r4_r2_device_inv(
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

/// Normalization: multiply every element by a scalar (n^-1 mod p).
/// params[0] = n (element count)
/// params[1] = scalar (the inverse of n in M31)
kernel void ct_gs_r4_normalize(
    device uint* data            [[buffer(0)]],
    device const uint* params    [[buffer(1)]],
    uint tid                     [[thread_position_in_grid]]
) {
    uint n      = params[0];
    uint scalar = params[1];
    if (tid >= n) return;
    data[tid] = m31_mul(data[tid], scalar);
}

#endif // NTT_CT_GS_R4_H
