// BabyBear NTT shared definitions for Metal GPU.
//
// Uses BabyBear Montgomery field arithmetic (not Circle NTT).
// Key difference from M31 Circle NTT twiddle layout:
//   Standard NTT: twiddle at position j within stride, same for all blocks.
//   Circle NTT:   twiddle per block (different for each block).
//
// Phase 1 (threadgroup): Small-stride layers in on-chip memory.
// Phase 2 (device memory): Large-stride layers, one dispatch per layer.

#ifndef BB_COMMON_H
#define BB_COMMON_H

#include "../fields/babybear_field.metal"

#define BB_R2_MAX_TILE 8192

#endif // BB_COMMON_H
