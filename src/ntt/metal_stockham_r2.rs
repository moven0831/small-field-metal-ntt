//! Variant 3: Stockham radix-2 out-of-place NTT on Metal GPU.
//!
//! Key difference from Variants 1 & 2: reads from buffer A, writes to buffer B,
//! then swaps. This tests whether the out-of-place ping-pong memory pattern
//! helps or hurts on Apple Silicon's Unified Memory Architecture.
//!
//! Two-phase execution:
//! - Device-memory phase: Large-stride layers, one dispatch per layer.
//!   Each dispatch reads from one device buffer and writes to the other.
//! - Threadgroup phase: Small-stride layers processed on-chip with two
//!   threadgroup arrays (2 x 4096 = 32 KB) ping-ponging between stages.
//!
//! Uses the same CT-DIT butterfly and twiddle factors as Variants 1 & 2.
//! Output is bit-reversed (same as CPU reference forward NTT).
//!
//! Memory overhead: 2x (two full-size device buffers).
//! Threadgroup tile: 4096 elements (half of Variant 2's 8192, due to needing
//! two arrays). Handles 12 stages on-chip vs Variant 2's 13.

use crate::field::circle::Coset;
use crate::field::m31::M31;
use crate::field::Field;
use crate::gpu::MetalContext;
use crate::ntt::{NttBackend, NttError};
use metal::*;
use std::path::Path;

/// Maximum elements per threadgroup tile: 4096 (two arrays of 16 KB each = 32 KB total).
const MAX_TILE_LOG: usize = 12;

pub struct MetalStockhamR2 {
    ctx: MetalContext,
    forward_tg_pipeline: ComputePipelineState,
    forward_device_pipeline: ComputePipelineState,
}

impl MetalStockhamR2 {
    pub fn new(shader_dir: &Path) -> Result<Self, NttError> {
        let ctx = MetalContext::new(shader_dir)?;
        let forward_tg = ctx.make_pipeline("stockham_r2_forward_tg")?;
        let forward_dev = ctx.make_pipeline("stockham_r2_butterfly_device")?;

        Ok(MetalStockhamR2 {
            ctx,
            forward_tg_pipeline: forward_tg,
            forward_device_pipeline: forward_dev,
        })
    }

    /// Run forward Circle NTT (CT-DIT, out-of-place) on GPU.
    /// Returns (result_data, total_gpu_time_ns).
    pub fn forward_ntt_gpu(
        &self,
        input: &[u32],
        log_n: usize,
    ) -> Result<(Vec<u32>, u64), NttError> {
        let n = input.len();
        if n != (1 << log_n) {
            return Err(NttError::InvalidSize(n));
        }
        if log_n > 30 {
            return Err(NttError::InvalidSize(n));
        }
        if log_n == 0 {
            return Ok((input.to_vec(), 0));
        }

        let coset = Coset::odds(log_n as u32);
        let twiddles = generate_twiddles(&coset);

        // Allocate two device buffers for ping-pong
        let buf_a = self.ctx.buffer_from_slice(input)?;
        let buf_b = self.ctx.alloc_buffer(n * std::mem::size_of::<u32>())?;
        let mut total_ns: u64 = 0;

        let tile_log = log_n.min(MAX_TILE_LOG);

        // Track which buffer is current input vs output
        // read_from_a: true means read from buf_a, write to buf_b
        let mut read_from_a = true;

        // ── Phase 1: device-memory stages (large strides) ──────────────
        // Forward: layers from (log_n-1) down to tile_log
        let max_tpg =
            MetalContext::max_threads_per_threadgroup(&self.forward_device_pipeline) as u64;
        let tpg_width = max_tpg.min(256);

        for layer in (tile_log..log_n).rev() {
            let stride = 1usize << layer;
            let num_butterflies = (n / 2) as u64;

            let tw_data: Vec<u32> = twiddles[layer].iter().map(|m| m.0).collect();
            let buf_tw = self.ctx.buffer_from_slice(&tw_data)?;
            let params: Vec<u32> = vec![stride as u32, n as u32];
            let buf_p = self.ctx.buffer_from_slice(&params)?;

            let tg = MTLSize::new(
                (num_butterflies + tpg_width - 1) / tpg_width,
                1,
                1,
            );
            let tpg = MTLSize::new(tpg_width.min(num_butterflies), 1, 1);

            let (cur_in, cur_out) = if read_from_a {
                (&buf_a, &buf_b)
            } else {
                (&buf_b, &buf_a)
            };

            let ns = self.ctx.dispatch_and_wait(
                &self.forward_device_pipeline,
                &[cur_in, cur_out, &buf_tw, &buf_p],
                tg,
                tpg,
            )?;
            total_ns += ns;
            read_from_a = !read_from_a;
        }

        // ── Phase 2: threadgroup stages (small strides) ────────────────
        // Forward: layers from (tile_log-1) down to 0
        if tile_log > 0 {
            let num_tg_layers = tile_log;
            let start_layer = tile_log - 1;

            // Flatten twiddles for all threadgroup layers
            let mut flat_tw = Vec::new();
            let mut tw_offsets = Vec::new();
            for li in 0..num_tg_layers {
                let layer = start_layer - li;
                tw_offsets.push(flat_tw.len() as u32);
                flat_tw.extend(twiddles[layer].iter().map(|m| m.0));
            }
            let buf_tw = self.ctx.buffer_from_slice(&flat_tw)?;

            // params: [tile_log, num_layers, start_layer, offsets...]
            let mut params: Vec<u32> = vec![
                tile_log as u32,
                num_tg_layers as u32,
                start_layer as u32,
            ];
            params.extend(tw_offsets);
            let buf_p = self.ctx.buffer_from_slice(&params)?;

            let tile_size = 1usize << tile_log;
            let num_tiles = n / tile_size;
            let max_tg_threads =
                MetalContext::max_threads_per_threadgroup(&self.forward_tg_pipeline) as u64;
            let threads = max_tg_threads.min(tile_size as u64 / 2).max(1);

            let tg = MTLSize::new(num_tiles as u64, 1, 1);
            let tpg = MTLSize::new(threads, 1, 1);

            let (cur_in, cur_out) = if read_from_a {
                (&buf_a, &buf_b)
            } else {
                (&buf_b, &buf_a)
            };

            let ns = self.ctx.dispatch_and_wait(
                &self.forward_tg_pipeline,
                &[cur_in, cur_out, &buf_tw, &buf_p],
                tg,
                tpg,
            )?;
            total_ns += ns;
            read_from_a = !read_from_a;
        }

        // Result is in whichever buffer was last written to
        let result_buf = if read_from_a { &buf_a } else { &buf_b };
        let result = MetalContext::read_buffer(result_buf, n);
        Ok((result, total_ns))
    }
}

impl NttBackend<M31> for MetalStockhamR2 {
    fn name(&self) -> &str {
        "metal-stockham-r2"
    }

    fn forward_ntt(&self, data: &mut [M31], _twiddles: &[M31]) -> Result<(), NttError> {
        let n = data.len();
        if n == 0 || (n & (n - 1)) != 0 {
            return Err(NttError::InvalidSize(n));
        }
        if n == 1 {
            return Ok(());
        }
        let log_n = n.trailing_zeros() as usize;

        let input: Vec<u32> = data.iter().map(|m| m.0).collect();
        let (result, _) = self.forward_ntt_gpu(&input, log_n)?;
        for (i, val) in result.iter().enumerate() {
            data[i] = M31(*val);
        }
        Ok(())
    }

    fn inverse_ntt(&self, _data: &mut [M31], _twiddles: &[M31]) -> Result<(), NttError> {
        Err(NttError::GpuExecutionError(
            "Inverse NTT not implemented for Stockham — forward-only benchmark variant".into(),
        ))
    }

    fn pointwise_mul(&self, a: &[M31], b: &[M31], out: &mut [M31]) -> Result<(), NttError> {
        if a.len() != b.len() || a.len() != out.len() {
            return Err(NttError::InvalidSize(a.len()));
        }
        for i in 0..a.len() {
            out[i] = a[i].mul(b[i]);
        }
        Ok(())
    }
}

// ─── Twiddle generation (same as Variants 1 & 2) ────────────────────────

fn generate_twiddles(coset: &Coset) -> Vec<Vec<M31>> {
    let log_n = coset.log_size as usize;
    let mut result = Vec::with_capacity(log_n);
    let mut current = coset.clone();

    for layer_idx in 0..log_n {
        let half_size = current.size() / 2;
        let is_last = layer_idx == log_n - 1;
        let layer_tw: Vec<M31> = (0..half_size)
            .map(|i| {
                let p = current.at(bit_reverse_idx(i, current.log_size - 1));
                if is_last {
                    p.y
                } else {
                    p.x
                }
            })
            .collect();
        result.push(layer_tw);
        current = current.double();
    }
    result
}

fn bit_reverse_idx(index: usize, log_size: u32) -> usize {
    let mut val = index as u32;
    let mut result = 0u32;
    for _ in 0..log_size {
        result = (result << 1) | (val & 1);
        val >>= 1;
    }
    result as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ntt::cpu_reference::CpuReferenceBackend;
    use crate::ntt::NttBackend;
    use std::path::PathBuf;

    fn shader_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders")
    }

    fn skip_if_no_metal() -> Option<MetalStockhamR2> {
        match MetalStockhamR2::new(&shader_dir()) {
            Ok(g) => Some(g),
            Err(NttError::DeviceNotFound) => {
                eprintln!("No Metal device — skipping");
                None
            }
            Err(e) => panic!("Failed: {}", e),
        }
    }

    // ── Forward NTT: GPU vs CPU reference ────────────────────────────────

    #[test]
    fn test_forward_matches_cpu_size4() {
        let gpu = match skip_if_no_metal() {
            Some(g) => g,
            None => return,
        };
        let cpu = CpuReferenceBackend;
        let original = vec![M31(1), M31(2), M31(3), M31(4)];
        let mut cpu_data = original.clone();
        cpu.forward_ntt(&mut cpu_data, &[]).unwrap();
        let mut gpu_data = original.clone();
        gpu.forward_ntt(&mut gpu_data, &[]).unwrap();
        assert_eq!(gpu_data, cpu_data, "Forward mismatch at size 4");
    }

    #[test]
    fn test_forward_matches_cpu_size16() {
        let gpu = match skip_if_no_metal() {
            Some(g) => g,
            None => return,
        };
        let cpu = CpuReferenceBackend;
        let n = 16;
        let original: Vec<M31> = (0..n).map(|i| M31(i as u32 * 7 + 3)).collect();
        let mut cpu_data = original.clone();
        cpu.forward_ntt(&mut cpu_data, &[]).unwrap();
        let mut gpu_data = original.clone();
        gpu.forward_ntt(&mut gpu_data, &[]).unwrap();
        assert_eq!(gpu_data, cpu_data, "Forward mismatch at size {}", n);
    }

    #[test]
    fn test_forward_matches_cpu_size256() {
        let gpu = match skip_if_no_metal() {
            Some(g) => g,
            None => return,
        };
        let cpu = CpuReferenceBackend;
        let n = 256;
        let original: Vec<M31> = (0..n).map(|i| M31((i as u32 * 13 + 7) % M31::P)).collect();
        let mut cpu_data = original.clone();
        cpu.forward_ntt(&mut cpu_data, &[]).unwrap();
        let mut gpu_data = original.clone();
        gpu.forward_ntt(&mut gpu_data, &[]).unwrap();
        assert_eq!(gpu_data, cpu_data, "Forward mismatch at size {}", n);
    }

    #[test]
    fn test_forward_matches_cpu_size1024() {
        let gpu = match skip_if_no_metal() {
            Some(g) => g,
            None => return,
        };
        let cpu = CpuReferenceBackend;
        let n = 1024;
        let original: Vec<M31> = (0..n).map(|i| M31((i as u32 * 17 + 11) % M31::P)).collect();
        let mut cpu_data = original.clone();
        cpu.forward_ntt(&mut cpu_data, &[]).unwrap();
        let mut gpu_data = original.clone();
        gpu.forward_ntt(&mut gpu_data, &[]).unwrap();
        assert_eq!(gpu_data, cpu_data, "Forward mismatch at size {}", n);
    }

    #[test]
    fn test_forward_matches_cpu_size4096() {
        // Exactly one threadgroup tile
        let gpu = match skip_if_no_metal() {
            Some(g) => g,
            None => return,
        };
        let cpu = CpuReferenceBackend;
        let n = 4096;
        let original: Vec<M31> = lcg_data(n, 98765);
        let mut cpu_data = original.clone();
        cpu.forward_ntt(&mut cpu_data, &[]).unwrap();
        let mut gpu_data = original.clone();
        gpu.forward_ntt(&mut gpu_data, &[]).unwrap();
        assert_eq!(gpu_data, cpu_data, "Forward mismatch at size {}", n);
    }

    #[test]
    fn test_forward_matches_cpu_size8192() {
        // Two tiles (tests device phase + threadgroup phase)
        let gpu = match skip_if_no_metal() {
            Some(g) => g,
            None => return,
        };
        let cpu = CpuReferenceBackend;
        let n = 8192;
        let original: Vec<M31> = lcg_data(n, 11111);
        let mut cpu_data = original.clone();
        cpu.forward_ntt(&mut cpu_data, &[]).unwrap();
        let mut gpu_data = original.clone();
        gpu.forward_ntt(&mut gpu_data, &[]).unwrap();
        assert_eq!(gpu_data, cpu_data, "Forward mismatch at size {}", n);
    }

    #[test]
    fn test_forward_matches_cpu_size16384() {
        // Multiple device-phase dispatches + threadgroup
        let gpu = match skip_if_no_metal() {
            Some(g) => g,
            None => return,
        };
        let cpu = CpuReferenceBackend;
        let n = 16384;
        let original: Vec<M31> = lcg_data(n, 22222);
        let mut cpu_data = original.clone();
        cpu.forward_ntt(&mut cpu_data, &[]).unwrap();
        let mut gpu_data = original.clone();
        gpu.forward_ntt(&mut gpu_data, &[]).unwrap();
        assert_eq!(gpu_data, cpu_data, "Forward mismatch at size {}", n);
    }

    // ── Edge cases ───────────────────────────────────────────────────────

    #[test]
    fn test_all_zeros() {
        let gpu = match skip_if_no_metal() {
            Some(g) => g,
            None => return,
        };
        let mut data = vec![M31(0); 64];
        gpu.forward_ntt(&mut data, &[]).unwrap();
        assert!(data.iter().all(|&x| x == M31(0)));
    }

    #[test]
    fn test_size2() {
        let gpu = match skip_if_no_metal() {
            Some(g) => g,
            None => return,
        };
        let cpu = CpuReferenceBackend;
        let original = vec![M31(100), M31(200)];
        let mut cpu_data = original.clone();
        cpu.forward_ntt(&mut cpu_data, &[]).unwrap();
        let mut gpu_data = original.clone();
        gpu.forward_ntt(&mut gpu_data, &[]).unwrap();
        assert_eq!(gpu_data, cpu_data, "Forward mismatch at size 2");
    }

    #[test]
    fn test_invalid_size_not_power_of_two() {
        let gpu = match skip_if_no_metal() {
            Some(g) => g,
            None => return,
        };
        let mut data = vec![M31(1); 3];
        assert!(gpu.forward_ntt(&mut data, &[]).is_err());
    }

    #[test]
    fn test_invalid_size_zero() {
        let gpu = match skip_if_no_metal() {
            Some(g) => g,
            None => return,
        };
        let mut data: Vec<M31> = vec![];
        assert!(gpu.forward_ntt(&mut data, &[]).is_err());
    }

    // ── Helpers ──────────────────────────────────────────────────────────

    fn lcg_data(n: usize, seed: u64) -> Vec<M31> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                M31(((s >> 33) as u32) % M31::P)
            })
            .collect()
    }
}
