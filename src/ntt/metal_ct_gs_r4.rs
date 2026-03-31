//! Variant 4: CT-DIT/GS-DIF radix-4 NTT on Metal GPU — in-place with threadgroup memory.
//!
//! Each radix-4 butterfly replaces two radix-2 stages, halving the barrier count.
//! When log_n is odd, the final stage falls back to radix-2.
//!
//! For a 2^20 NTT:
//! - Threadgroup phase: 6 radix-4 + 1 radix-2 = 7 barriers (vs V2's 13)
//! - Device phase: 3 radix-4 + 1 radix-2 = 4 dispatches (vs V2's 7)
//! - Total: 5 dispatches (vs V2's 8)
//!
//! Forward NTT only (benchmark variant).

use crate::field::circle::Coset;
use crate::field::m31::M31;
use crate::field::Field;
use crate::gpu::MetalContext;
use crate::ntt::{NttBackend, NttError};
use metal::*;
use std::path::Path;

const MAX_TILE_LOG: usize = 13;

pub struct MetalCtGsR4 {
    ctx: MetalContext,
    forward_tg_pipeline: ComputePipelineState,
    r4_device_pipeline: ComputePipelineState,
    r2_device_pipeline: ComputePipelineState,
}

impl MetalCtGsR4 {
    pub fn new(shader_dir: &Path) -> Result<Self, NttError> {
        let ctx = MetalContext::new(shader_dir)?;
        let forward_tg = ctx.make_pipeline("ct_gs_r4_forward_tg")?;
        let r4_dev = ctx.make_pipeline("ct_gs_r4_butterfly_device")?;
        let r2_dev = ctx.make_pipeline("ct_gs_r4_butterfly_device_r2")?;

        Ok(MetalCtGsR4 {
            ctx,
            forward_tg_pipeline: forward_tg,
            r4_device_pipeline: r4_dev,
            r2_device_pipeline: r2_dev,
        })
    }

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

        let buf_data = self.ctx.buffer_from_slice(input)?;
        let mut total_ns: u64 = 0;

        let tile_log = log_n.min(MAX_TILE_LOG);

        // ── Device-memory phase: layers (log_n-1) down to tile_log ─────
        let num_device_layers = log_n - tile_log;
        if num_device_layers > 0 {
            total_ns += self.dispatch_device_phase(
                &buf_data,
                &twiddles,
                n,
                log_n,
                tile_log,
            )?;
        }

        // ── Threadgroup phase: layers (tile_log-1) down to 0 ───────────
        if tile_log > 0 {
            total_ns += self.dispatch_threadgroup_phase(
                &buf_data,
                &twiddles,
                n,
                tile_log,
            )?;
        }

        let result = MetalContext::read_buffer(&buf_data, n);
        Ok((result, total_ns))
    }

    fn dispatch_device_phase(
        &self,
        buf_data: &Buffer,
        twiddles: &[Vec<M31>],
        n: usize,
        log_n: usize,
        tile_log: usize,
    ) -> Result<u64, NttError> {
        let mut total_ns: u64 = 0;
        let num_device_layers = log_n - tile_log;
        let num_r4 = num_device_layers / 2;
        let has_r2 = num_device_layers % 2 == 1;

        let max_tpg_r4 =
            MetalContext::max_threads_per_threadgroup(&self.r4_device_pipeline) as u64;
        let tpg_r4 = max_tpg_r4.min(256);
        let max_tpg_r2 =
            MetalContext::max_threads_per_threadgroup(&self.r2_device_pipeline) as u64;
        let tpg_r2 = max_tpg_r2.min(256);

        let mut current_layer = log_n - 1;

        // Radix-4 paired dispatches
        for _ in 0..num_r4 {
            let k = current_layer;
            let outer = 1usize << k;
            let inner = 1usize << (k - 1);
            let num_butterflies = (n / 4) as u64;

            let tw_outer: Vec<u32> = twiddles[k].iter().map(|m| m.0).collect();
            let tw_inner: Vec<u32> = twiddles[k - 1].iter().map(|m| m.0).collect();
            let buf_tw_o = self.ctx.buffer_from_slice(&tw_outer)?;
            let buf_tw_i = self.ctx.buffer_from_slice(&tw_inner)?;
            let params: Vec<u32> = vec![outer as u32, inner as u32, n as u32];
            let buf_p = self.ctx.buffer_from_slice(&params)?;

            let tg = MTLSize::new(
                (num_butterflies + tpg_r4 - 1) / tpg_r4,
                1,
                1,
            );
            let tpg = MTLSize::new(tpg_r4.min(num_butterflies), 1, 1);

            let ns = self.ctx.dispatch_and_wait(
                &self.r4_device_pipeline,
                &[buf_data, &buf_tw_o, &buf_tw_i, &buf_p],
                tg,
                tpg,
            )?;
            total_ns += ns;
            current_layer -= 2;
        }

        // Final radix-2 if odd
        if has_r2 {
            let k = current_layer;
            let stride = 1usize << k;
            let num_butterflies = (n / 2) as u64;

            let tw_data: Vec<u32> = twiddles[k].iter().map(|m| m.0).collect();
            let buf_tw = self.ctx.buffer_from_slice(&tw_data)?;
            let params: Vec<u32> = vec![stride as u32, n as u32];
            let buf_p = self.ctx.buffer_from_slice(&params)?;

            let tg = MTLSize::new(
                (num_butterflies + tpg_r2 - 1) / tpg_r2,
                1,
                1,
            );
            let tpg = MTLSize::new(tpg_r2.min(num_butterflies), 1, 1);

            let ns = self.ctx.dispatch_and_wait(
                &self.r2_device_pipeline,
                &[buf_data, &buf_tw, &buf_p],
                tg,
                tpg,
            )?;
            total_ns += ns;
        }

        Ok(total_ns)
    }

    fn dispatch_threadgroup_phase(
        &self,
        buf_data: &Buffer,
        twiddles: &[Vec<M31>],
        n: usize,
        tile_log: usize,
    ) -> Result<u64, NttError> {
        let num_tg_layers = tile_log;
        let num_r4 = num_tg_layers / 2;
        let has_final_r2 = num_tg_layers % 2 == 1;
        let start_layer = tile_log - 1;

        // Flatten twiddles for all threadgroup stages
        let mut flat_tw = Vec::new();
        let mut tw_offsets: Vec<u32> = Vec::new();

        // Radix-4 stages: pairs (start_layer, start_layer-1), (start_layer-2, start_layer-3), ...
        for s in 0..num_r4 {
            let k = start_layer - 2 * s;
            // Outer twiddles (layer k)
            tw_offsets.push(flat_tw.len() as u32);
            flat_tw.extend(twiddles[k].iter().map(|m| m.0));
            // Inner twiddles (layer k-1)
            tw_offsets.push(flat_tw.len() as u32);
            flat_tw.extend(twiddles[k - 1].iter().map(|m| m.0));
        }

        // Final radix-2 stage (layer 0 if odd)
        if has_final_r2 {
            tw_offsets.push(flat_tw.len() as u32);
            flat_tw.extend(twiddles[0].iter().map(|m| m.0));
        }

        let buf_tw = self.ctx.buffer_from_slice(&flat_tw)?;

        // params: [tile_log, num_r4, has_final_r2, offsets...]
        let mut params: Vec<u32> = vec![
            tile_log as u32,
            num_r4 as u32,
            has_final_r2 as u32,
        ];
        params.extend(tw_offsets);
        let buf_p = self.ctx.buffer_from_slice(&params)?;

        let tile_size = 1usize << tile_log;
        let num_tiles = n / tile_size;
        let max_tg_threads =
            MetalContext::max_threads_per_threadgroup(&self.forward_tg_pipeline) as u64;
        let threads = max_tg_threads.min(tile_size as u64 / 4).max(1);

        let tg = MTLSize::new(num_tiles as u64, 1, 1);
        let tpg = MTLSize::new(threads, 1, 1);

        self.ctx.dispatch_and_wait(
            &self.forward_tg_pipeline,
            &[buf_data, &buf_tw, &buf_p],
            tg,
            tpg,
        )
    }
}

impl NttBackend<M31> for MetalCtGsR4 {
    fn name(&self) -> &str {
        "metal-ct-gs-r4"
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
            "Inverse NTT not implemented for radix-4 — forward-only benchmark variant".into(),
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

// ─── Twiddle generation (same as other variants) ────────────────────────

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

    fn skip_if_no_metal() -> Option<MetalCtGsR4> {
        match MetalCtGsR4::new(&shader_dir()) {
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
    fn test_forward_matches_cpu_size8() {
        // size 8 = log_n 3 (odd), tests radix-4 + final radix-2
        let gpu = match skip_if_no_metal() {
            Some(g) => g,
            None => return,
        };
        let cpu = CpuReferenceBackend;
        let n = 8;
        let original: Vec<M31> = (0..n).map(|i| M31(i as u32 * 7 + 3)).collect();
        let mut cpu_data = original.clone();
        cpu.forward_ntt(&mut cpu_data, &[]).unwrap();
        let mut gpu_data = original.clone();
        gpu.forward_ntt(&mut gpu_data, &[]).unwrap();
        assert_eq!(gpu_data, cpu_data, "Forward mismatch at size {}", n);
    }

    #[test]
    fn test_forward_matches_cpu_size16() {
        // size 16 = log_n 4 (even), pure radix-4
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
    fn test_forward_matches_cpu_size32() {
        // size 32 = log_n 5 (odd), tests r4+r4+r2
        let gpu = match skip_if_no_metal() {
            Some(g) => g,
            None => return,
        };
        let cpu = CpuReferenceBackend;
        let n = 32;
        let original: Vec<M31> = (0..n).map(|i| M31((i as u32 * 13 + 7) % M31::P)).collect();
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
    fn test_forward_matches_cpu_size2048() {
        // size 2048 = log_n 11 (odd), tests odd threadgroup layers
        let gpu = match skip_if_no_metal() {
            Some(g) => g,
            None => return,
        };
        let cpu = CpuReferenceBackend;
        let n = 2048;
        let original: Vec<M31> = lcg_data(n, 55555);
        let mut cpu_data = original.clone();
        cpu.forward_ntt(&mut cpu_data, &[]).unwrap();
        let mut gpu_data = original.clone();
        gpu.forward_ntt(&mut gpu_data, &[]).unwrap();
        assert_eq!(gpu_data, cpu_data, "Forward mismatch at size {}", n);
    }

    #[test]
    fn test_forward_matches_cpu_size4096() {
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
        // Device phase + threadgroup phase
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
    fn test_size_one_identity() {
        let gpu = match skip_if_no_metal() {
            Some(g) => g,
            None => return,
        };
        let mut data = vec![M31(42)];
        gpu.forward_ntt(&mut data, &[]).unwrap();
        assert_eq!(data[0], M31(42));
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

    #[test]
    fn test_inverse_returns_err() {
        let gpu = match skip_if_no_metal() {
            Some(g) => g,
            None => return,
        };
        let mut data = vec![M31(1), M31(2), M31(3), M31(4)];
        assert!(gpu.inverse_ntt(&mut data, &[]).is_err());
    }

    #[test]
    fn test_pointwise_mul() {
        let gpu = match skip_if_no_metal() {
            Some(g) => g,
            None => return,
        };
        let a = vec![M31(2), M31(3), M31(4), M31(5)];
        let b = vec![M31(10), M31(20), M31(30), M31(40)];
        let mut out = vec![M31(0); 4];
        gpu.pointwise_mul(&a, &b, &mut out).unwrap();
        assert_eq!(out, vec![M31(20), M31(60), M31(120), M31(200)]);
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
