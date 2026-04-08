//! Cache coherency verification tests for CPU-GPU shared memory (UMA).
//!
//! These tests verify that StorageModeShared MTLBuffers provide correct
//! cache coherency on Apple Silicon: CPU writes are visible to GPU reads,
//! and GPU writes are visible to CPU reads after waitUntilCompleted().

use small_field_metal_ntt::gpu::MetalContext;
use std::path::PathBuf;

fn shader_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders")
}

fn skip_if_no_uma_metal() -> Option<MetalContext> {
    match MetalContext::new(&shader_dir()) {
        Ok(ctx) => {
            if !ctx.device_info().has_unified_memory {
                eprintln!("Not a UMA device — skipping cache coherency test");
                return None;
            }
            Some(ctx)
        }
        Err(_) => {
            eprintln!("No Metal device — skipping GPU test");
            None
        }
    }
}

/// Core coherency test: CPU writes -> GPU XORs -> CPU reads back.
///
/// Allocates a 1 MB StorageModeShared buffer (256K u32 values),
/// fills it from the CPU side, dispatches a GPU kernel that XORs
/// each element, then reads back from the CPU and verifies.
#[test]
fn test_cpu_write_gpu_read_cpu_readback() {
    let ctx = match skip_if_no_uma_metal() {
        Some(c) => c,
        None => return,
    };

    let n: usize = 256 * 1024; // 256K elements = 1 MB
    let xor_val: u32 = 0x12345678;

    // Allocate shared buffer
    let data_buf = ctx.alloc_buffer(n * 4).unwrap();

    // CPU writes known pattern into the buffer
    unsafe {
        let ptr = data_buf.contents() as *mut u32;
        for i in 0..n {
            *ptr.add(i) = (i as u32).wrapping_mul(0x9E3779B9).wrapping_add(0xDEADBEEF);
        }
    }
    // Compiler fence: ensure CPU stores are not reordered past the GPU dispatch.
    // Metal's command submission provides the hardware fence on UMA, but the Rust
    // compiler could theoretically sink raw pointer stores past safe function calls.
    std::sync::atomic::fence(std::sync::atomic::Ordering::Release);

    // Prepare params buffer: [n, xor_val]
    let params = ctx.buffer_from_slice(&[n as u32, xor_val]).unwrap();

    // GPU kernel XORs each element
    let pipeline = ctx.make_pipeline("cache_coherency_xor").unwrap();
    let threads_per = 256u64;
    let threadgroups = metal::MTLSize::new((n as u64 + threads_per - 1) / threads_per, 1, 1);
    let threads_per_group = metal::MTLSize::new(threads_per, 1, 1);

    ctx.dispatch_and_wait(
        &pipeline,
        &[&data_buf, &params],
        threadgroups,
        threads_per_group,
    )
    .unwrap();

    // CPU reads back and verifies
    let result = MetalContext::read_buffer(&data_buf, n);
    for i in 0..n {
        let original = (i as u32).wrapping_mul(0x9E3779B9).wrapping_add(0xDEADBEEF);
        let expected = original ^ xor_val;
        assert_eq!(
            result[i], expected,
            "Coherency mismatch at index {}: got {:#010X}, expected {:#010X}",
            i, result[i], expected
        );
    }
}

/// Reverse direction: GPU writes -> CPU reads.
///
/// The GPU kernel writes a deterministic pattern into the buffer,
/// then the CPU reads it back after waitUntilCompleted() and verifies.
#[test]
fn test_gpu_write_cpu_read() {
    let ctx = match skip_if_no_uma_metal() {
        Some(c) => c,
        None => return,
    };

    let n: usize = 256 * 1024;

    let data_buf = ctx.alloc_buffer(n * 4).unwrap();
    let params = ctx.buffer_from_slice(&[n as u32]).unwrap();

    let pipeline = ctx.make_pipeline("cache_coherency_write").unwrap();
    let threads_per = 256u64;
    let threadgroups = metal::MTLSize::new((n as u64 + threads_per - 1) / threads_per, 1, 1);
    let threads_per_group = metal::MTLSize::new(threads_per, 1, 1);

    ctx.dispatch_and_wait(
        &pipeline,
        &[&data_buf, &params],
        threadgroups,
        threads_per_group,
    )
    .unwrap();

    // CPU reads back GPU-written data
    let result = MetalContext::read_buffer(&data_buf, n);
    for i in 0..n {
        let expected = (i as u32).wrapping_mul(0x9E3779B9).wrapping_add(0xCAFEBABE);
        assert_eq!(
            result[i], expected,
            "GPU->CPU coherency mismatch at index {}: got {:#010X}, expected {:#010X}",
            i, result[i], expected
        );
    }
}

/// Tiny buffer coherency test (4 elements).
///
/// Exercises the smallest practical case to catch any alignment or
/// minimum-size issues with cache coherency.
#[test]
fn test_small_buffer_coherency() {
    let ctx = match skip_if_no_uma_metal() {
        Some(c) => c,
        None => return,
    };

    let n: usize = 4;
    let xor_val: u32 = 0xFF;
    let input: [u32; 4] = [1, 2, 3, 4];

    // Allocate and write from CPU
    let data_buf = ctx.buffer_from_slice(&input).unwrap();
    let params = ctx.buffer_from_slice(&[n as u32, xor_val]).unwrap();

    let pipeline = ctx.make_pipeline("cache_coherency_xor").unwrap();
    let threadgroups = metal::MTLSize::new(1, 1, 1);
    let threads_per_group = metal::MTLSize::new(n as u64, 1, 1);

    ctx.dispatch_and_wait(
        &pipeline,
        &[&data_buf, &params],
        threadgroups,
        threads_per_group,
    )
    .unwrap();

    let result = MetalContext::read_buffer(&data_buf, n);
    let expected: Vec<u32> = input.iter().map(|v| v ^ xor_val).collect();
    assert_eq!(
        result, expected,
        "Small buffer coherency mismatch: got {:?}, expected {:?}",
        result, expected
    );
}
