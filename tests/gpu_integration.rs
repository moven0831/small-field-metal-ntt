//! Integration tests: verify GPU Metal compute matches CPU reference.

use small_field_metal_ntt::field::m31::M31;
use small_field_metal_ntt::field::Field;
use small_field_metal_ntt::gpu::MetalContext;
use std::path::PathBuf;

fn shader_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders")
}

fn skip_if_no_metal() -> Option<MetalContext> {
    match MetalContext::new(&shader_dir()) {
        Ok(ctx) => Some(ctx),
        Err(_) => {
            eprintln!("No Metal device — skipping GPU test");
            None
        }
    }
}

#[test]
fn test_gpu_m31_add() {
    let ctx = match skip_if_no_metal() {
        Some(c) => c,
        None => return,
    };

    let pipeline = ctx.make_pipeline("m31_add_kernel").unwrap();

    let n = 1024;
    let a_data: Vec<u32> = (0..n).map(|i| i as u32 * 7 + 3).collect();
    let b_data: Vec<u32> = (0..n).map(|i| i as u32 * 13 + 11).collect();

    let buf_a = ctx.buffer_from_slice(&a_data).unwrap();
    let buf_b = ctx.buffer_from_slice(&b_data).unwrap();
    let buf_out = ctx.alloc_buffer(n * 4).unwrap();

    let threadgroups = metal::MTLSize::new((n as u64 + 255) / 256, 1, 1);
    let threads_per = metal::MTLSize::new(256, 1, 1);

    ctx.dispatch_and_wait(
        &pipeline,
        &[&buf_a, &buf_b, &buf_out],
        threadgroups,
        threads_per,
    )
    .unwrap();

    let result = MetalContext::read_buffer(&buf_out, n);

    // Verify against CPU
    for i in 0..n {
        let expected = M31(a_data[i]).add(M31(b_data[i]));
        assert_eq!(
            result[i], expected.0,
            "GPU m31_add mismatch at index {}: GPU={}, CPU={}",
            i, result[i], expected.0
        );
    }
}

#[test]
fn test_gpu_m31_mul() {
    let ctx = match skip_if_no_metal() {
        Some(c) => c,
        None => return,
    };

    let pipeline = ctx.make_pipeline("m31_mul_kernel").unwrap();

    let n = 1024;
    let a_data: Vec<u32> = (0..n).map(|i| (i as u32 * 17 + 5) % M31::P).collect();
    let b_data: Vec<u32> = (0..n).map(|i| (i as u32 * 23 + 7) % M31::P).collect();

    let buf_a = ctx.buffer_from_slice(&a_data).unwrap();
    let buf_b = ctx.buffer_from_slice(&b_data).unwrap();
    let buf_out = ctx.alloc_buffer(n * 4).unwrap();

    let threadgroups = metal::MTLSize::new((n as u64 + 255) / 256, 1, 1);
    let threads_per = metal::MTLSize::new(256, 1, 1);

    ctx.dispatch_and_wait(
        &pipeline,
        &[&buf_a, &buf_b, &buf_out],
        threadgroups,
        threads_per,
    )
    .unwrap();

    let result = MetalContext::read_buffer(&buf_out, n);

    for i in 0..n {
        let expected = M31(a_data[i]).mul(M31(b_data[i]));
        assert_eq!(
            result[i], expected.0,
            "GPU m31_mul mismatch at index {}: GPU={}, CPU={}",
            i, result[i], expected.0
        );
    }
}

#[test]
fn test_gpu_m31_mul_edge_cases() {
    let ctx = match skip_if_no_metal() {
        Some(c) => c,
        None => return,
    };

    let pipeline = ctx.make_pipeline("m31_mul_kernel").unwrap();

    // Edge cases: 0, 1, p-1, large values
    let a_data: Vec<u32> = vec![0, 1, M31::P - 1, M31::P - 1, 1000000000, 42];
    let b_data: Vec<u32> = vec![42, 42, M31::P - 1, 1, 1000000000, 0];
    let n = a_data.len();

    let buf_a = ctx.buffer_from_slice(&a_data).unwrap();
    let buf_b = ctx.buffer_from_slice(&b_data).unwrap();
    let buf_out = ctx.alloc_buffer(n * 4).unwrap();

    let threadgroups = metal::MTLSize::new(1, 1, 1);
    let threads_per = metal::MTLSize::new(n as u64, 1, 1);

    ctx.dispatch_and_wait(
        &pipeline,
        &[&buf_a, &buf_b, &buf_out],
        threadgroups,
        threads_per,
    )
    .unwrap();

    let result = MetalContext::read_buffer(&buf_out, n);

    for i in 0..n {
        let expected = M31(a_data[i]).mul(M31(b_data[i]));
        assert_eq!(
            result[i], expected.0,
            "GPU m31_mul edge case mismatch at index {}: a={}, b={}, GPU={}, CPU={}",
            i, a_data[i], b_data[i], result[i], expected.0
        );
    }
}

#[test]
fn test_gpu_butterfly_single_stage() {
    let ctx = match skip_if_no_metal() {
        Some(c) => c,
        None => return,
    };

    let pipeline = ctx.make_pipeline("butterfly_forward_kernel").unwrap();

    // Test: 8 elements, stride 1 (first layer of an 8-point NTT)
    // 4 butterfly pairs: (0,1), (2,3), (4,5), (6,7)
    let n: usize = 8;
    let stride: u32 = 1;
    let mut data: Vec<u32> = vec![10, 20, 30, 40, 50, 60, 70, 80];
    let twiddles: Vec<u32> = vec![3, 5, 7, 11]; // 4 twiddles for 4 blocks

    let buf_data = ctx.buffer_from_slice(&data).unwrap();
    let buf_tw = ctx.buffer_from_slice(&twiddles).unwrap();
    let params: Vec<u32> = vec![stride, n as u32];
    let buf_params = ctx.buffer_from_slice(&params).unwrap();

    let num_butterflies = n / 2;
    let threadgroups = metal::MTLSize::new(1, 1, 1);
    let threads_per = metal::MTLSize::new(num_butterflies as u64, 1, 1);

    ctx.dispatch_and_wait(
        &pipeline,
        &[&buf_data, &buf_tw, &buf_params],
        threadgroups,
        threads_per,
    )
    .unwrap();

    let gpu_result = MetalContext::read_buffer(&buf_data, n);

    // Compute expected result on CPU
    for block in 0..(n / (stride as usize * 2)) {
        let tw = M31(twiddles[block]);
        for j in 0..stride as usize {
            let idx0 = block * (stride as usize * 2) + j;
            let idx1 = idx0 + stride as usize;
            let a = M31(data[idx0]);
            let b = M31(data[idx1]);
            let t = b.mul(tw);
            data[idx0] = a.add(t).0;
            data[idx1] = a.sub(t).0;
        }
    }

    assert_eq!(
        gpu_result, data,
        "GPU butterfly mismatch: GPU={:?}, CPU={:?}",
        gpu_result, data
    );
}
