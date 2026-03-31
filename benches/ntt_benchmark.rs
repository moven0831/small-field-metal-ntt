// NTT Algorithm Shootout — Metal over M31 (Circle NTT)
//
// Compares 4 GPU variants + CPU reference:
//   V1: CT-DIT radix-2 (naive baseline, all device memory)
//   V2: CT-DIT/GS-DIF radix-2 (threadgroup memory, in-place)
//   V3: Stockham radix-2 (out-of-place, ping-pong buffers)
//   V4: CT-DIT/GS-DIF radix-4 (half the barriers)
//
// Run: cargo bench --release
// Output: CSV to stdout, summary table to stderr

use small_field_metal_ntt::field::m31::M31;
use small_field_metal_ntt::field::Field;
use small_field_metal_ntt::gpu::MetalContext;
use small_field_metal_ntt::ntt::cpu_reference::CpuReferenceBackend;
use small_field_metal_ntt::ntt::metal_ct_dit_r2::MetalCtDitR2;
use small_field_metal_ntt::ntt::metal_ct_gs_r2::MetalCtGsR2;
use small_field_metal_ntt::ntt::metal_ct_gs_r4::MetalCtGsR4;
use small_field_metal_ntt::ntt::metal_stockham_r2::MetalStockhamR2;
use small_field_metal_ntt::ntt::NttBackend;
use std::path::PathBuf;
use std::time::Instant;

const WARMUP: usize = 5;
const ITERATIONS: usize = 20;
const SIZES: &[usize] = &[
    1 << 10,  // 1K
    1 << 12,  // 4K
    1 << 14,  // 16K
    1 << 16,  // 64K
    1 << 18,  // 256K
    1 << 20,  // 1M
];

fn shader_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders")
}

fn lcg_data(n: usize, seed: u64) -> Vec<M31> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            M31(((s >> 33) as u32) % M31::P)
        })
        .collect()
}

struct BenchResult {
    variant: String,
    size: usize,
    median_us: f64,
    min_us: f64,
    max_us: f64,
    throughput_melem_s: f64,
}

fn bench_variant<F: Field>(
    name: &str,
    backend: &dyn NttBackend<F>,
    data_template: &[F],
    size: usize,
) -> BenchResult {
    let mut times = Vec::with_capacity(WARMUP + ITERATIONS);

    for i in 0..(WARMUP + ITERATIONS) {
        let mut data = data_template.to_vec();
        let start = Instant::now();
        backend.forward_ntt(&mut data, &[]).unwrap();
        let elapsed = start.elapsed();
        if i >= WARMUP {
            times.push(elapsed.as_secs_f64() * 1_000_000.0); // microseconds
        }
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[times.len() / 2];
    let min = times[0];
    let max = times[times.len() - 1];
    let throughput = (size as f64) / (median / 1_000_000.0) / 1_000_000.0; // Melem/s

    BenchResult {
        variant: name.to_string(),
        size,
        median_us: median,
        min_us: min,
        max_us: max,
        throughput_melem_s: throughput,
    }
}

fn main() {
    let dir = shader_dir();

    // Print device info
    if let Ok(ctx) = MetalContext::new(&dir) {
        let info = ctx.device_info();
        eprintln!("GPU: {}", info);
        eprintln!();
    }

    // Initialize backends
    let cpu = CpuReferenceBackend;
    let v1 = MetalCtDitR2::new(&dir).expect("Failed to init V1");
    let v2 = MetalCtGsR2::new(&dir).expect("Failed to init V2");
    let v3 = MetalStockhamR2::new(&dir).expect("Failed to init V3");
    let v4 = MetalCtGsR4::new(&dir).expect("Failed to init V4");

    // CSV header
    println!("variant,size,median_us,min_us,max_us,throughput_melem_s");

    let mut all_results: Vec<BenchResult> = Vec::new();

    for &size in SIZES {
        let data = lcg_data(size, 12345 + size as u64);

        eprintln!("Benchmarking size 2^{} ({} elements)...", size.trailing_zeros(), size);

        // CPU reference
        let r = bench_variant("cpu-reference", &cpu, &data, size);
        println!("{},{},{:.1},{:.1},{:.1},{:.1}", r.variant, r.size, r.median_us, r.min_us, r.max_us, r.throughput_melem_s);
        all_results.push(r);

        // V1: CT-DIT r2 (naive)
        let r = bench_variant("v1-ct-dit-r2", &v1, &data, size);
        println!("{},{},{:.1},{:.1},{:.1},{:.1}", r.variant, r.size, r.median_us, r.min_us, r.max_us, r.throughput_melem_s);
        all_results.push(r);

        // V2: CT-GS r2 (threadgroup)
        let r = bench_variant("v2-ct-gs-r2", &v2, &data, size);
        println!("{},{},{:.1},{:.1},{:.1},{:.1}", r.variant, r.size, r.median_us, r.min_us, r.max_us, r.throughput_melem_s);
        all_results.push(r);

        // V3: Stockham r2 (out-of-place)
        let r = bench_variant("v3-stockham-r2", &v3, &data, size);
        println!("{},{},{:.1},{:.1},{:.1},{:.1}", r.variant, r.size, r.median_us, r.min_us, r.max_us, r.throughput_melem_s);
        all_results.push(r);

        // V4: CT-GS r4 (radix-4)
        let r = bench_variant("v4-ct-gs-r4", &v4, &data, size);
        println!("{},{},{:.1},{:.1},{:.1},{:.1}", r.variant, r.size, r.median_us, r.min_us, r.max_us, r.throughput_melem_s);
        all_results.push(r);
    }

    // Summary table to stderr
    eprintln!();
    eprintln!("╔══════════════════════════════════════════════════════════════════════════╗");
    eprintln!("║                    NTT Algorithm Shootout Results                        ║");
    eprintln!("╠══════════════╦════════╦════════╦════════╦════════╦════════╦══════════════╣");
    eprintln!("║ Variant      ║ 2^10   ║ 2^12   ║ 2^14   ║ 2^16   ║ 2^18   ║ 2^20       ║");
    eprintln!("║              ║ (us)   ║ (us)   ║ (us)   ║ (us)   ║ (us)   ║ (us)        ║");
    eprintln!("╠══════════════╬════════╬════════╬════════╬════════╬════════╬══════════════╣");

    let variants = ["cpu-reference", "v1-ct-dit-r2", "v2-ct-gs-r2", "v3-stockham-r2", "v4-ct-gs-r4"];
    let labels = ["CPU ref    ", "V1 CT-DIT  ", "V2 CT-GS r2", "V3 Stockham", "V4 CT-GS r4"];

    for (vi, variant) in variants.iter().enumerate() {
        let mut row = format!("║ {} ║", labels[vi]);
        for &size in SIZES {
            if let Some(r) = all_results.iter().find(|r| r.variant == *variant && r.size == size) {
                row.push_str(&format!(" {:>6.0} ║", r.median_us));
            } else {
                row.push_str("    N/A ║");
            }
        }
        eprintln!("{}", row);
    }

    eprintln!("╚══════════════╩════════╩════════╩════════╩════════╩════════╩══════════════╝");

    // Find winner at largest size
    let largest = SIZES.last().unwrap();
    let gpu_results: Vec<&BenchResult> = all_results
        .iter()
        .filter(|r| r.size == *largest && r.variant != "cpu-reference")
        .collect();
    if let Some(winner) = gpu_results.iter().min_by(|a, b| a.median_us.partial_cmp(&b.median_us).unwrap()) {
        eprintln!();
        eprintln!("Winner at 2^{}: {} ({:.0} us, {:.1} Melem/s)",
            largest.trailing_zeros(), winner.variant, winner.median_us, winner.throughput_melem_s);
    }

    eprintln!();
    eprintln!("Config: {} warmup + {} iterations per (variant, size) pair", WARMUP, ITERATIONS);
}
