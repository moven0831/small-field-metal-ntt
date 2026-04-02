// NTT Algorithm Shootout — Metal over M31 (Circle NTT)
//
// Two benchmark modes:
//   1. Algorithm shootout: 4 GPU variants + CPU reference (default)
//   2. Cooperative split-point sweep: CPU-GPU partition at every layer (--cooperative)
//
// Run:
//   cargo bench                    # algorithm shootout
//   cargo bench -- --cooperative   # split-point sweep
//
// Output: CSV to stdout, summary table to stderr

use small_field_metal_ntt::field::m31::M31;
use small_field_metal_ntt::field::Field;
use small_field_metal_ntt::gpu::MetalContext;
use small_field_metal_ntt::ntt::cooperative::{self, CooperativeNttContext};
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

// Cooperative sweep: more iterations for statistical significance.
// Each sweep tests all split values (0..log_n), so total calls = sizes * splits * (warmup+iters).
const COOP_WARMUP: usize = 5;
const COOP_ITERATIONS: usize = 20;
const COOP_SIZES: &[usize] = &[
    1 << 10,  // 1K
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

fn lcg_data_u32(n: usize, seed: u64) -> Vec<u32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) as u32) % M31::P
        })
        .collect()
}

/// Benchmark one split_layer value for the cooperative NTT.
struct CoopBenchResult {
    size: usize,
    log_n: usize,
    split_layer: usize,
    median_total_us: f64,
    median_cpu_us: f64,
    median_gpu_us: f64,
    min_total_us: f64,
    max_total_us: f64,
}

fn bench_cooperative(
    coop: &CooperativeNttContext,
    input: &[u32],
    log_n: usize,
    split_layer: usize,
) -> CoopBenchResult {
    let n = input.len();
    let mut totals = Vec::with_capacity(COOP_WARMUP + COOP_ITERATIONS);
    let mut cpus = Vec::with_capacity(COOP_WARMUP + COOP_ITERATIONS);
    let mut gpus = Vec::with_capacity(COOP_WARMUP + COOP_ITERATIONS);

    for i in 0..(COOP_WARMUP + COOP_ITERATIONS) {
        let result = cooperative::cooperative_forward_ntt(coop, input, log_n, split_layer)
            .expect("Cooperative NTT failed");
        if i >= COOP_WARMUP {
            totals.push(result.total_time_us);
            cpus.push(result.cpu_time_us);
            gpus.push(result.gpu_time_us);
        }
    }

    totals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    cpus.sort_by(|a, b| a.partial_cmp(b).unwrap());
    gpus.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mid = totals.len() / 2;

    CoopBenchResult {
        size: n,
        log_n,
        split_layer,
        median_total_us: totals[mid],
        median_cpu_us: cpus[mid],
        median_gpu_us: gpus[mid],
        min_total_us: totals[0],
        max_total_us: totals[totals.len() - 1],
    }
}

fn run_algorithm_shootout() {
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

        let r = bench_variant("cpu-reference", &cpu, &data, size);
        println!("{},{},{:.1},{:.1},{:.1},{:.1}", r.variant, r.size, r.median_us, r.min_us, r.max_us, r.throughput_melem_s);
        all_results.push(r);

        let r = bench_variant("v1-ct-dit-r2", &v1, &data, size);
        println!("{},{},{:.1},{:.1},{:.1},{:.1}", r.variant, r.size, r.median_us, r.min_us, r.max_us, r.throughput_melem_s);
        all_results.push(r);

        let r = bench_variant("v2-ct-gs-r2", &v2, &data, size);
        println!("{},{},{:.1},{:.1},{:.1},{:.1}", r.variant, r.size, r.median_us, r.min_us, r.max_us, r.throughput_melem_s);
        all_results.push(r);

        let r = bench_variant("v3-stockham-r2", &v3, &data, size);
        println!("{},{},{:.1},{:.1},{:.1},{:.1}", r.variant, r.size, r.median_us, r.min_us, r.max_us, r.throughput_melem_s);
        all_results.push(r);

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

fn run_cooperative_sweep() {
    let dir = shader_dir();

    if let Ok(ctx) = MetalContext::new(&dir) {
        let info = ctx.device_info();
        eprintln!("GPU: {}", info);
        eprintln!();
    }

    let coop = CooperativeNttContext::new(&dir).expect("Failed to init cooperative NTT");

    // CSV header
    println!("size,log_n,split_layer,median_total_us,median_cpu_us,median_gpu_us,min_total_us,max_total_us");

    for &size in COOP_SIZES {
        let log_n = size.trailing_zeros() as usize;
        let input = lcg_data_u32(size, 12345 + size as u64);

        eprintln!("Cooperative sweep: size 2^{} ({} elements), {} split values...",
            log_n, size, log_n + 1);

        let mut results: Vec<CoopBenchResult> = Vec::new();
        for split in 0..=log_n {
            let r = bench_cooperative(&coop, &input, log_n, split);
            println!("{},{},{},{:.1},{:.1},{:.1},{:.1},{:.1}",
                r.size, r.log_n, r.split_layer,
                r.median_total_us, r.median_cpu_us, r.median_gpu_us,
                r.min_total_us, r.max_total_us);
            results.push(r);
        }

        // Find optimal split from already-measured results
        let best = results.iter()
            .min_by(|a, b| a.median_total_us.partial_cmp(&b.median_total_us).unwrap())
            .unwrap();
        eprintln!("  Optimal split for 2^{}: {} (CPU does {} layers, GPU does {}) = {:.0} us",
            log_n, best.split_layer, best.split_layer, log_n - best.split_layer, best.median_total_us);
    }

    eprintln!();
    eprintln!("Config: {} warmup + {} iterations per (size, split) pair", COOP_WARMUP, COOP_ITERATIONS);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--cooperative" || a == "--coop") {
        run_cooperative_sweep();
    } else {
        run_algorithm_shootout();
    }
}
