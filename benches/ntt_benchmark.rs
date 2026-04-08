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
use small_field_metal_ntt::ntt::m31::cooperative::{self, CooperativeNttContext};
use small_field_metal_ntt::ntt::m31::cpu_reference::CpuReferenceBackend;
use small_field_metal_ntt::ntt::m31::metal_ct_dit_r2::MetalCtDitR2;
use small_field_metal_ntt::ntt::m31::metal_ct_gs_r2::MetalCtGsR2;
use small_field_metal_ntt::ntt::m31::metal_ct_gs_r4::MetalCtGsR4;
use small_field_metal_ntt::ntt::m31::metal_stockham_r2::MetalStockhamR2;
use small_field_metal_ntt::ntt::NttBackend;
use std::path::PathBuf;
use std::time::Instant;

const WARMUP: usize = 5;
const ITERATIONS: usize = 20;
const SIZES: &[usize] = &[
    1 << 10, // 1K
    1 << 12, // 4K
    1 << 14, // 16K
    1 << 16, // 64K
    1 << 18, // 256K
    1 << 20, // 1M
];

// Cooperative sweep: more iterations for statistical significance.
// Each sweep tests all split values (0..log_n), so total calls = sizes * splits * (warmup+iters).
const COOP_WARMUP: usize = 5;
const COOP_ITERATIONS: usize = 20;
const COOP_SIZES: &[usize] = &[
    1 << 10, // 1K
    1 << 14, // 16K
    1 << 16, // 64K
    1 << 18, // 256K
    1 << 20, // 1M
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
    stddev_us: f64,
    p95_us: f64,
    cv_pct: f64,
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
    let n = times.len();
    let median = times[n / 2];
    let min = times[0];
    let max = times[n - 1];
    let p95 = times[(n as f64 * 0.95) as usize];
    let mean = times.iter().sum::<f64>() / n as f64;
    let variance = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / n as f64;
    let stddev = variance.sqrt();
    let cv = if mean > 0.0 {
        stddev / mean * 100.0
    } else {
        0.0
    };
    let throughput = (size as f64) / (median / 1_000_000.0) / 1_000_000.0; // Melem/s

    BenchResult {
        variant: name.to_string(),
        size,
        median_us: median,
        min_us: min,
        max_us: max,
        stddev_us: stddev,
        p95_us: p95,
        cv_pct: cv,
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
    let cpu = CpuReferenceBackend::new();
    let v1 = MetalCtDitR2::new(&dir).expect("Failed to init V1");
    let v2 = MetalCtGsR2::new(&dir).expect("Failed to init V2");
    let v3 = MetalStockhamR2::new(&dir).expect("Failed to init V3");
    let v4 = MetalCtGsR4::new(&dir).expect("Failed to init V4");

    // CSV header
    println!("variant,size,median_us,min_us,max_us,stddev_us,p95_us,cv_pct,throughput_melem_s");

    let mut all_results: Vec<BenchResult> = Vec::new();

    for &size in SIZES {
        let data = lcg_data(size, 12345 + size as u64);

        eprintln!(
            "Benchmarking size 2^{} ({} elements)...",
            size.trailing_zeros(),
            size
        );

        let r = bench_variant("cpu-reference", &cpu, &data, size);
        println!(
            "{},{},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1}",
            r.variant,
            r.size,
            r.median_us,
            r.min_us,
            r.max_us,
            r.stddev_us,
            r.p95_us,
            r.cv_pct,
            r.throughput_melem_s
        );
        all_results.push(r);

        let r = bench_variant("v1-ct-dit-r2", &v1, &data, size);
        println!(
            "{},{},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1}",
            r.variant,
            r.size,
            r.median_us,
            r.min_us,
            r.max_us,
            r.stddev_us,
            r.p95_us,
            r.cv_pct,
            r.throughput_melem_s
        );
        all_results.push(r);

        let r = bench_variant("v2-ct-gs-r2", &v2, &data, size);
        println!(
            "{},{},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1}",
            r.variant,
            r.size,
            r.median_us,
            r.min_us,
            r.max_us,
            r.stddev_us,
            r.p95_us,
            r.cv_pct,
            r.throughput_melem_s
        );
        all_results.push(r);

        let r = bench_variant("v3-stockham-r2", &v3, &data, size);
        println!(
            "{},{},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1}",
            r.variant,
            r.size,
            r.median_us,
            r.min_us,
            r.max_us,
            r.stddev_us,
            r.p95_us,
            r.cv_pct,
            r.throughput_melem_s
        );
        all_results.push(r);

        let r = bench_variant("v4-ct-gs-r4", &v4, &data, size);
        println!(
            "{},{},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1}",
            r.variant,
            r.size,
            r.median_us,
            r.min_us,
            r.max_us,
            r.stddev_us,
            r.p95_us,
            r.cv_pct,
            r.throughput_melem_s
        );
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

    let variants = [
        "cpu-reference",
        "v1-ct-dit-r2",
        "v2-ct-gs-r2",
        "v3-stockham-r2",
        "v4-ct-gs-r4",
    ];
    let labels = [
        "CPU ref    ",
        "V1 CT-DIT  ",
        "V2 CT-GS r2",
        "V3 Stockham",
        "V4 CT-GS r4",
    ];

    for (vi, variant) in variants.iter().enumerate() {
        let mut row = format!("║ {} ║", labels[vi]);
        for &size in SIZES {
            if let Some(r) = all_results
                .iter()
                .find(|r| r.variant == *variant && r.size == size)
            {
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
    if let Some(winner) = gpu_results
        .iter()
        .min_by(|a, b| a.median_us.partial_cmp(&b.median_us).unwrap())
    {
        eprintln!();
        eprintln!(
            "Winner at 2^{}: {} ({:.0} us, {:.1} Melem/s)",
            largest.trailing_zeros(),
            winner.variant,
            winner.median_us,
            winner.throughput_melem_s
        );
    }

    // Warn about noisy measurements (CV > 10%)
    let noisy: Vec<&BenchResult> = all_results.iter().filter(|r| r.cv_pct > 10.0).collect();
    if !noisy.is_empty() {
        eprintln!();
        eprintln!(
            "Warning: {} measurement(s) with CV > 10% (noisy):",
            noisy.len()
        );
        for r in &noisy {
            eprintln!(
                "  {} at 2^{}: CV={:.1}% (stddev={:.1}us, median={:.1}us)",
                r.variant,
                r.size.trailing_zeros(),
                r.cv_pct,
                r.stddev_us,
                r.median_us
            );
        }
    }

    eprintln!();
    eprintln!(
        "Config: {} warmup + {} iterations per (variant, size) pair",
        WARMUP, ITERATIONS
    );
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

        eprintln!(
            "Cooperative sweep: size 2^{} ({} elements), {} split values...",
            log_n,
            size,
            log_n + 1
        );

        let mut results: Vec<CoopBenchResult> = Vec::new();
        for split in 0..=log_n {
            let r = bench_cooperative(&coop, &input, log_n, split);
            println!(
                "{},{},{},{:.1},{:.1},{:.1},{:.1},{:.1}",
                r.size,
                r.log_n,
                r.split_layer,
                r.median_total_us,
                r.median_cpu_us,
                r.median_gpu_us,
                r.min_total_us,
                r.max_total_us
            );
            results.push(r);
        }

        // Find optimal split from already-measured results
        let best = results
            .iter()
            .min_by(|a, b| a.median_total_us.partial_cmp(&b.median_total_us).unwrap())
            .unwrap();
        eprintln!(
            "  Optimal split for 2^{}: {} (CPU does {} layers, GPU does {}) = {:.0} us",
            log_n,
            best.split_layer,
            best.split_layer,
            log_n - best.split_layer,
            best.median_total_us
        );
    }

    eprintln!();
    eprintln!(
        "Config: {} warmup + {} iterations per (size, split) pair",
        COOP_WARMUP, COOP_ITERATIONS
    );
}

// ─── Coset LDE Benchmark ────────────────────────────────────────────────
// Matches Plonky3 coset_lde_batch parameters: BabyBear field, 2^20 x 256 columns, 2x expansion.

fn run_coset_lde_benchmark() {
    use small_field_metal_ntt::field::babybear::BabyBear;
    use small_field_metal_ntt::lde::CosetLdeBatch;

    let dir = shader_dir();
    let lde = CosetLdeBatch::new(&dir).expect("Failed to init CosetLdeBatch");

    let info = lde.device_info();
    eprintln!("GPU: {}", info);
    eprintln!();

    println!(
        "mode,n,batch_size,expansion,median_ms,min_ms,max_ms,stddev_ms,cv_pct,throughput_melem_s"
    );

    let configs: &[(usize, usize, usize)] = &[
        // (log_n, batch_size, added_bits)
        (16, 256, 1), // warmup: smaller size
        (18, 256, 1), // medium
        (20, 256, 1), // target: matches zk-autoresearch (2x expansion)
        (20, 256, 2), // 4x expansion
    ];

    for &(log_n, batch_size, added_bits) in configs {
        let n = 1usize << log_n;
        let n_ext = n << added_bits;
        let expansion = 1 << added_bits;

        eprintln!(
            "Coset LDE: 2^{} x {} cols, {}x expansion (output: {} Melems, {:.0} MB)...",
            log_n,
            batch_size,
            expansion,
            n_ext * batch_size / 1_000_000,
            (n_ext * batch_size * 4) as f64 / (1024.0 * 1024.0),
        );

        // Generate test data (BabyBear Montgomery form)
        let input: Vec<u32> = {
            let mut seed: u64 = 42 + log_n as u64;
            (0..n * batch_size)
                .map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    BabyBear::to_monty(((seed >> 33) as u32) % BabyBear::P).0
                })
                .collect()
        };

        let mut times_ms = Vec::with_capacity(WARMUP + ITERATIONS);

        for i in 0..(WARMUP + ITERATIONS) {
            let result = lde
                .execute(&input, log_n, batch_size, added_bits)
                .expect("LDE execution failed");
            if i >= WARMUP {
                times_ms.push(result.total_ns as f64 / 1_000_000.0);
            }
        }

        times_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let nm = times_ms.len();
        let median = times_ms[nm / 2];
        let min = times_ms[0];
        let max = times_ms[nm - 1];
        let mean = times_ms.iter().sum::<f64>() / nm as f64;
        let variance = times_ms.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / nm as f64;
        let stddev = variance.sqrt();
        let cv = if mean > 0.0 {
            stddev / mean * 100.0
        } else {
            0.0
        };
        let output_elems = (n_ext * batch_size) as f64;
        let throughput = output_elems / (median / 1000.0) / 1_000_000.0; // Melem/s

        println!(
            "coset-lde,{},{},{},{:.2},{:.2},{:.2},{:.2},{:.1},{:.1}",
            n, batch_size, expansion, median, min, max, stddev, cv, throughput
        );

        eprintln!(
            "  Median: {:.2} ms | Min: {:.2} ms | CV: {:.1}% | Throughput: {:.1} Melem/s",
            median, min, cv, throughput
        );
    }

    eprintln!();
    eprintln!("╔═══════════════════════════════════════════════════════════════════╗");
    eprintln!("║              Coset LDE Comparison                                ║");
    eprintln!("╠═══════════════════════════════════════════════════════════════════╣");
    eprintln!("║ Plonky3 CPU (same M3, Radix2DitParallel, --features parallel):   ║");
    eprintln!("║   2^16 x 256: ~69 ms | 2^18: ~237 ms | 2^20: ~1177 ms          ║");
    eprintln!("║                                                                   ║");
    eprintln!("║ This benchmark (Apple Metal GPU, BabyBear):                       ║");
    eprintln!("║   See results above                                               ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════════╝");
    eprintln!();
    eprintln!("Config: {} warmup + {} iterations", WARMUP, ITERATIONS);
}

// ─── Batched NTT Benchmark ──────────────────────────────────────────────
// Raw batched forward NTT throughput (no LDE overhead).

fn run_batch_ntt_benchmark() {
    use small_field_metal_ntt::field::babybear::BabyBear;
    use small_field_metal_ntt::ntt::bb_metal_r2::BbMetalR2;

    let dir = shader_dir();
    let gpu = BbMetalR2::new(&dir).expect("Failed to init BbMetalR2");

    let info = gpu.ctx().device_info();
    eprintln!("GPU: {}", info);
    eprintln!();

    println!("mode,n,batch_size,median_ms,min_ms,max_ms,throughput_melem_s");

    let configs: &[(usize, usize)] = &[
        // (log_n, batch_size)
        (14, 256),
        (16, 256),
        (18, 256),
        (20, 256),
        (20, 1), // single NTT for comparison
    ];

    for &(log_n, batch_size) in configs {
        let n = 1usize << log_n;
        let total_elems = n * batch_size;

        eprintln!(
            "Batch NTT: 2^{} x {} cols ({} Melems)...",
            log_n,
            batch_size,
            total_elems / 1_000_000,
        );

        let input: Vec<u32> = {
            let mut seed: u64 = 99 + log_n as u64;
            (0..total_elems)
                .map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    BabyBear::to_monty(((seed >> 33) as u32) % BabyBear::P).0
                })
                .collect()
        };

        let mut times_ms = Vec::with_capacity(WARMUP + ITERATIONS);

        for i in 0..(WARMUP + ITERATIONS) {
            let (_, ns) = gpu
                .forward_ntt_batch_gpu(&input, log_n, batch_size)
                .expect("Batch NTT failed");
            if i >= WARMUP {
                times_ms.push(ns as f64 / 1_000_000.0);
            }
        }

        times_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let nm = times_ms.len();
        let median = times_ms[nm / 2];
        let min = times_ms[0];
        let max = times_ms[nm - 1];
        let throughput = (total_elems as f64) / (median / 1000.0) / 1_000_000.0;

        println!(
            "batch-ntt,{},{},{:.2},{:.2},{:.2},{:.1}",
            n, batch_size, median, min, max, throughput
        );

        eprintln!(
            "  Median: {:.2} ms | Throughput: {:.1} Melem/s",
            median, throughput
        );
    }

    eprintln!();
    eprintln!("Config: {} warmup + {} iterations", WARMUP, ITERATIONS);
}

// ─── BabyBear Algorithm Shootout ─────────────────────────────────────────
// Compare all BabyBear NTT variants: CPU ref, V1 CT-DIT, V2 CT-GS R2, V3 Stockham, V4 CT-GS R4

fn run_bb_shootout() {
    use small_field_metal_ntt::field::babybear::BabyBear;
    use small_field_metal_ntt::ntt::bb_cpu_reference::BbCpuReferenceBackend;
    use small_field_metal_ntt::ntt::bb_metal_ct_dit_r2::BbMetalCtDitR2;
    use small_field_metal_ntt::ntt::bb_metal_ct_gs_r4::BbMetalCtGsR4;
    use small_field_metal_ntt::ntt::bb_metal_r2::BbMetalR2;
    use small_field_metal_ntt::ntt::bb_metal_stockham_r2::BbMetalStockhamR2;

    let dir = shader_dir();

    if let Ok(ctx) = small_field_metal_ntt::gpu::MetalContext::new(&dir) {
        let info = ctx.device_info();
        eprintln!("GPU: {}", info);
        eprintln!();
    }

    let cpu = BbCpuReferenceBackend::new();
    let v1 = BbMetalCtDitR2::new(&dir).expect("Failed to init BB V1");
    let v2 = BbMetalR2::new(&dir).expect("Failed to init BB V2");
    let v3 = BbMetalStockhamR2::new(&dir).expect("Failed to init BB V3");
    let v4 = BbMetalCtGsR4::new(&dir).expect("Failed to init BB V4");

    println!("variant,size,median_us,min_us,max_us,stddev_us,p95_us,cv_pct,throughput_melem_s");

    let mut all_results: Vec<BenchResult> = Vec::new();

    for &size in SIZES {
        let data: Vec<BabyBear> = {
            let mut seed: u64 = 12345 + size as u64;
            (0..size)
                .map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    BabyBear::to_monty(((seed >> 33) as u32) % BabyBear::P)
                })
                .collect()
        };

        eprintln!(
            "BB Shootout: size 2^{} ({} elements)...",
            size.trailing_zeros(),
            size
        );

        let r = bench_variant("bb-cpu-ref", &cpu, &data, size);
        println!(
            "{},{},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1}",
            r.variant,
            r.size,
            r.median_us,
            r.min_us,
            r.max_us,
            r.stddev_us,
            r.p95_us,
            r.cv_pct,
            r.throughput_melem_s
        );
        all_results.push(r);

        let r = bench_variant("bb-v1-ct-dit-r2", &v1, &data, size);
        println!(
            "{},{},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1}",
            r.variant,
            r.size,
            r.median_us,
            r.min_us,
            r.max_us,
            r.stddev_us,
            r.p95_us,
            r.cv_pct,
            r.throughput_melem_s
        );
        all_results.push(r);

        let r = bench_variant("bb-v2-ct-gs-r2", &v2, &data, size);
        println!(
            "{},{},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1}",
            r.variant,
            r.size,
            r.median_us,
            r.min_us,
            r.max_us,
            r.stddev_us,
            r.p95_us,
            r.cv_pct,
            r.throughput_melem_s
        );
        all_results.push(r);

        let r = bench_variant("bb-v3-stockham-r2", &v3, &data, size);
        println!(
            "{},{},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1}",
            r.variant,
            r.size,
            r.median_us,
            r.min_us,
            r.max_us,
            r.stddev_us,
            r.p95_us,
            r.cv_pct,
            r.throughput_melem_s
        );
        all_results.push(r);

        let r = bench_variant("bb-v4-ct-gs-r4", &v4, &data, size);
        println!(
            "{},{},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1}",
            r.variant,
            r.size,
            r.median_us,
            r.min_us,
            r.max_us,
            r.stddev_us,
            r.p95_us,
            r.cv_pct,
            r.throughput_melem_s
        );
        all_results.push(r);
    }

    // Summary table
    eprintln!();
    eprintln!("╔═══════════════════════════════════════════════════════════════════════════════╗");
    eprintln!("║                BabyBear NTT Algorithm Shootout (Montgomery)                   ║");
    eprintln!("╠════════════════╦════════╦════════╦════════╦════════╦════════╦═════════════════╣");
    eprintln!("║ Variant        ║ 2^10   ║ 2^12   ║ 2^14   ║ 2^16   ║ 2^18   ║ 2^20          ║");
    eprintln!("║                ║ (us)   ║ (us)   ║ (us)   ║ (us)   ║ (us)   ║ (us)           ║");
    eprintln!("╠════════════════╬════════╬════════╬════════╬════════╬════════╬═════════════════╣");

    let variants = [
        "bb-cpu-ref",
        "bb-v1-ct-dit-r2",
        "bb-v2-ct-gs-r2",
        "bb-v3-stockham-r2",
        "bb-v4-ct-gs-r4",
    ];
    let labels = [
        "CPU ref      ",
        "V1 CT-DIT    ",
        "V2 CT-GS r2  ",
        "V3 Stockham  ",
        "V4 CT-GS r4  ",
    ];

    for (vi, variant) in variants.iter().enumerate() {
        let mut row = format!("║ {} ║", labels[vi]);
        for &size in SIZES {
            if let Some(r) = all_results
                .iter()
                .find(|r| r.variant == *variant && r.size == size)
            {
                row.push_str(&format!(" {:>6.0} ║", r.median_us));
            } else {
                row.push_str("    N/A ║");
            }
        }
        eprintln!("{}", row);
    }

    eprintln!("╚════════════════╩════════╩════════╩════════╩════════╩════════╩═════════════════╝");

    let largest = SIZES.last().unwrap();
    let gpu_results: Vec<&BenchResult> = all_results
        .iter()
        .filter(|r| r.size == *largest && r.variant != "bb-cpu-ref")
        .collect();
    if let Some(winner) = gpu_results
        .iter()
        .min_by(|a, b| a.median_us.partial_cmp(&b.median_us).unwrap())
    {
        eprintln!();
        eprintln!(
            "Winner at 2^{}: {} ({:.0} us, {:.1} Melem/s)",
            largest.trailing_zeros(),
            winner.variant,
            winner.median_us,
            winner.throughput_melem_s
        );
    }

    eprintln!();
    eprintln!(
        "Config: {} warmup + {} iterations per (variant, size) pair",
        WARMUP, ITERATIONS
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--cooperative" || a == "--coop") {
        run_cooperative_sweep();
    } else if args.iter().any(|a| a == "--coset-lde" || a == "--lde") {
        run_coset_lde_benchmark();
    } else if args.iter().any(|a| a == "--batch-ntt" || a == "--batch") {
        run_batch_ntt_benchmark();
    } else if args.iter().any(|a| a == "--bb-shootout" || a == "--bb") {
        run_bb_shootout();
    } else {
        run_algorithm_shootout();
    }
}
