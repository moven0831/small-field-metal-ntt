/// Benchmark harness for NTT algorithm comparison.
///
/// Supports:
/// - GPU timing via MTLCommandBuffer timestamps
/// - Workgroup size sweep (64/128/256/512/1024)
/// - Thermal stability testing (60s continuous)
/// - Round-trip benchmark (fwd + mul + inv)
/// - CSV output + HTML report generation
///
/// See design doc for full methodology.
pub struct BenchConfig {
    /// Transform sizes to benchmark (powers of 2).
    pub sizes: Vec<usize>,
    /// Number of iterations per (variant, size) pair.
    pub iterations: usize,
    /// Number of warmup iterations to discard.
    pub warmup: usize,
    /// Workgroup sizes to sweep.
    pub workgroup_sizes: Vec<u32>,
    /// Duration for thermal stability test (seconds).
    pub thermal_duration_secs: u64,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            sizes: vec![1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18, 1 << 20],
            iterations: 20,
            warmup: 5,
            workgroup_sizes: vec![64, 128, 256, 512, 1024],
            thermal_duration_secs: 60,
        }
    }
}
