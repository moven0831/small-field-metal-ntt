// NTT algorithm comparison benchmark.
//
// Compares 4 variants over M31 (Circle NTT) on Apple Metal:
//   1. CT-DIT radix-2 with explicit bit-reversal (naive baseline)
//   2. CT-DIT/GS-DIF paired radix-2 (in-place, no permutation)
//   3. Stockham radix-2 (out-of-place, autosort)
//   4. CT-DIT/GS-DIF paired radix-4 (half the barrier count)
//
// Plus CPU NEON reference baseline.
//
// Outputs CSV to stdout. Use `cargo bench --release` to run.

fn main() {
    println!("NTT Algorithm Shootout — Metal over M31");
    println!("========================================");
    println!();
    println!("TODO: Implement benchmark harness");
    println!("See design doc for full methodology.");
}
