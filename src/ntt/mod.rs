pub mod cooperative;
pub mod cpu_reference;
pub mod metal_ct_dit_r2;
pub mod metal_ct_gs_r2;
pub mod metal_ct_gs_r4;
pub mod metal_stockham_r2;
pub mod twiddles;

use crate::field::Field;

/// Backend trait for NTT implementations.
///
/// This trait abstracts over the GPU API (Metal, Vulkan, WebGPU) and algorithm
/// variant. The benchmark harness operates on this trait, enabling fair comparison
/// across algorithm families and future multi-platform support.
///
/// ```text
/// trait NttBackend<F: Field>
///   |
///   +-- CpuReferenceBackend       (forward + inverse)
///   +-- MetalCtDitR2              (V1: naive baseline, forward only)
///   +-- MetalCtGsR2              (V2: in-place, forward + inverse)
///   +-- MetalStockhamR2          (V3: out-of-place, forward + inverse)
///   +-- MetalCtGsR4              (V4: radix-4, forward + inverse)
/// ```
pub trait NttBackend<F: Field> {
    /// Algorithm variant name for reporting.
    fn name(&self) -> &str;

    /// Forward NTT (evaluation): coefficients -> evaluations.
    /// For M31 Circle NTT, this is the DCCT forward transform.
    ///
    /// The `twiddles` parameter provides pre-computed twiddle factors.
    /// GPU backends use these (stored in device memory). The CPU reference
    /// backend ignores this parameter and generates twiddles internally
    /// for simplicity.
    fn forward_ntt(&self, data: &mut [F], twiddles: &[F]) -> Result<(), NttError>;

    /// Inverse NTT (interpolation): evaluations -> coefficients.
    /// For M31 Circle NTT, this is the DCCT inverse transform (iDCCT).
    ///
    /// The `twiddles` parameter provides pre-computed inverse twiddle factors.
    /// See `forward_ntt` for the contract on twiddle usage per backend.
    fn inverse_ntt(&self, data: &mut [F], twiddles: &[F]) -> Result<(), NttError>;

    /// Pointwise multiplication of two vectors in evaluation domain.
    /// Used for round-trip benchmark: fwd(a) * fwd(b) -> inv() = a*b.
    fn pointwise_mul(&self, a: &[F], b: &[F], out: &mut [F]) -> Result<(), NttError>;
}

#[derive(Debug)]
pub enum NttError {
    /// Input size is not a power of two.
    InvalidSize(usize),
    /// GPU device not found or not supported.
    DeviceNotFound,
    /// GPU buffer allocation failed.
    BufferAllocFailed { requested_bytes: usize },
    /// Metal shader compilation failed.
    ShaderCompileError(String),
    /// GPU execution error.
    GpuExecutionError(String),
    /// Correctness check failed: GPU output != CPU reference.
    CorrectnessMismatch { index: usize, expected: u32, got: u32 },
}

impl std::fmt::Display for NttError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NttError::InvalidSize(n) => write!(f, "Invalid NTT size: {} (must be power of 2)", n),
            NttError::DeviceNotFound => write!(f, "No Metal GPU device found"),
            NttError::BufferAllocFailed { requested_bytes } =>
                write!(f, "GPU buffer allocation failed ({} bytes requested)", requested_bytes),
            NttError::ShaderCompileError(msg) => write!(f, "Metal shader compilation error: {}", msg),
            NttError::GpuExecutionError(msg) => write!(f, "GPU execution error: {}", msg),
            NttError::CorrectnessMismatch { index, expected, got } =>
                write!(f, "Correctness mismatch at index {}: expected {}, got {}", index, expected, got),
        }
    }
}

impl std::error::Error for NttError {}
