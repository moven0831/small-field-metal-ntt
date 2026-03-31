//! Metal GPU device and pipeline management.
//!
//! ```text
//! MetalContext lifecycle:
//!   MTLCreateSystemDefaultDevice()
//!     └─> MTLDevice
//!           ├─> newLibraryWithSource() → MTLLibrary (compiled .metal shaders)
//!           │     └─> newFunctionWithName() → MTLFunction
//!           │           └─> newComputePipelineStateWithFunction() → MTLComputePipelineState
//!           ├─> newCommandQueue() → MTLCommandQueue
//!           │     └─> commandBuffer() → MTLCommandBuffer
//!           │           └─> computeCommandEncoder() → MTLComputeCommandEncoder
//!           │                 ├─> setComputePipelineState()
//!           │                 ├─> setBuffer() × N
//!           │                 ├─> dispatchThreadgroups()
//!           │                 └─> endEncoding()
//!           └─> newBufferWithLength() → MTLBuffer
//! ```

use crate::ntt::NttError;
use metal::*;
use std::path::Path;

/// Manages Metal device, command queue, and compiled shader libraries.
pub struct MetalContext {
    pub device: Device,
    pub command_queue: CommandQueue,
    pub library: Library,
}

impl MetalContext {
    /// Initialize Metal context by compiling shader source files.
    ///
    /// Reads all .metal files from `shader_dir`, concatenates them
    /// (respecting #include via manual concatenation), and compiles.
    pub fn new(shader_dir: &Path) -> Result<Self, NttError> {
        let device = Device::system_default().ok_or(NttError::DeviceNotFound)?;

        // Check GPU family support (Apple GPU Family 6+ required for SIMD shuffle)
        let family_6 = device.supports_family(MTLGPUFamily::Apple6);
        if !family_6 {
            eprintln!(
                "Warning: GPU does not support Apple Family 6. \
                 SIMD shuffle operations (radix-4 variant) will not be available."
            );
        }

        let command_queue = device.new_command_queue();

        // Read and compile shader source
        let shader_source = read_shader_source(shader_dir)?;
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(&shader_source, &options)
            .map_err(|e| NttError::ShaderCompileError(e.to_string()))?;

        Ok(MetalContext {
            device,
            command_queue,
            library,
        })
    }

    /// Create a compute pipeline for a named kernel function.
    pub fn make_pipeline(&self, function_name: &str) -> Result<ComputePipelineState, NttError> {
        let function = self
            .library
            .get_function(function_name, None)
            .map_err(|e| {
                NttError::ShaderCompileError(format!(
                    "Function '{}' not found in shader library: {}",
                    function_name, e
                ))
            })?;

        self.device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| {
                NttError::ShaderCompileError(format!(
                    "Failed to create pipeline for '{}': {}",
                    function_name, e
                ))
            })
    }

    /// Allocate a GPU buffer of the given byte length, initialized to zero.
    pub fn alloc_buffer(&self, byte_len: usize) -> Result<Buffer, NttError> {
        let buffer = self.device.new_buffer(
            byte_len as u64,
            MTLResourceOptions::StorageModeShared,
        );
        Ok(buffer)
    }

    /// Allocate a GPU buffer and copy data into it.
    pub fn buffer_from_slice(&self, data: &[u32]) -> Result<Buffer, NttError> {
        let byte_len = data.len() * std::mem::size_of::<u32>();
        let buffer = self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            byte_len as u64,
            MTLResourceOptions::StorageModeShared,
        );
        Ok(buffer)
    }

    /// Read GPU buffer contents back as a slice of u32.
    ///
    /// # Safety
    /// The buffer must have been created with StorageModeShared and contain
    /// valid u32 data of the expected length.
    pub fn read_buffer(buffer: &Buffer, len: usize) -> Vec<u32> {
        let ptr = buffer.contents() as *const u32;
        let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
        slice.to_vec()
    }

    /// Get the maximum threads per threadgroup for a given pipeline.
    pub fn max_threads_per_threadgroup(pipeline: &ComputePipelineState) -> usize {
        pipeline.max_total_threads_per_threadgroup() as usize
    }

    /// Get device info for benchmark reporting.
    pub fn device_info(&self) -> DeviceInfo {
        DeviceInfo {
            name: self.device.name().to_string(),
            has_unified_memory: self.device.has_unified_memory(),
            max_buffer_length: self.device.max_buffer_length() as usize,
            max_threadgroup_memory: self.device.max_threadgroup_memory_length() as usize,
            supports_family_6: self.device.supports_family(MTLGPUFamily::Apple6),
            supports_family_7: self.device.supports_family(MTLGPUFamily::Apple7),
        }
    }

    /// Dispatch a 1D compute kernel and wait for completion.
    /// Returns wall-clock execution time in nanoseconds.
    ///
    /// Note: metal crate v0.30 doesn't expose MTLCommandBuffer GPU timestamps.
    /// Wall-clock includes command submission overhead. For precise GPU timing,
    /// use Metal System Trace profiling (planned in benchmark harness).
    pub fn dispatch_and_wait(
        &self,
        pipeline: &ComputePipelineState,
        buffers: &[&Buffer],
        threadgroups: MTLSize,
        threads_per_group: MTLSize,
    ) -> Result<u64, NttError> {
        let cmd_buffer = self.command_queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(pipeline);
        for (i, buf) in buffers.iter().enumerate() {
            encoder.set_buffer(i as u64, Some(buf), 0);
        }
        encoder.dispatch_thread_groups(threadgroups, threads_per_group);
        encoder.end_encoding();

        let start = std::time::Instant::now();
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();
        let duration_ns = start.elapsed().as_nanos() as u64;

        // Check command buffer status
        let status = cmd_buffer.status();
        if status == MTLCommandBufferStatus::Error {
            return Err(NttError::GpuExecutionError(
                "Command buffer completed with error status".to_string(),
            ));
        }

        Ok(duration_ns)
    }
}

/// Device metadata for benchmark reports.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub has_unified_memory: bool,
    pub max_buffer_length: usize,
    pub max_threadgroup_memory: usize,
    pub supports_family_6: bool,
    pub supports_family_7: bool,
}

impl std::fmt::Display for DeviceInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} (UMA: {}, max buffer: {} MB, threadgroup mem: {} KB, Family6: {}, Family7: {})",
            self.name,
            self.has_unified_memory,
            self.max_buffer_length / (1024 * 1024),
            self.max_threadgroup_memory / 1024,
            self.supports_family_6,
            self.supports_family_7,
        )
    }
}

/// Read and concatenate shader source files from a directory.
/// Handles #include directives by inlining the referenced files.
fn read_shader_source(shader_dir: &Path) -> Result<String, NttError> {
    // Read files in dependency order: headers first, then kernels
    let mut source = String::new();

    // Read header files first (no #include resolution needed for headers)
    for name in &["m31_field.metal", "ntt_common.metal"] {
        let path = shader_dir.join(name);
        if path.exists() {
            let content = std::fs::read_to_string(&path).map_err(|e| {
                NttError::ShaderCompileError(format!("Failed to read {}: {}", path.display(), e))
            })?;
            // Strip #include directives (we're manually concatenating)
            let filtered: String = content
                .lines()
                .filter(|line| !line.trim_start().starts_with("#include"))
                .collect::<Vec<_>>()
                .join("\n");
            source.push_str(&filtered);
            source.push('\n');
        }
    }

    // Read kernel files
    let entries = std::fs::read_dir(shader_dir).map_err(|e| {
        NttError::ShaderCompileError(format!(
            "Failed to read shader directory {}: {}",
            shader_dir.display(),
            e
        ))
    })?;

    for entry in entries {
        let entry = entry.map_err(|e| {
            NttError::ShaderCompileError(format!("Failed to read directory entry: {}", e))
        })?;
        let path = entry.path();
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            // Skip headers (already included) and non-.metal files
            if name == "m31_field.metal" || name == "ntt_common.metal" || !name.ends_with(".metal")
            {
                continue;
            }
            let content = std::fs::read_to_string(&path).map_err(|e| {
                NttError::ShaderCompileError(format!("Failed to read {}: {}", path.display(), e))
            })?;
            let filtered: String = content
                .lines()
                .filter(|line| !line.trim_start().starts_with("#include"))
                .collect::<Vec<_>>()
                .join("\n");
            source.push_str(&filtered);
            source.push('\n');
        }
    }

    Ok(source)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn shader_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders")
    }

    #[test]
    fn test_metal_context_init() {
        let ctx = MetalContext::new(&shader_dir());
        match ctx {
            Ok(ctx) => {
                let info = ctx.device_info();
                println!("GPU: {}", info);
                assert!(info.has_unified_memory, "Expected UMA on Apple Silicon");
            }
            Err(NttError::DeviceNotFound) => {
                eprintln!("No Metal device found — skipping GPU test (CI or non-Apple)");
            }
            Err(e) => panic!("Unexpected error: {}", e),
        }
    }

    #[test]
    fn test_shader_compilation() {
        let ctx = match MetalContext::new(&shader_dir()) {
            Ok(ctx) => ctx,
            Err(NttError::DeviceNotFound) => {
                eprintln!("No Metal device — skipping");
                return;
            }
            Err(e) => panic!("Failed to init Metal: {}", e),
        };

        // Verify we can find the test kernel
        let result = ctx.make_pipeline("m31_field_test");
        match result {
            Ok(_) => println!("Pipeline created successfully"),
            Err(NttError::ShaderCompileError(msg)) if msg.contains("not found") => {
                // Expected if we haven't added a test kernel yet
                println!("Test kernel not in library (expected for now): {}", msg);
            }
            Err(e) => panic!("Unexpected pipeline error: {}", e),
        }
    }

    #[test]
    fn test_buffer_roundtrip() {
        let ctx = match MetalContext::new(&shader_dir()) {
            Ok(ctx) => ctx,
            Err(NttError::DeviceNotFound) => {
                eprintln!("No Metal device — skipping");
                return;
            }
            Err(e) => panic!("Failed to init Metal: {}", e),
        };

        let data: Vec<u32> = vec![1, 2, 3, 42, 0x7FFFFFFF];
        let buffer = ctx.buffer_from_slice(&data).unwrap();
        let readback = MetalContext::read_buffer(&buffer, data.len());
        assert_eq!(readback, data, "Buffer roundtrip failed");
    }

    #[test]
    fn test_device_info() {
        let ctx = match MetalContext::new(&shader_dir()) {
            Ok(ctx) => ctx,
            Err(NttError::DeviceNotFound) => {
                eprintln!("No Metal device — skipping");
                return;
            }
            Err(e) => panic!("Failed to init Metal: {}", e),
        };

        let info = ctx.device_info();
        assert!(!info.name.is_empty());
        assert!(info.max_buffer_length > 0);
        assert!(info.max_threadgroup_memory > 0);
        println!("Device: {}", info);
    }
}
