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

use crate::field::m31::M31;
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
    ///
    /// Note: the `metal` crate panics internally if the allocation fails (Obj-C
    /// null return). Zero-length allocations are rejected here to avoid
    /// undefined behavior in Metal.
    pub fn alloc_buffer(&self, byte_len: usize) -> Result<Buffer, NttError> {
        if byte_len == 0 {
            return Err(NttError::BufferAllocFailed { requested_bytes: 0 });
        }
        let buffer = self
            .device
            .new_buffer(byte_len as u64, MTLResourceOptions::StorageModeShared);
        Ok(buffer)
    }

    /// Allocate a GPU buffer and copy data into it.
    ///
    /// Note: the `metal` crate panics internally if the allocation fails.
    /// Empty slices are rejected to avoid zero-length Metal allocations.
    pub fn buffer_from_slice(&self, data: &[u32]) -> Result<Buffer, NttError> {
        if data.is_empty() {
            return Err(NttError::BufferAllocFailed { requested_bytes: 0 });
        }
        let byte_len = std::mem::size_of_val(data);
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

    // ── Batch dispatch API ───────────────────────────────────────────────
    // Encode multiple dispatches into a single command buffer, submit once.
    // Metal inserts implicit memory barriers between compute encoders on
    // the same command buffer, so sequential NTT layers are safe.
    //
    // Temporary buffers (twiddles, params) created during encoding are pushed
    // into a `Vec<Buffer>` ("retain list") that the caller must keep alive
    // until `submit_batch` returns. This ensures GPU reads valid memory
    // regardless of Metal's Obj-C retain/release bridging details.

    /// Create a new command buffer for batching multiple dispatches.
    pub fn begin_batch(&self) -> &CommandBufferRef {
        self.command_queue.new_command_buffer()
    }

    /// Encode a single compute dispatch into an existing command buffer.
    /// Does NOT commit — call `submit_batch` when all dispatches are encoded.
    pub fn encode_dispatch(
        cmd_buffer: &CommandBufferRef,
        pipeline: &ComputePipelineState,
        buffers: &[&Buffer],
        threadgroups: MTLSize,
        threads_per_group: MTLSize,
    ) {
        let encoder = cmd_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        for (i, buf) in buffers.iter().enumerate() {
            encoder.set_buffer(i as u64, Some(buf), 0);
        }
        encoder.dispatch_thread_groups(threadgroups, threads_per_group);
        encoder.end_encoding();
    }

    /// Commit a batched command buffer and wait for GPU completion.
    /// Returns wall-clock time in nanoseconds.
    ///
    /// The `_retain` parameter keeps temporary buffers alive until GPU
    /// completion. Drop it after this call returns.
    pub fn submit_batch(
        cmd_buffer: &CommandBufferRef,
        _retain: &[Buffer],
    ) -> Result<u64, NttError> {
        let start = std::time::Instant::now();
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();
        let duration_ns = start.elapsed().as_nanos() as u64;

        let status = cmd_buffer.status();
        if status == MTLCommandBufferStatus::Error {
            return Err(NttError::GpuExecutionError(
                "Batch command buffer completed with error status".to_string(),
            ));
        }

        Ok(duration_ns)
    }

    /// Encode a radix-2 butterfly dispatch into a batch command buffer.
    /// Pushes temporary buffers into `retain` to keep them alive until GPU completion.
    pub fn encode_butterfly_r2(
        &self,
        cmd_buffer: &CommandBufferRef,
        retain: &mut Vec<Buffer>,
        pipeline: &ComputePipelineState,
        buf_data: &Buffer,
        twiddles: &[M31],
        stride: usize,
        n: usize,
    ) -> Result<(), NttError> {
        let tw_data: Vec<u32> = twiddles.iter().map(|m| m.0).collect();
        let buf_tw = self.buffer_from_slice(&tw_data)?;
        let params: Vec<u32> = vec![stride as u32, n as u32];
        let buf_p = self.buffer_from_slice(&params)?;

        let num_butterflies = (n / 2) as u64;
        let max_tpg = Self::max_threads_per_threadgroup(pipeline) as u64;
        let (tg, tpg) = Self::compute_grid_1d(num_butterflies, max_tpg.min(256));

        Self::encode_dispatch(cmd_buffer, pipeline, &[buf_data, &buf_tw, &buf_p], tg, tpg);
        retain.push(buf_tw);
        retain.push(buf_p);
        Ok(())
    }

    /// Encode a radix-4 butterfly dispatch into a batch command buffer.
    /// Pushes temporary buffers into `retain` to keep them alive until GPU completion.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_butterfly_r4(
        &self,
        cmd_buffer: &CommandBufferRef,
        retain: &mut Vec<Buffer>,
        pipeline: &ComputePipelineState,
        buf_data: &Buffer,
        tw_outer: &[M31],
        tw_inner: &[M31],
        outer_stride: usize,
        inner_stride: usize,
        n: usize,
    ) -> Result<(), NttError> {
        let tw_o: Vec<u32> = tw_outer.iter().map(|m| m.0).collect();
        let tw_i: Vec<u32> = tw_inner.iter().map(|m| m.0).collect();
        let buf_tw_o = self.buffer_from_slice(&tw_o)?;
        let buf_tw_i = self.buffer_from_slice(&tw_i)?;
        let params: Vec<u32> = vec![outer_stride as u32, inner_stride as u32, n as u32];
        let buf_p = self.buffer_from_slice(&params)?;

        let num_butterflies = (n / 4) as u64;
        let max_tpg = Self::max_threads_per_threadgroup(pipeline) as u64;
        let (tg, tpg) = Self::compute_grid_1d(num_butterflies, max_tpg.min(256));

        Self::encode_dispatch(
            cmd_buffer,
            pipeline,
            &[buf_data, &buf_tw_o, &buf_tw_i, &buf_p],
            tg,
            tpg,
        );
        retain.push(buf_tw_o);
        retain.push(buf_tw_i);
        retain.push(buf_p);
        Ok(())
    }

    /// Encode a normalize dispatch into a batch command buffer.
    /// Pushes temporary buffers into `retain` to keep them alive until GPU completion.
    pub fn encode_normalize(
        &self,
        cmd_buffer: &CommandBufferRef,
        retain: &mut Vec<Buffer>,
        pipeline: &ComputePipelineState,
        buf_data: &Buffer,
        n: usize,
        inv_n: M31,
    ) -> Result<(), NttError> {
        let params: Vec<u32> = vec![n as u32, inv_n.0];
        let buf_p = self.buffer_from_slice(&params)?;

        let max_tpg = Self::max_threads_per_threadgroup(pipeline) as u64;
        let (tg, tpg) = Self::compute_grid_1d(n as u64, max_tpg.min(256));

        Self::encode_dispatch(cmd_buffer, pipeline, &[buf_data, &buf_p], tg, tpg);
        retain.push(buf_p);
        Ok(())
    }

    // ── Single-dispatch helpers (kept for tests and simple use cases) ────

    /// Compute 1D threadgroup/grid sizes for a given number of work items.
    /// Returns (threadgroups, threads_per_group).
    ///
    /// Both `num_items` and `max_tpg` must be > 0.
    pub(crate) fn compute_grid_1d(num_items: u64, max_tpg: u64) -> (MTLSize, MTLSize) {
        debug_assert!(num_items > 0, "compute_grid_1d: num_items must be > 0");
        debug_assert!(max_tpg > 0, "compute_grid_1d: max_tpg must be > 0");
        let tpg = max_tpg.min(num_items);
        let groups = num_items.div_ceil(tpg);
        (MTLSize::new(groups, 1, 1), MTLSize::new(tpg, 1, 1))
    }

    /// Dispatch a radix-2 butterfly stage on device memory.
    ///
    /// Each layer has `n/2` butterflies at stride `2^layer`.
    /// Buffers bound: [data, twiddles, params].
    pub fn dispatch_butterfly_r2(
        &self,
        pipeline: &ComputePipelineState,
        buf_data: &Buffer,
        twiddles: &[M31],
        stride: usize,
        n: usize,
    ) -> Result<u64, NttError> {
        let tw_data: Vec<u32> = twiddles.iter().map(|m| m.0).collect();
        let buf_tw = self.buffer_from_slice(&tw_data)?;
        let params: Vec<u32> = vec![stride as u32, n as u32];
        let buf_p = self.buffer_from_slice(&params)?;

        let num_butterflies = (n / 2) as u64;
        let max_tpg = Self::max_threads_per_threadgroup(pipeline) as u64;
        let (tg, tpg) = Self::compute_grid_1d(num_butterflies, max_tpg.min(256));

        self.dispatch_and_wait(pipeline, &[buf_data, &buf_tw, &buf_p], tg, tpg)
    }

    /// Dispatch a radix-4 butterfly stage on device memory.
    ///
    /// Processes two layers at once (k, k-1). Each radix-4 butterfly
    /// operates on 4 elements, so there are `n/4` butterflies.
    /// Buffers bound: [data, twiddles_outer, twiddles_inner, params].
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_butterfly_r4(
        &self,
        pipeline: &ComputePipelineState,
        buf_data: &Buffer,
        tw_outer: &[M31],
        tw_inner: &[M31],
        outer_stride: usize,
        inner_stride: usize,
        n: usize,
    ) -> Result<u64, NttError> {
        let tw_o: Vec<u32> = tw_outer.iter().map(|m| m.0).collect();
        let tw_i: Vec<u32> = tw_inner.iter().map(|m| m.0).collect();
        let buf_tw_o = self.buffer_from_slice(&tw_o)?;
        let buf_tw_i = self.buffer_from_slice(&tw_i)?;
        let params: Vec<u32> = vec![outer_stride as u32, inner_stride as u32, n as u32];
        let buf_p = self.buffer_from_slice(&params)?;

        let num_butterflies = (n / 4) as u64;
        let max_tpg = Self::max_threads_per_threadgroup(pipeline) as u64;
        let (tg, tpg) = Self::compute_grid_1d(num_butterflies, max_tpg.min(256));

        self.dispatch_and_wait(pipeline, &[buf_data, &buf_tw_o, &buf_tw_i, &buf_p], tg, tpg)
    }

    /// Dispatch the normalization kernel (multiply all elements by inv_n).
    /// Buffers bound: [data, params].
    pub fn dispatch_normalize(
        &self,
        pipeline: &ComputePipelineState,
        buf_data: &Buffer,
        n: usize,
        inv_n: M31,
    ) -> Result<u64, NttError> {
        let params: Vec<u32> = vec![n as u32, inv_n.0];
        let buf_p = self.buffer_from_slice(&params)?;

        let max_tpg = Self::max_threads_per_threadgroup(pipeline) as u64;
        let (tg, tpg) = Self::compute_grid_1d(n as u64, max_tpg.min(256));

        self.dispatch_and_wait(pipeline, &[buf_data, &buf_p], tg, tpg)
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
///
/// Uses an explicit ordered file list (not directory scan) to ensure
/// deterministic compilation across platforms. Each file gets a marker
/// comment so Metal compiler errors can be traced back to the source file.
fn read_shader_source(shader_dir: &Path) -> Result<String, NttError> {
    // Explicit ordered list: headers first (dependency order), then kernels (alphabetical).
    // This avoids non-deterministic std::fs::read_dir ordering.
    let headers = &["m31_field.metal", "ntt_common.metal"];
    let kernels = &[
        "babybear_field.metal",
        "bb_ntt.metal",
        "ntt_ct_dit_r2.metal",
        "ntt_ct_gs_r2.metal",
        "ntt_ct_gs_r4.metal",
        "ntt_stockham_r2.metal",
        "test_kernels.metal",
    ];

    let mut source = String::new();

    for &name in headers.iter().chain(kernels.iter()) {
        let path = shader_dir.join(name);
        if !path.exists() {
            continue;
        }
        let content = std::fs::read_to_string(&path).map_err(|e| {
            NttError::ShaderCompileError(format!("Failed to read {}: {}", path.display(), e))
        })?;
        // Marker for error diagnostics: Metal compiler errors reference line numbers
        // in the concatenated source; this comment helps trace back to the original file.
        source.push_str(&format!("\n// --- begin {} ---\n", name));
        let filtered: String = content
            .lines()
            .filter(|line| !line.trim_start().starts_with("#include"))
            .collect::<Vec<_>>()
            .join("\n");
        source.push_str(&filtered);
        source.push_str(&format!("\n// --- end {} ---\n", name));
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
