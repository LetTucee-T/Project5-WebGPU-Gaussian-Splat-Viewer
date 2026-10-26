// Optional pass timestamps for the separate performance harness. Normal viewing
// does not request timestamp-query or allocate any profiling resources.
export interface FrameTiming {
  preprocess?: GPUComputePassTimestampWrites;
  sort?: GPUComputePassTimestampWrites;
  render?: GPURenderPassTimestampWrites;
}
