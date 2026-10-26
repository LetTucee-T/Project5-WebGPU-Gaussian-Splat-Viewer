// Precision adapters share the projection / culling / compaction implementation.
fn useHalfSH(safe_input: bool) -> bool { return false; }
fn useHalfRotation(scale: vec3<f32>) -> bool { return false; }
fn evaluateSH(dir: vec3<f32>, idx: u32, degree: u32, half_precision: bool) -> vec3<f32> {
    return computeColorFromSH(dir,idx,degree);
}
fn evaluateRotation(q: vec4<f32>, half_precision: bool) -> mat3x3<f32> {
    return rotationFromQuaternion32(q);
}
