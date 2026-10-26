// The host adds `enable f16;` before the shared preprocessing module.
// Only bounded arithmetic uses f16. Camera transforms, exponentials, covariance
// projection/inversion and sort depth retain the FP32 reference implementation.
fn useHalfSH(safe_input: bool) -> bool { return safe_input; }
fn useHalfRotation(scale: vec3<f32>) -> bool {
    let smallest = min(scale.x,min(scale.y,scale.z));
    let largest = max(scale.x,max(scale.y,scale.z));
    // Rotation rounding is amplified by highly anisotropic covariances.
    return smallest > 0. && largest <= 8. * smallest;
}
fn sh_half_scalar(half_index: u32) -> f16 {
    let pair = bitcast<vec2<f16>>(sh_coefficients[half_index / 2u]);
    return pair[half_index % 2u];
}
fn sh_half_coef(idx: u32, coefficient: u32) -> vec3<f16> {
    let base = idx * 48u + coefficient * 3u;
    return vec3<f16>(sh_half_scalar(base),sh_half_scalar(base+1u),sh_half_scalar(base+2u));
}
fn evaluateSH(direction: vec3<f32>, idx: u32, degree: u32, half_precision: bool) -> vec3<f32> {
    if !half_precision { return computeColorFromSH(direction,idx,degree); }
    let dir = vec3<f16>(direction);
    var result = f16(SH_C0) * sh_half_coef(idx,0u);
    if degree > 0u {
        let x=dir.x; let y=dir.y; let z=dir.z;
        result += -f16(SH_C1)*y*sh_half_coef(idx,1u) + f16(SH_C1)*z*sh_half_coef(idx,2u) - f16(SH_C1)*x*sh_half_coef(idx,3u);
        if degree > 1u {
            let xx=x*x; let yy=y*y; let zz=z*z; let xy=x*y; let yz=y*z; let xz=x*z;
            result += f16(SH_C2[0])*xy*sh_half_coef(idx,4u)
                + f16(SH_C2[1])*yz*sh_half_coef(idx,5u)
                + f16(SH_C2[2])*(2.0h*zz-xx-yy)*sh_half_coef(idx,6u)
                + f16(SH_C2[3])*xz*sh_half_coef(idx,7u)
                + f16(SH_C2[4])*(xx-yy)*sh_half_coef(idx,8u);
            if degree > 2u {
                result += f16(SH_C3[0])*y*(3.0h*xx-yy)*sh_half_coef(idx,9u)
                    + f16(SH_C3[1])*xy*z*sh_half_coef(idx,10u)
                    + f16(SH_C3[2])*y*(4.0h*zz-xx-yy)*sh_half_coef(idx,11u)
                    + f16(SH_C3[3])*z*(2.0h*zz-3.0h*xx-3.0h*yy)*sh_half_coef(idx,12u)
                    + f16(SH_C3[4])*x*(4.0h*zz-xx-yy)*sh_half_coef(idx,13u)
                    + f16(SH_C3[5])*z*(xx-yy)*sh_half_coef(idx,14u)
                    + f16(SH_C3[6])*x*(xx-3.0h*yy)*sh_half_coef(idx,15u);
            }
        }
    }
    result += vec3<f16>(0.5h);
    return vec3<f32>(max(vec3<f16>(0.0h),result));
}
fn evaluateRotation(normalized_q: vec4<f32>, half_precision: bool) -> mat3x3<f32> {
    if !half_precision { return rotationFromQuaternion32(normalized_q); }
    let q=vec4<f16>(normalized_q);
    let w=q.x; let x=q.y; let y=q.z; let z=q.w;
    let r=mat3x3<f16>(
        vec3<f16>(1.0h-2.0h*(y*y+z*z),2.0h*(x*y+w*z),2.0h*(x*z-w*y)),
        vec3<f16>(2.0h*(x*y-w*z),1.0h-2.0h*(x*x+z*z),2.0h*(y*z+w*x)),
        vec3<f16>(2.0h*(x*z+w*y),2.0h*(y*z-w*x),1.0h-2.0h*(x*x+y*y))
    );
    return mat3x3<f32>(vec3<f32>(r[0]),vec3<f32>(r[1]),vec3<f32>(r[2]));
}
