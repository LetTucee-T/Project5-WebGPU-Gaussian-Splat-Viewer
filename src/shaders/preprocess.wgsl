const SH_C0: f32 = 0.28209479177387814;
const SH_C1 = 0.4886025119029199;
const SH_C2 = array<f32,5>(
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
);
const SH_C3 = array<f32,7>(
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
);

override workgroupSize: u32;
// Sorting capacity is independent of the preprocessing workgroup size.
override sortKeysPerWorkgroup: u32;

struct DispatchIndirect {
    dispatch_x: atomic<u32>,
    dispatch_y: u32,
    dispatch_z: u32,
}

struct SortInfos {
    keys_size: atomic<u32>,  // instance_count in DrawIndirect
    //data below is for info inside radix sort 
    padded_size: u32, 
    passes: u32,
    even_pass: u32,
    odd_pass: u32,
}

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct RenderSettings {
    gaussian_scaling: f32,
    sh_deg: u32,
    culling_enabled: u32,
    display_mode: u32,
}

struct Gaussian {
    pos_opacity: array<u32,2>,
    rot: array<u32,2>,
    scale: array<u32,2>
};

struct Splat {
    center: vec4<f32>,          // NDC xyz, positive view depth. Offset 0.
    conic_opacity: vec4<f32>,   // Inverse pixel covariance (xx,xy,yy), opacity. Offset 16.
    color: vec4<f32>,           // SH RGB; precision flags (SH=1, rotation=2). Offset 32.
    extent: vec4<f32>,          // NDC half width/height, pixel radius, unused. Offset 48.
};

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;
@group(0) @binding(1)
var<uniform> settings: RenderSettings;
@group(1) @binding(0)
var<storage, read> gaussians: array<Gaussian>;
@group(1) @binding(1)
var<storage, read_write> splats: array<Splat>;
@group(1) @binding(2)
var<storage, read> sh_coefficients: array<u32>;
@group(2) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1)
var<storage, read_write> sort_depths : array<u32>;
@group(2) @binding(2)
var<storage, read_write> sort_indices : array<u32>;
@group(2) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;

// load.ts stores 16 RGB coefficients (48 halves / 24 words) per Gaussian.
fn sh_scalar(half_index: u32) -> f32 {
    let pair = unpack2x16float(sh_coefficients[half_index / 2u]);
    return pair[half_index % 2u];
}
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    let base = splat_idx * 48u + c_idx * 3u;
    return vec3(sh_scalar(base), sh_scalar(base + 1u), sh_scalar(base + 2u));
}

// spherical harmonics evaluation with Condon–Shortley phase
fn computeColorFromSH(dir: vec3<f32>, v_idx: u32, sh_deg: u32) -> vec3<f32> {
    var result = SH_C0 * sh_coef(v_idx, 0u);

    if sh_deg > 0u {

        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        result += - SH_C1 * y * sh_coef(v_idx, 1u) + SH_C1 * z * sh_coef(v_idx, 2u) - SH_C1 * x * sh_coef(v_idx, 3u);

        if sh_deg > 1u {

            let xx = dir.x * dir.x;
            let yy = dir.y * dir.y;
            let zz = dir.z * dir.z;
            let xy = dir.x * dir.y;
            let yz = dir.y * dir.z;
            let xz = dir.x * dir.z;

            result += SH_C2[0] * xy * sh_coef(v_idx, 4u) + SH_C2[1] * yz * sh_coef(v_idx, 5u) + SH_C2[2] * (2.0 * zz - xx - yy) * sh_coef(v_idx, 6u) + SH_C2[3] * xz * sh_coef(v_idx, 7u) + SH_C2[4] * (xx - yy) * sh_coef(v_idx, 8u);

            if sh_deg > 2u {
                result += SH_C3[0] * y * (3.0 * xx - yy) * sh_coef(v_idx, 9u) + SH_C3[1] * xy * z * sh_coef(v_idx, 10u) + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_coef(v_idx, 11u) + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coef(v_idx, 12u) + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_coef(v_idx, 13u) + SH_C3[5] * z * (xx - yy) * sh_coef(v_idx, 14u) + SH_C3[6] * x * (xx - 3.0 * yy) * sh_coef(v_idx, 15u);
            }
        }
    }
    result += 0.5;

    return  max(vec3<f32>(0.), result);
}

fn rotationFromQuaternion32(q: vec4<f32>) -> mat3x3<f32> {
    let w=q.x; let x=q.y; let y=q.z; let z=q.w;
    return mat3x3<f32>(
        vec3(1.-2.*(y*y+z*z), 2.*(x*y+w*z), 2.*(x*z-w*y)),
        vec3(2.*(x*y-w*z), 1.-2.*(x*x+z*z), 2.*(y*z+w*x)),
        vec3(2.*(x*z+w*y), 2.*(y*z-w*x), 1.-2.*(x*x+y*y))
    );
}

@compute @workgroup_size(workgroupSize,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= arrayLength(&gaussians) { return; }
    let gaussian = gaussians[idx];
    let xy = unpack2x16float(gaussian.pos_opacity[0]);
    let zo = unpack2x16float(gaussian.pos_opacity[1]);
    let view_position = camera.view * vec4(xy,zo.x,1.);
    let clip = camera.proj * view_position;
    // Retain depth clipping even when optional lateral culling is disabled.
    if !(clip.w > 0. && clip.z >= 0. && clip.z <= clip.w) { return; }
    let ndc = clip.xyz / clip.w;
    if !all(abs(ndc) <= vec3<f32>(3.402823e38)) { return; }
    if settings.gaussian_scaling <= 0. { return; }

    // Stable sigmoid also maps the course data's +/-infinite logits to 1/0.
    let e = exp(-abs(zo.y));
    let opacity = select(e / (1. + e), 1. / (1. + e), zo.y >= 0.);
    if opacity < 1. / 255. { return; }
    let q0 = unpack2x16float(gaussian.rot[0]);
    let q1 = unpack2x16float(gaussian.rot[1]);
    let raw_q = vec4(q0,q1); // PLY order: w,x,y,z.
    let norm2 = dot(raw_q,raw_q);
    if !(norm2 > 1e-12) { return; }
    let q = raw_q * inverseSqrt(norm2);
    let scale_xy = unpack2x16float(gaussian.scale[0]);
    let scale_z_flags = unpack2x16float(gaussian.scale[1]);
    let scale_z = scale_z_flags.x;
    let scale = exp(vec3(scale_xy,scale_z)) * settings.gaussian_scaling;
    let half_rotation = useHalfRotation(scale);
    let rotation = evaluateRotation(q,half_rotation);
    let rs = mat3x3(rotation[0]*scale.x,rotation[1]*scale.y,rotation[2]*scale.z);
    let covariance3d = rs * transpose(rs);
    let view_linear = mat3x3(camera.view[0].xyz,camera.view[1].xyz,camera.view[2].xyz);
    let covariance_view = view_linear * covariance3d * transpose(view_linear);

    // Pixel coordinates point down in Y, matching the camera's flipped NDC Y.
    // Bound the off-axis Jacobian while leaving the projected centre unchanged.
    let inv_z = 1. / view_position.z;
    let limit = 1.3 * camera.viewport / (2. * camera.focal);
    let slope = clamp(view_position.xy * inv_z, -limit, limit);
    let jx = vec3(camera.focal.x*inv_z,0.,-camera.focal.x*slope.x*inv_z);
    let jy = vec3(0.,camera.focal.y*inv_z,-camera.focal.y*slope.y*inv_z);
    // Pixel-space low-pass filter; arithmetic stays f32 in this baseline.
    let a = dot(jx,covariance_view*jx) + 0.3;
    let b = dot(jx,covariance_view*jy);
    let c = dot(jy,covariance_view*jy) + 0.3;
    let det = a*c-b*b;
    if !(det > 0.) { return; }
    let conic = vec3(c,-b,a) / det;
    let half_difference = 0.5*(a-c);
    let lambda_max = 0.5*(a+c) + sqrt(max(0.1,half_difference*half_difference+b*b));
    let radius = ceil(3. * sqrt(lambda_max));
    let half_extent = 2. * vec2(radius) / camera.viewport;
    if !(radius > 0. && radius <= 3.402823e38) || !all(abs(conic) <= vec3(3.402823e38)) { return; }
    if settings.culling_enabled != 0u && any(abs(ndc.xy) > vec2(1.2) + half_extent) { return; }

    let camera_position = camera.view_inv[3].xyz;
    let direction = normalize(vec3(xy,zo.x)-camera_position);
    let half_sh = useHalfSH(scale_z_flags.y > 0.5);
    let color = evaluateSH(direction,idx,min(settings.sh_deg,3u),half_sh);
    let precision_flags = u32(half_sh) | (u32(half_rotation) << 1u);
    // Preserve original identities; sort only the compacted index/key pairs.
    splats[idx] = Splat(vec4(ndc,view_position.z),vec4(conic,opacity),vec4(color,f32(precision_flags)),vec4(half_extent,radius,0.));
    let slot = atomicAdd(&sort_infos.keys_size,1u);
    sort_indices[slot] = idx;
    // Positive finite depths have monotonic IEEE bits. Invert for far-to-near
    // order with the supplied ascending unsigned radix sort (padding stays UINT_MAX).
    sort_depths[slot] = ~bitcast<u32>(view_position.z);
    if slot % sortKeysPerWorkgroup == 0u {
        atomicAdd(&sort_dispatch.dispatch_x,1u);
    }
}
