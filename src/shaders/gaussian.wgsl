struct Splat {
    center: vec4<f32>,
    conic_opacity: vec4<f32>,
    color: vec4<f32>,
    extent: vec4<f32>, // Shared 64-byte layout with preprocess.wgsl.
};
struct RenderSettings {
    gaussian_scaling: f32,
    sh_deg: u32,
    culling_enabled: u32,
    display_mode: u32,
};

@group(0) @binding(0)
var<storage, read> splats: array<Splat>;
@group(0) @binding(1)
var<storage, read> visible_indices: array<u32>;
@group(0) @binding(2)
var<uniform> settings: RenderSettings;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) pixel_offset: vec2<f32>,
    @location(1) @interpolate(flat) conic_opacity: vec4<f32>,
    @location(2) @interpolate(flat) color_radius: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> VertexOutput {
    let corners = array<vec2<f32>,6>(
        vec2(-1.,-1.), vec2(1.,-1.), vec2(-1.,1.),
        vec2(-1.,1.), vec2(1.,-1.), vec2(1.,1.)
    );
    let splat = splats[visible_indices[instance]];
    var out: VertexOutput;
    let corner = corners[vertex];
    out.position = vec4(splat.center.xy + corner * splat.extent.xy, splat.center.z, 1.);
    out.pixel_offset = vec2(corner.x,-corner.y)*splat.extent.z;
    out.conic_opacity = splat.conic_opacity;
    out.color_radius = vec4(splat.color.rgb,splat.extent.z);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if settings.display_mode == 1u { return vec4(1.); }
    if settings.display_mode == 2u { return vec4(in.color_radius.rgb,1.); }
    if settings.display_mode == 3u {
        return vec4(clamp(vec3(in.color_radius.w/64.,in.color_radius.w/128.,0.),vec3(0.),vec3(1.)),1.);
    }
    let d = in.pixel_offset;
    if settings.display_mode == 4u { return vec4(0.5+0.5*d/in.color_radius.w,0.,1.); }
    let q = in.conic_opacity;
    let power = -0.5*(q.x*d.x*d.x + 2.*q.y*d.x*d.y + q.z*d.y*d.y);
    if power > 0. { discard; }
    let alpha = min(0.99,q.w*exp(power));
    if alpha < 1./255. { discard; }
    // Premultiplied output, blended with ONE / ONE_MINUS_SRC_ALPHA.
    return vec4(in.color_radius.rgb*alpha,alpha);
}
