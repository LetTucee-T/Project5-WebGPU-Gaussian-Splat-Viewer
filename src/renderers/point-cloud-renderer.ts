import { PointCloud } from '../utils/load';
import pointcloud_wgsl from '../shaders/point_cloud.wgsl';
import { Renderer } from './renderer';
import type { FrameTiming } from './timing';

export default async function get_renderer(
  pc: PointCloud,
  device: GPUDevice,
  presentation_format: GPUTextureFormat,
  camera_buffer: GPUBuffer): Promise<Renderer> {
  const render_shader = device.createShaderModule({label: 'Point cloud MVP', code: pointcloud_wgsl});
  const compilation = await render_shader.getCompilationInfo();
  const errors = compilation.messages.filter(message => message.type === 'error');
  if (errors.length) throw new Error(errors.map(message => message.message).join('\n'));
  const render_pipeline = await device.createRenderPipelineAsync({
    label: 'Point cloud render',
    layout: 'auto',
    vertex: {
      module: render_shader,
      entryPoint: 'vs_main',
    },
    fragment: {
      module: render_shader,
      entryPoint: 'fs_main',
      targets: [{ format: presentation_format }],
    },
    primitive: {
      topology: 'point-list',
    },
  });

  const camera_bind_group = device.createBindGroup({
    label: 'point cloud camera',
    layout: render_pipeline.getBindGroupLayout(0),
    entries: [{binding: 0, resource: { buffer: camera_buffer }}],
  });

  const gaussian_bind_group = device.createBindGroup({
    label: 'point cloud gaussians',
    layout: render_pipeline.getBindGroupLayout(1),
    entries: [
      {binding: 0, resource: { buffer: pc.gaussian_3d_buffer }},
    ],
  });

  const render = (encoder: GPUCommandEncoder, texture_view: GPUTextureView, timing?: FrameTiming) => {
    const pass = encoder.beginRenderPass({
      label: 'point cloud render',
      timestampWrites: timing?.render,
      colorAttachments: [
        {
          view: texture_view,
          clearValue: [0, 0, 0, 1],
          loadOp: 'clear',
          storeOp: 'store',
        }
      ],
    });
    pass.setPipeline(render_pipeline);
    pass.setBindGroup(0, camera_bind_group);
    pass.setBindGroup(1, gaussian_bind_group);

    pass.draw(pc.num_points);
    pass.end();
  };

  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView, timing?: FrameTiming) => {
      render(encoder, texture_view, timing);
    },

    camera_buffer,
    // The application owns the shared input and camera buffers.
    destroy: () => {},
  };
}
