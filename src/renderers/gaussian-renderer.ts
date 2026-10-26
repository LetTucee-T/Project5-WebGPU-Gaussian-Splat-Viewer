import { PointCloud } from '../utils/load';
import { preprocessShader, PreprocessPrecision } from './preprocess-shader';
import renderWGSL from '../shaders/gaussian.wgsl';
import { C, c_histogram_block_rows, get_sorter, SortStuff } from '../sort/sort';
import type { Renderer } from './renderer';
import type { FrameTiming } from './timing';

export const PREPROCESS_WORKGROUP_SIZE = 256;
export const SPLAT_STRIDE = 64;
const SORT_KEYS_PER_WORKGROUP = C.histogram_wg_size * c_histogram_block_rows;
export const DISPLAY_MODES = { gaussian: 0, white: 1, color: 2, size: 3, offset: 4 } as const;

export interface GaussianSettings {
  scaling: number; culling: boolean; sorting: boolean;
  mode: keyof typeof DISPLAY_MODES; shDegree: number;
  precision?: PreprocessPrecision;
}
export interface GaussianStatistics {
  retained: number | null;
  frame: number;
  settingsVersion: number;
  error: string;
}
export interface GaussianOptions {
  workgroupSize?: number;
  // The benchmark reads counters outside measured batches instead of polling.
  collectStatistics?: boolean;
}
export interface GaussianRenderer extends Renderer {
  setSettings: (settings: GaussianSettings) => void;
  statistics: () => GaussianStatistics;
  precision: () => {requested: PreprocessPrecision; active: PreprocessPrecision; supported: boolean; shFallbackPoints: number};
  inspect: () => Promise<{
    draw: number[]; sortInfo: number[]; dispatch: number[];
    indices: Uint32Array; depthKeys: Uint32Array; splats: Float32Array;
    frame: number; settings: GaussianSettings; splatStride: number;
  }>;
}

export default async function get_renderer(pc: PointCloud, device: GPUDevice,
  format: GPUTextureFormat, camera_buffer: GPUBuffer, options: GaussianOptions = {}): Promise<GaussianRenderer> {
  const workgroupSize = options.workgroupSize ?? PREPROCESS_WORKGROUP_SIZE;
  if (!Number.isInteger(workgroupSize) || workgroupSize < 1 ||
      workgroupSize > Math.min(device.limits.maxComputeWorkgroupSizeX,device.limits.maxComputeInvocationsPerWorkgroup)) {
    throw new Error('Preprocessing workgroup size exceeds this device\'s limits.');
  }
  const workgroups = Math.ceil(pc.num_points / workgroupSize);
  if (workgroups > device.limits.maxComputeWorkgroupsPerDimension) {
    throw new Error('Scene exceeds the current one-dimensional preprocessing dispatch limit.');
  }
  if (pc.num_points * SPLAT_STRIDE > Math.min(device.limits.maxBufferSize,device.limits.maxStorageBufferBindingSize)) {
    throw new Error('Projected Gaussians exceed this GPU\'s storage limit.');
  }
  const owned: GPUBuffer[] = [];
  let sorter: SortStuff | undefined;
  const transient = new Set<GPUBuffer>();
  let disposed = false;
  const destroy = () => {
    if (disposed) return;
    disposed = true;
    sorter?.destroy();
    for (const buffer of owned) buffer.destroy();
    for (const buffer of transient) buffer.destroy();
    transient.clear();
  };
  function buffer(label: string, size: number, usage: GPUBufferUsageFlags, data?: Uint32Array) {
    const result = device.createBuffer({label, size, usage});
    owned.push(result);
    if (data) device.queue.writeBuffer(result, 0, data);
    return result;
  }
  try {
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const splats = buffer('Projected Gaussian splats (64 bytes each)', pc.num_points * SPLAT_STRIDE, storage);
    sorter = await get_sorter(pc.num_points,device);
    const indices = sorter.ping_pong[0].sort_indices_buffer;
    const depths = sorter.ping_pong[0].sort_depths_buffer;
    const sortInfo = sorter.sort_info_buffer;
    const sortDispatch = sorter.sort_dispatch_indirect_buffer;
    const drawArgs = buffer('Quad indirect draw [6, count, 0, 0]', 16,
      GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, new Uint32Array([6, 0, 0, 0]));
    const settingsBuffer = buffer('Quad render settings', 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const statsReadback = buffer('Quad asynchronous statistics', 48, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);

    const renderModule = device.createShaderModule({label:'Quad rendering', code:renderWGSL});
    const supported = device.features.has('shader-f16');
    const precisions: PreprocessPrecision[] = supported ? ['f32','f16'] : ['f32'];
    const computeModules = precisions.map(precision => device.createShaderModule({
      label:`Gaussian ${precision === 'f16' ? 'FP16/Mixed' : 'FP32'} preprocessing`,code:preprocessShader(precision)}));
    for (const module of [...computeModules, renderModule]) {
      const messages = await module.getCompilationInfo();
      const errors = messages.messages.filter(m => m.type === 'error');
      if (errors.length) throw new Error(errors.map(m => m.message).join('\n'));
    }
    // Explicit shared layouts allow both pipelines to reuse the SAME buffers
    // and bind groups. Auto layouts would be exclusive to each pipeline.
    const groupLayout = (types: GPUBufferBindingType[]) => device.createBindGroupLayout({
      entries:types.map((type,binding)=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type}}))});
    const computeLayouts = [groupLayout(['uniform','uniform']),
      groupLayout(['read-only-storage','storage','read-only-storage']),groupLayout(['storage','storage','storage','storage'])];
    const computeLayout = device.createPipelineLayout({bindGroupLayouts:computeLayouts});
    const pipelines: Partial<Record<PreprocessPrecision,GPUComputePipeline>> = {};
    await Promise.all(precisions.map(async (precision,i) => {
      pipelines[precision] = await device.createComputePipelineAsync({
        label:`Project, shade and compact Gaussians (${precision})`,layout:computeLayout,
        compute:{module:computeModules[i],entryPoint:'preprocess',
          constants:{workgroupSize,sortKeysPerWorkgroup:SORT_KEYS_PER_WORKGROUP}}
      });
    }));
    const renderPipeline = await device.createRenderPipelineAsync({
      label:'Sorted Gaussian splats and debug views', layout:'auto',
      vertex:{module:renderModule, entryPoint:'vs_main'},
      fragment:{module:renderModule, entryPoint:'fs_main', targets:[{format,blend:{
        color:{operation:'add',srcFactor:'one',dstFactor:'one-minus-src-alpha'},
        alpha:{operation:'add',srcFactor:'one',dstFactor:'one-minus-src-alpha'},
      }}]},
      primitive:{topology:'triangle-list', cullMode:'none'},
    });
    const bind = (label: string, layout: GPUBindGroupLayout, buffers: GPUBuffer[]) => device.createBindGroup({
      label, layout, entries:buffers.map((buffer,binding) => ({binding,resource:{buffer}}))
    });
    const cameraGroup = bind('Quad camera and settings', computeLayouts[0], [camera_buffer,settingsBuffer]);
    const dataGroup = bind('Gaussian input, projected output and SH', computeLayouts[1], [pc.gaussian_3d_buffer,splats,pc.sh_buffer]);
    const compactGroup = bind('Quad compacted list', computeLayouts[2], [sortInfo,depths,indices,sortDispatch]);
    const renderGroup = bind('Gaussian render inputs', renderPipeline.getBindGroupLayout(0), [splats,indices,settingsBuffer]);

    let settings: GaussianSettings = {scaling:1,culling:true,sorting:true,mode:'gaussian',shDegree:pc.sh_deg,precision:'f32'};
    let activePrecision: PreprocessPrecision = 'f32';
    let settingsVersion = 0, frameNumber = 0;
    let stats: GaussianStatistics = {retained:null,frame:0,settingsVersion:0,error:''};
    let statsBusy = false, statsQueued = false, nextStatsTime = 0;
    let sampleFrame = 0, sampleVersion = 0;
    const setSettings = (value: GaussianSettings) => {
      if (!Number.isFinite(value.scaling) || value.scaling < 0 || value.scaling > 4) {
        throw new Error('Gaussian scale must be between 0 and 4.');
      }
      if (disposed) return;
      if (!(value.mode in DISPLAY_MODES) || !Number.isInteger(value.shDegree) || value.shDegree < 0 || value.shDegree > 3) {
        throw new Error('Invalid Gaussian display settings.');
      }
      if (value.precision !== undefined && value.precision !== 'f32' && value.precision !== 'f16') throw new Error('Invalid preprocessing precision.');
      settings = {...value,shDegree:Math.min(value.shDegree,pc.sh_deg),precision:value.precision ?? 'f32'};
      activePrecision = settings.precision === 'f16' && supported ? 'f16' : 'f32';
      settingsVersion++;
      const bytes = new ArrayBuffer(16), view = new DataView(bytes);
      view.setFloat32(0,settings.scaling,true);
      view.setUint32(4,settings.shDegree,true);
      view.setUint32(8,settings.culling ? 1 : 0,true);
      view.setUint32(12,DISPLAY_MODES[settings.mode],true);
      device.queue.writeBuffer(settingsBuffer,0,bytes);
      stats = {...stats,retained:null};
      nextStatsTime = 0;
    };
    setSettings(settings);

    function copyHeader(encoder: GPUCommandEncoder, destination: GPUBuffer) {
      encoder.copyBufferToBuffer(drawArgs,0,destination,0,16);
      encoder.copyBufferToBuffer(sortInfo,0,destination,16,20);
      encoder.copyBufferToBuffer(sortDispatch,0,destination,36,12);
    }
    function frame(encoder: GPUCommandEncoder, view: GPUTextureView, timing?: FrameTiming) {
      if (disposed) return;
      frameNumber++;
      // Clear only dynamic counters. Keep padded_size, passes and dispatch Y/Z intact.
      encoder.clearBuffer(sortInfo,0,4);
      encoder.clearBuffer(sortDispatch,0,4);
      const compute = encoder.beginComputePass({label:'Quad project / cull / compact',timestampWrites:timing?.preprocess});
      compute.setPipeline(pipelines[activePrecision]!);
      compute.setBindGroup(0,cameraGroup);
      compute.setBindGroup(1,dataGroup);
      compute.setBindGroup(2,compactGroup);
      compute.dispatchWorkgroups(workgroups);
      compute.end();
      if (settings.sorting) sorter!.sort(encoder,timing?.sort);
      // All workgroups have completed before this ordered GPU copy.
      encoder.copyBufferToBuffer(sortInfo,0,drawArgs,4,4);
      const render = encoder.beginRenderPass({
        label:'Quad indirect rendering',
        timestampWrites:timing?.render,
        colorAttachments:[{view,clearValue:[0,0,0,1],loadOp:'clear',storeOp:'store'}]
      });
      render.setPipeline(renderPipeline);
      render.setBindGroup(0,renderGroup);
      render.drawIndirect(drawArgs,0);
      render.end();
      if (options.collectStatistics !== false && !statsBusy && performance.now() >= nextStatsTime) {
        copyHeader(encoder,statsReadback);
        statsBusy = statsQueued = true;
        sampleFrame = frameNumber;
        sampleVersion = settingsVersion;
        nextStatsTime = performance.now() + 250;
      }
    }
    function afterSubmit() {
      if (!statsQueued || disposed) return;
      statsQueued = false;
      void statsReadback.mapAsync(GPUMapMode.READ).then(() => {
        if (disposed) return;
        const header = new Uint32Array(statsReadback.getMappedRange());
        if (sampleVersion === settingsVersion) {
          stats = {retained:header[1],frame:sampleFrame,settingsVersion:sampleVersion,error:''};
        }
      }).catch(error => {
        if (!disposed) stats = {...stats,error:String(error)};
      }).finally(() => {
        if (!disposed && statsReadback.mapState === 'mapped') statsReadback.unmap();
        statsBusy = false;
      });
    }
    // Development-only callers request a full GPU snapshot; normal rendering never does this.
    async function inspect() {
      if (disposed) throw new Error('Renderer was disposed.');
      const indexOffset = 48, depthOffset = indexOffset + pc.num_points*4;
      const splatOffset = depthOffset + pc.num_points*4;
      const staging = device.createBuffer({label:'Quad validation snapshot',
        size:splatOffset + pc.num_points*SPLAT_STRIDE,usage:GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
      transient.add(staging);
      const capturedFrame = frameNumber, capturedSettings = {...settings};
      try {
        const encoder = device.createCommandEncoder({label:'Read quad validation snapshot'});
        copyHeader(encoder,staging);
        encoder.copyBufferToBuffer(indices,0,staging,indexOffset,pc.num_points*4);
        encoder.copyBufferToBuffer(depths,0,staging,depthOffset,pc.num_points*4);
        encoder.copyBufferToBuffer(splats,0,staging,splatOffset,pc.num_points*SPLAT_STRIDE);
        device.queue.submit([encoder.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        const mapped = staging.getMappedRange(), header = new Uint32Array(mapped,0,12);
        const count = Math.min(header[1],pc.num_points);
        return {draw:Array.from(header.slice(0,4)),sortInfo:Array.from(header.slice(4,9)),dispatch:Array.from(header.slice(9,12)),
          indices:new Uint32Array(mapped,indexOffset,count).slice(),depthKeys:new Uint32Array(mapped,depthOffset,count).slice(),
          splats:new Float32Array(mapped,splatOffset,pc.num_points*SPLAT_STRIDE/4).slice(),
          frame:capturedFrame,settings:capturedSettings,splatStride:SPLAT_STRIDE};
      } finally {
        staging.destroy();
        transient.delete(staging);
      }
    }
    return {frame,afterSubmit,camera_buffer,destroy,setSettings,statistics:() => ({...stats}),inspect,
      precision:() => ({requested:settings.precision!,active:activePrecision,supported,shFallbackPoints:pc.shHalfFallbackPoints})};
  } catch (error) {
    destroy();
    throw error;
  }
}
