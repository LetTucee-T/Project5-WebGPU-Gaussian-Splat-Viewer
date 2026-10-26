import { load, PointCloud, FileData } from '../utils/load';
import { Pane } from 'tweakpane';
import * as FileImport from 'tweakpane-plugin-file-import';
import getPointCloudRenderer from './point-cloud-renderer';
import getGaussianRenderer, { GaussianRenderer, GaussianSettings } from './gaussian-renderer';
import { Camera, CameraPreset, load_camera_presets } from '../camera/camera';
import { CameraControl } from '../camera/camera-control';
import type { FrameTiming } from './timing';

export interface Renderer {
  frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView, timing?: FrameTiming) => void;
  camera_buffer: GPUBuffer;
  afterSubmit?: () => void;
  destroy: () => void;
}

interface SceneRenderers {
  cloud: PointCloud;
  pointcloud: Renderer;
  gaussian: GaussianRenderer;
  destroy: () => void;
}

export default async function init(canvas: HTMLCanvasElement, context: GPUCanvasContext,
  device: GPUDevice, gpuInfo: Record<string, unknown> = {}) {
  const status = document.querySelector<HTMLElement>('#status')!;
  const setStatus = (message: string, error = false) => {
    status.textContent = message;
    status.dataset.kind = error ? 'error' : 'info';
  };
  const camera = new Camera(canvas, device);
  const control = new CameraControl(camera);
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });

  let pc: PointCloud | undefined, renderer: Renderer | undefined;
  let activeScene: SceneRenderers | undefined;
  let cameras: CameraPreset[] = [], sceneName = '';
  let sceneRequest = 0, cameraRequest = 0, loading = 0, frames = 0, currentPreset = 0;
  let stopped = false, needsClear = true, lastError = '';
  let updatingFileControls = false;
  const gpuErrors: string[] = [];
  const params = { renderer: 'gaussian' as 'pointcloud' | 'gaussian', points: 0, cameras: 0, camera: 0, fps: 0,
    scaling: 1, culling: true, sorting: true, display: 'gaussian' as GaussianSettings['mode'], shDegree: 3, retained: '—',
    precision: 'f32' as 'f32' | 'f16',
    ply_file: '' as string | File, cam_file: '' as string | File };
  const currentSettings = (): GaussianSettings => ({scaling:params.scaling,culling:params.culling,
    sorting:params.sorting,mode:params.display,shDegree:params.shDegree,precision:params.precision});
  const applySettings = () => {activeScene?.gaussian.setSettings(currentSettings());if(activeScene) describe();};
  const pane = new Pane({ title: 'Gaussian splats · Precision' });
  pane.registerPlugin(FileImport);
  const modeInput = pane.addInput(params, 'renderer', {
    label: 'Renderer', options: {'Point cloud':'pointcloud','Gaussian splats':'gaussian'}
  }).on('change', () => {
    renderer = activeScene?.[params.renderer];
    params.retained = '—';
    describe();
  });
  modeInput.element.querySelector('select')?.setAttribute('aria-label','Renderer');
  const precisionInput = pane.addInput(params,'precision',{label:'Preprocess',
    options:{'FP32':'f32','FP16 / Mixed':'f16'},disabled:!device.features.has('shader-f16')}).on('change',applySettings);
  precisionInput.element.querySelector('select')?.setAttribute('aria-label','Preprocess precision');
  precisionInput.element.title = device.features.has('shader-f16')
    ? 'Mixed mode uses FP16 SH and bounded rotation arithmetic; projection and sorting retain FP32.'
    : 'This device has no shader-f16 feature; FP32 is used automatically.';
  const displayInput = pane.addInput(params,'display',{label:'Display',options:{
    'Gaussian':'gaussian','White quads':'white','Colored quads':'color','Size heatmap':'size','Pixel offset':'offset',
  }}).on('change',applySettings);
  displayInput.element.querySelector('select')?.setAttribute('aria-label','Display');
  const scaleInput = pane.addInput(params,'scaling',{label:'Gaussian scale',min:0,max:4,step:.25})
    .on('change',applySettings);
  scaleInput.element.querySelector('input')?.setAttribute('aria-label','Gaussian scale');
  const cullInput = pane.addInput(params,'culling',{label:'Frustum cull'})
    .on('change',applySettings);
  cullInput.element.querySelector('input')?.setAttribute('aria-label','Frustum cull');
  const sortingInput = pane.addInput(params,'sorting',{label:'Depth sort'}).on('change',applySettings);
  sortingInput.element.querySelector('input')?.setAttribute('aria-label','Depth sort');
  const shInput = pane.addInput(params,'shDegree',{label:'SH degree',options:{'0':0,'1':1,'2':2,'3':3}}).on('change',applySettings);
  shInput.element.querySelector('select')?.setAttribute('aria-label','SH degree');
  pane.addMonitor(params,'retained',{label:'Retained'});
  pane.addMonitor(params, 'points', { label: 'Points', format: v => v.toLocaleString() });
  pane.addMonitor(params, 'cameras', { label: 'Cameras', format: v => v.toFixed(0) });
  pane.addMonitor(params, 'fps', { label: 'Display FPS', format: v => v.toFixed(0) });
  const presetInput = pane.addInput(params, 'camera', { label: 'Camera index', min: 0, step: 1, disabled: true });
  presetInput.element.querySelector('input')?.setAttribute('aria-label','Camera index');
  const describe = () => setStatus(pc
    ? `${sceneName} · ${pc.num_points.toLocaleString()} points · ${cameras.length ? 'camera ' + currentPreset + ' / ' + (cameras.length - 1) : 'auto-framed view'}${params.renderer === 'gaussian' ? ' · ' + (activeScene?.gaussian.precision().active === 'f16' ? 'FP16 / Mixed' : 'FP32') : ''}${!device.features.has('shader-f16') ? ' · FP16 unavailable' : ''}`
    : 'Load Bonsai, or choose a Gaussian PLY and optional camera JSON.');
  function applyPreset(index: number) {
    if (!Number.isInteger(index) || index < 0 || index >= cameras.length) {
      params.camera = currentPreset;
      pane.refresh();
      setStatus('Camera index is outside the loaded preset range.', true);
      return;
    }
    currentPreset = params.camera = index;
    camera.set_preset(cameras[index]);
    pane.refresh();
    describe();
  }
  presetInput.on('change', e => { if (e.value !== currentPreset) applyPreset(e.value); });
  function installCameras(next: CameraPreset[]) {
    cameras = next;
    currentPreset = params.camera = 0;
    params.cameras = cameras.length;
    presetInput.disabled = cameras.length === 0;
    if (cameras.length) camera.set_preset(cameras[0]);
    else if (pc) camera.frame_bounds(pc.bounds);
  }
  function fail(error: unknown) {
    lastError = error instanceof Error ? error.message : String(error);
    setStatus(lastError, true);
  }
  function resetFileSelection(kind: 'ply' | 'cam') {
    params[kind === 'ply' ? 'ply_file' : 'cam_file'] = '';
    const accept = kind === 'ply' ? '.ply' : '.json';
    const input = pane.element.querySelector<HTMLInputElement>('input[type="file"][accept="' + accept + '"]');
    if (input) input.value = '';
    // A programmatic widget reset must not unload the newly committed scene.
    updatingFileControls = true;
    try { pane.refresh(); } finally { updatingFileControls = false; }
  }
  async function candidate(file: FileData): Promise<SceneRenderers> {
    const cloud = await load(file, device);
    let pointcloud: Renderer | undefined;
    try {
      pointcloud = await getPointCloudRenderer(cloud, device, format, camera.uniform_buffer);
      const gaussian = await getGaussianRenderer(cloud, device, format, camera.uniform_buffer);
      let destroyed = false;
      return {cloud,pointcloud,gaussian,destroy() {
        if (destroyed) return;
        destroyed = true;
        pointcloud.destroy();
        gaussian.destroy();
        cloud.destroy();
      }};
    } catch (error) {
      pointcloud?.destroy();
      cloud.destroy();
      throw error;
    }
  }
  function commit(next: Awaited<ReturnType<typeof candidate>>, name: string) {
    const previous = activeScene;
    activeScene = next;
    pc = next.cloud;
    next.gaussian.setSettings(currentSettings());
    renderer = next[params.renderer];
    sceneName = name;
    params.points = pc.num_points;
    if (cameras.length) camera.set_preset(cameras[currentPreset]);
    else camera.frame_bounds(pc.bounds);
    previous?.destroy();
    lastError = '';
    pane.refresh();
    describe();
  }
  async function loadPly(file: File) {
    const ticket = ++sceneRequest;
    loading++;
    setStatus('Loading ' + file.name + '…');
    try {
      const next = await candidate(file);
      if (ticket !== sceneRequest || stopped) { next.destroy(); return; }
      commit(next, file.name);
    } catch (error) { if (ticket === sceneRequest && !stopped) fail(error); }
    finally { loading--; }
  }
  async function loadCameras(file: File) {
    const ticket = ++cameraRequest;
    loading++;
    try {
      const next = await load_camera_presets(file);
      if (ticket !== cameraRequest || stopped) return;
      installCameras(next);
      lastError = '';
      pane.refresh();
      describe();
    } catch (error) { if (ticket === cameraRequest && !stopped) fail(error); }
    finally { loading--; }
  }
  function clearPly() {
    sceneRequest++;
    activeScene?.destroy();
    activeScene = undefined;
    pc = undefined;
    renderer = undefined;
    sceneName = '';
    params.points = 0;
    params.retained = '—';
    resetFileSelection('ply');
    lastError = '';
    needsClear = true;
    pane.refresh();
    describe();
  }
  function clearCameras() {
    cameraRequest++;
    resetFileSelection('cam');
    installCameras([]);
    lastError = '';
    pane.refresh();
    describe();
  }
  pane.addInput(params, 'ply_file', { label: 'Gaussian PLY', view: 'file-input', lineCount: 2, filetypes: ['.ply'] })
    .on('change', e => { if (updatingFileControls) return; if (e.value instanceof File) void loadPly(e.value); else clearPly(); });
  pane.addInput(params, 'cam_file', { label: 'Camera JSON', view: 'file-input', lineCount: 2, filetypes: ['.json'] })
    .on('change', e => { if (updatingFileControls) return; if (e.value instanceof File) void loadCameras(e.value); else clearCameras(); });
  pane.addButton({ title: 'Load Bonsai' }).on('click', async () => {
    const ticket = ++sceneRequest, cameraTicket = ++cameraRequest;
    loading++;
    setStatus('Loading course Bonsai scene…');
    let next: Awaited<ReturnType<typeof candidate>> | undefined;
    try {
      const base = import.meta.env.BASE_URL + 'scenes/bonsai/';
      const [sceneResponse, cameraResponse] = await Promise.all([fetch(base + 'bonsai.ply'), fetch(base + 'cameras.json')]);
      if (!sceneResponse.ok || !cameraResponse.ok) throw new Error('Bonsai files are missing from public/scenes/bonsai/. You can also choose local files.');
      const presets = await load_camera_presets(await cameraResponse.arrayBuffer());
      next = await candidate(await sceneResponse.arrayBuffer());
      if (ticket !== sceneRequest || cameraTicket !== cameraRequest || stopped) { next.destroy(); return; }
      resetFileSelection('ply');
      resetFileSelection('cam');
      installCameras(presets);
      commit(next, 'Bonsai');
    } catch (error) {
      next?.destroy();
      if (ticket === sceneRequest && !stopped) fail(error);
    } finally { loading--; }
  });
  pane.addButton({ title: 'Reset view' }).on('click', () => {
    if (cameras.length) applyPreset(0);
    else if (pc) { camera.frame_bounds(pc.bounds); describe(); }
  });
  pane.addButton({ title: 'Clear scene' }).on('click', () => { clearPly(); clearCameras(); });

  const resize = () => {
    const limit = device.limits.maxTextureDimension2D;
    canvas.width = Math.max(1, Math.min(limit, Math.round(canvas.clientWidth * devicePixelRatio)));
    canvas.height = Math.max(1, Math.min(limit, Math.round(canvas.clientHeight * devicePixelRatio)));
    camera.on_update_canvas();
    needsClear = true;
  };
  resize();
  const observer = new ResizeObserver(resize);
  observer.observe(canvas);
  const keyHandler = (event: KeyboardEvent) => {
    const target = event.target as HTMLElement | null;
    if (target?.matches('input, textarea, select') || target?.isContentEditable) return;
    if (/^[0-9]$/.test(event.key) && Number(event.key) < cameras.length) applyPreset(Number(event.key));
  };
  document.addEventListener('keydown', keyHandler);
  device.addEventListener('uncapturederror', (event: GPUUncapturedErrorEvent) => {
    gpuErrors.push(event.error.message);
    console.error(event.error.message);
    stopped = true;
    fail(new Error('WebGPU: ' + event.error.message));
  });
  void device.lost.then(info => {
    if (info.reason !== 'destroyed') { stopped = true; fail(new Error('WebGPU device lost: ' + info.message)); }
  });

  // Read-only development snapshots support repeatable camera and lifecycle checks.
  if (import.meta.env.DEV) Object.defineProperty(window, '__gaussianDiagnostics', { configurable: true, value: Object.freeze({
    snapshot: () => ({ stage: 5, renderer: params.renderer, scene: sceneName, points: pc?.num_points ?? 0,
      settings: {...currentSettings(),shDegree:Math.min(params.shDegree,pc?.sh_deg ?? 0)},
      gaussian: activeScene?.gaussian.statistics(),
      precision: activeScene?.gaussian.precision(),
      shDegree: pc?.sh_deg, cameras: cameras.length, preset: currentPreset, frames, loading,
      lastError, gpuErrors: [...gpuErrors], gpuInfo, camera: camera.snapshot() }),
    inspectGpu: () => {
      if (!activeScene) return Promise.reject(new Error('No scene loaded.'));
      return activeScene.gaussian.inspect();
    }
  }) });
  let request = 0, fpsStart = performance.now(), displayFrames = 0;
  function frame(now: number) {
    if (stopped) return;
    if (renderer || needsClear) {
      const encoder = device.createCommandEncoder({ label: 'Viewer frame' });
      const view = context.getCurrentTexture().createView();
      if (renderer) { renderer.frame(encoder, view); frames++; displayFrames++; }
      else {
        const pass = encoder.beginRenderPass({ colorAttachments: [{ view, clearValue: [0, 0, 0, 1], loadOp: 'clear', storeOp: 'store' }] });
        pass.end();
      }
      device.queue.submit([encoder.finish()]);
      renderer?.afterSubmit?.();
      needsClear = false;
    }
    if (now - fpsStart >= 500) {
      params.fps = displayFrames * 1000 / (now - fpsStart);
      params.retained = params.renderer === 'gaussian' && activeScene ? (activeScene.gaussian.statistics().retained?.toLocaleString() ?? '…') : '—';
      displayFrames = 0;
      fpsStart = now;
    }
    request = requestAnimationFrame(frame);
  }
  window.addEventListener('pagehide', () => {
    stopped = true;
    sceneRequest++;
    cameraRequest++;
    cancelAnimationFrame(request);
    observer.disconnect();
    document.removeEventListener('keydown', keyHandler);
    control.dispose();
    activeScene?.destroy();
    camera.uniform_buffer.destroy();
    pane.dispose();
  }, { once: true });
  describe();
  request = requestAnimationFrame(frame);
}
