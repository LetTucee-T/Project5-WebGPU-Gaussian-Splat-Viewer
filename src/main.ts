import './style.css';
import init from './renderers/renderer';

(async () => {
  try {
    if (!navigator.gpu) throw new Error('WebGPU is unavailable. Open this viewer in a WebGPU-capable browser over HTTPS or localhost.');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No WebGPU adapter is available.');
    const device = await adapter.requestDevice({
      requiredFeatures: adapter.features.has('shader-f16') ? ['shader-f16'] : [],
      requiredLimits: {
        maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
      },
    });
    const canvas = document.querySelector<HTMLCanvasElement>('#webgpu-canvas')!;
    const context = canvas.getContext('webgpu');
    if (!context) throw new Error('Could not create a WebGPU canvas.');
    const info = adapter.info;
    await init(canvas, context, device, {
      vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description,
      shaderF16Supported: adapter.features.has('shader-f16'), shaderF16Enabled: device.features.has('shader-f16'),
      maxBufferSize: device.limits.maxBufferSize,
      maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
    });
  } catch (error) {
    const status = document.querySelector<HTMLElement>('#status')!;
    status.textContent = error instanceof Error ? error.message : String(error);
    status.dataset.kind = 'error';
    console.error(error);
  }
})();
