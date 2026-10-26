// Standalone development experiment; not imported by the shipped viewer.
import { Camera, CameraPreset, load_camera_presets } from './camera/camera';
import { load, PointCloud } from './utils/load';
import gaussianRenderer, { GaussianRenderer } from './renderers/gaussian-renderer';
import pointRenderer from './renderers/point-cloud-renderer';
import type { Renderer } from './renderers/renderer';
import type { FrameTiming } from './renderers/timing';

const BASE = '/output/playwright/stage4/';
const precisionSuite = new URLSearchParams(location.search).get('suite') === 'precision';
const baselineId = precisionSuite ? 'full-overview-f32' : 'wg256';
const WIDTH = 1280, HEIGHT = 720, BATCH = 16, SEED = 20261024;
interface Configuration {
  id: string; renderer: 'gaussian' | 'pointcloud'; points: number;
  camera: 'overview' | 'camera0'; workgroup: number; culling: boolean;
  precision?: 'f32' | 'f16';
}
const configs: Configuration[] = precisionSuite ? [
  {name:'full-overview',points:272956,camera:'overview' as const},
  {name:'full-camera0',points:272956,camera:'camera0' as const},
  {name:'100k-overview',points:100000,camera:'overview' as const},
  {name:'25k-overview',points:25000,camera:'overview' as const},
].flatMap(c => (['f32','f16'] as const).map(precision => ({id:`${c.name}-${precision}`,renderer:'gaussian' as const,
  points:c.points,camera:c.camera,workgroup:256,culling:true,precision}))) : [
  ...[32,64,128,256].map(workgroup => ({id:`wg${workgroup}`,renderer:'gaussian' as const,points:272956,camera:'overview' as const,workgroup,culling:true})),
  {id:'overview-no-cull',renderer:'gaussian',points:272956,camera:'overview',workgroup:256,culling:false},
  {id:'camera0-cull',renderer:'gaussian',points:272956,camera:'camera0',workgroup:256,culling:true},
  {id:'camera0-no-cull',renderer:'gaussian',points:272956,camera:'camera0',workgroup:256,culling:false},
  ...[25000,50000,100000,200000].map(points => ({id:`n${points}`,renderer:'gaussian' as const,points,camera:'overview' as const,workgroup:256,culling:true})),
  {id:'pointcloud',renderer:'pointcloud',points:272956,camera:'overview',workgroup:256,culling:true},
];
const canvas = document.querySelector<HTMLCanvasElement>('#result')!;
const status = document.querySelector<HTMLElement>('#status')!;
const errors: string[] = [];
let adapter: GPUAdapter, device: GPUDevice, camera: Camera, texture: GPUTexture;
let querySet: GPUQuerySet, resolve: GPUBuffer, readback: GPUBuffer;
let current: {config: Configuration; renderer: Renderer} | undefined;
const inputs = new Map<number, PointCloud>();
const presets = new Map<string, CameraPreset>();
const cameras: Record<string, unknown> = {};
let phase = 'initializing', progress = '', busy = false, disposed = false;
let result: any;
const checks: any[] = [];
const validation: Record<string, any> = {};
const images = new Map<string, Uint8ClampedArray>();
const format: GPUTextureFormat = 'bgra8unorm';
function announce(message: string) { progress = message; status.textContent = `${phase}: ${message}`; }
function assertHealthy() { if (errors.length) throw new Error(errors.join('\n')); }
function isGaussian(renderer: Renderer): renderer is GaussianRenderer { return 'inspect' in renderer; }
async function fetched(url: string) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  return response.arrayBuffer();
}
function dispose() {
  if (disposed) return;
  disposed = true;
  current?.renderer.destroy();
  for (const pc of inputs.values()) pc.destroy();
  camera?.uniform_buffer.destroy(); texture?.destroy();
  querySet?.destroy(); resolve?.destroy(); readback?.destroy(); device?.destroy();
}
window.addEventListener('pagehide',dispose);
window.addEventListener('error',e => errors.push(e.message));
window.addEventListener('unhandledrejection',e => errors.push(String(e.reason)));

async function initialize() {
  if (!import.meta.env.DEV) throw new Error('Performance harness is development-only.');
  adapter = (await navigator.gpu?.requestAdapter())!;
  if (!adapter || !adapter.features.has('timestamp-query')) throw new Error('This adapter does not support GPU timestamp-query.');
  if(precisionSuite && !adapter.features.has('shader-f16'))throw new Error('Precision benchmark requires shader-f16.');
  device = await adapter.requestDevice({requiredFeatures:precisionSuite?['timestamp-query','shader-f16']:['timestamp-query'],requiredLimits:{
    maxComputeWorkgroupStorageSize:adapter.limits.maxComputeWorkgroupStorageSize,
    maxBufferSize:adapter.limits.maxBufferSize,maxStorageBufferBindingSize:adapter.limits.maxStorageBufferBindingSize,
  }});
  device.addEventListener('uncapturederror',e => errors.push((e as GPUUncapturedErrorEvent).error.message));
  void device.lost.then(info => { if (!disposed) errors.push(`Device lost: ${info.reason}: ${info.message}`); });
  camera = new Camera(canvas,device);
  texture = device.createTexture({label:'Fixed 1280x720 benchmark target',size:[WIDTH,HEIGHT],format,
    usage:GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC});
  querySet = device.createQuerySet({type:'timestamp',count:BATCH*6});
  resolve = device.createBuffer({size:BATCH*6*8,usage:GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC});
  readback = device.createBuffer({size:BATCH*6*8,usage:GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
  for (const name of ['overview','camera0']) {
    presets.set(name,(await load_camera_presets(await fetched(`${BASE}fixtures/${name}.json`)))[0]);
    camera.set_preset(presets.get(name)!);
    cameras[name] = camera.snapshot();
  }
  phase = 'ready'; announce('已就绪；当前没有持续渲染。');
}

async function select(config: Configuration) {
  assertHealthy();
  if (current?.config.id === config.id) return current.renderer;
  current?.renderer.destroy(); current = undefined;
  let pc = inputs.get(config.points);
  if (!pc) {
    announce(`加载 ${config.points.toLocaleString()} 个点（不计入耗时）`);
    pc = await load(await fetched(config.points === 272956 ? '/scenes/bonsai/bonsai.ply' : `${BASE}fixtures/bonsai-${config.points}.ply`),device);
    inputs.set(config.points,pc);
  }
  camera.set_preset(presets.get(config.camera)!);
  const renderer = config.renderer === 'gaussian'
    ? await gaussianRenderer(pc,device,format,camera.uniform_buffer,{workgroupSize:config.workgroup,collectStatistics:false})
    : await pointRenderer(pc,device,format,camera.uniform_buffer);
  if (isGaussian(renderer)) renderer.setSettings({scaling:1,culling:config.culling,sorting:true,mode:'gaussian',shDegree:3,precision:config.precision??'f32'});
  current = {config,renderer};
  return renderer;
}

async function warmup(renderer: Renderer, frames: number) {
  for (let i=0;i<frames;i+=BATCH) {
    const encoder = device.createCommandEncoder();
    for (let j=i;j<Math.min(frames,i+BATCH);++j) renderer.frame(encoder,texture.createView());
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
  }
  assertHealthy();
}
function pair(offset: number): GPUComputePassTimestampWrites {
  return {querySet,beginningOfPassWriteIndex:offset,endOfPassWriteIndex:offset+1};
}
async function measuredBatch(renderer: Renderer, count: number, boundaryOnly = false) {
  const encoder = device.createCommandEncoder();
  const cpuBegin = performance.now();
  for (let i=0;i<count;++i) {
    let timing: FrameTiming;
    if (boundaryOnly) {
      timing = {};
      if (i === 0) timing[isGaussian(renderer) ? 'preprocess' : 'render'] = {querySet,beginningOfPassWriteIndex:0};
      if (i === count-1) timing.render = {...timing.render,querySet,endOfPassWriteIndex:1};
    } else timing = {preprocess:pair(i*6),sort:pair(i*6+2),render:pair(i*6+4)};
    renderer.frame(encoder,texture.createView(),timing);
  }
  encoder.resolveQuerySet(querySet,0,boundaryOnly ? 2 : count*6,resolve,0);
  encoder.copyBufferToBuffer(resolve,0,readback,0,boundaryOnly ? 16 : count*6*8);
  device.queue.submit([encoder.finish()]);
  const cpuSubmitMs = performance.now()-cpuBegin;
  await readback.mapAsync(GPUMapMode.READ);
  const elapsedMs = performance.now()-cpuBegin;
  const stamps = new BigUint64Array(readback.getMappedRange()).slice(0,boundaryOnly ? 2 : count*6);
  readback.unmap();
  assertHealthy();
  if (boundaryOnly) {
    if (stamps[1] <= stamps[0]) throw new Error('Invalid boundary timestamp sample.');
    return {rows:[],cpuSubmitMs,elapsedMs,boundaryMs:Number(stamps[1]-stamps[0])/1e6/count};
  }
  const rows = [];
  for (let i=0;i<count;++i) {
    const t = stamps.slice(i*6,i*6+6);
    const gaussian = isGaussian(renderer);
    const active = gaussian ? Array.from(t) : Array.from(t.slice(4));
    if (active.some((value,index) => index && value < active[index-1])) throw new Error('Non-monotonic GPU timestamps; discard this run and repeat.');
    const ms = (a: number,b: number) => Number(t[b]-t[a])/1e6;
    rows.push({preprocessMs:gaussian ? ms(0,1) : 0,sortMs:gaussian ? ms(2,3) : 0,renderMs:ms(4,5),
      totalMs:gaussian ? ms(0,5) : ms(4,5),timestampsNs:Array.from(t,String)});
  }
  return {rows,cpuSubmitMs,elapsedMs,boundaryMs:null};
}

async function pixels() {
  const stride = WIDTH*4;
  const staging = device.createBuffer({size:stride*HEIGHT,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer({texture},{buffer:staging,bytesPerRow:stride,rowsPerImage:HEIGHT},[WIDTH,HEIGHT]);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const bgra = new Uint8Array(staging.getMappedRange());
    const rgba = new Uint8ClampedArray(bgra.length);
    for (let i=0;i<bgra.length;i+=4) { rgba[i]=bgra[i+2];rgba[i+1]=bgra[i+1];rgba[i+2]=bgra[i];rgba[i+3]=255; }
    return rgba;
  } finally { staging.destroy(); }
}
async function digest(bytes: Uint8Array) {
  const hash = new Uint8Array(await crypto.subtle.digest('SHA-256',bytes));
  return Array.from(hash,v=>v.toString(16).padStart(2,'0')).join('');
}
function difference(a: Uint8ClampedArray,b: Uint8ClampedArray) {
  let maximum=0,total=0,changed=0,square=0;
  for (let i=0;i<a.length;i+=4) for(let c=0;c<3;++c) {
    const delta=Math.abs(a[i+c]-b[i+c]); maximum=Math.max(maximum,delta);total+=delta;square+=delta*delta;if(delta)changed++;
  }
  const rmse=Math.sqrt(square/(WIDTH*HEIGHT*3));
  return {maxChannelDifference:maximum,meanChannelDifference:total/(WIDTH*HEIGHT*3),changedChannels:changed,psnr:rmse?20*Math.log10(255/rmse):null};
}
async function validate(config: Configuration, renderer: Renderer) {
  await warmup(renderer,1);
  const before = await pixels();
  await measuredBatch(renderer,1);
  const after = await pixels();
  const instrumentationDifference = difference(before,after);
  // Atomic compaction can reorder equal-depth splats by a few 8-bit levels.
  checks.push({name:`${config.id}: timestamps preserve image`,passed:instrumentationDifference.maxChannelDifference <= 2,...instrumentationDifference});
  images.set(config.id,after);
  let retained = config.points, indicesSha256 = '', sorted = true, unique = true;
  if (isGaussian(renderer)) {
    const snapshot = await renderer.inspect();
    retained=snapshot.draw[1];
    for (let i=1;i<snapshot.depthKeys.length;++i) if(snapshot.depthKeys[i] < snapshot.depthKeys[i-1]) sorted=false;
    const ordered=snapshot.indices.slice().sort();
    for (let i=1;i<ordered.length;++i) if (ordered[i]===ordered[i-1]) unique=false;
    indicesSha256=await digest(new Uint8Array(ordered.buffer));
    checks.push({name:`${config.id}: valid sorted indirect draw`,passed:sorted&&unique&&snapshot.draw[0]===6&&retained<=config.points,
      retained,sorted,unique,dispatch:snapshot.dispatch});
  }
  const item = {config,retained,indicesSha256,instrumentationDifference,camera:camera.snapshot(),precision:isGaussian(renderer)?renderer.precision():null};
  validation[config.id]=item;
  canvas.getContext('2d')!.putImageData(new ImageData(after,WIDTH,HEIGHT),0,0);
  return item;
}

async function runOne(config: Configuration, round: number, frames: number, warmupFrames: number) {
  const renderer = await select(config);
  announce(`第 ${round+1} 轮 / ${config.id}：预热`);
  await warmup(renderer,warmupFrames);
  const startUtc=new Date().toISOString();
  const batches=[];
  for(let offset=0;offset<frames;offset+=BATCH) {
    const sample=await measuredBatch(renderer,Math.min(BATCH,frames-offset));
    batches.push({...sample,offset});
  }
  return {config:config.id,round,startUtc,endUtc:new Date().toISOString(),frames,batches};
}
function shuffle<T>(source: T[], seed: number) {
  const a=source.slice();let state=seed>>>0;
  const random=()=> {state^=state<<13;state^=state>>>17;state^=state<<5;return (state>>>0)/4294967296;};
  for(let i=a.length-1;i>0;--i){const j=Math.floor(random()*(i+1));[a[i],a[j]]=[a[j],a[i]];}
  return a;
}
async function validateAll() {
  phase='validation';
  checks.length=0;images.clear();
  for(const key of Object.keys(validation)) delete validation[key];
  for (const config of configs) await validate(config,await select(config));
  for(const id of precisionSuite?[]:['wg32','wg64','wg128']) {
    const diff=difference(images.get(id)!,images.get('wg256')!);
    checks.push({name:`${id}: same retained set and image as wg256`,passed:validation[id].indicesSha256===validation.wg256.indicesSha256&&diff.maxChannelDifference<=2,...diff});
  }
  for(const [a,b] of precisionSuite?[]:[['wg256','overview-no-cull'],['camera0-cull','camera0-no-cull']]) {
    const diff=difference(images.get(a)!,images.get(b)!);
    checks.push({name:`${a}: culling preserves image`,passed:diff.maxChannelDifference<=2,...diff});
  }
  if(precisionSuite)for(let i=0;i<configs.length;i+=2){
    const a=configs[i].id,b=configs[i+1].id,diff=difference(images.get(a)!,images.get(b)!);
    checks.push({name:`${a} vs ${b}: quality and active precision`,passed:(diff.psnr===null||diff.psnr>=40)&&validation[a].precision.active==='f32'&&validation[b].precision.active==='f16',...diff});
  }
  // Verify default path without optional timing/WG/statistics arguments.
  const pc=inputs.get(272956)!;
  camera.set_preset(presets.get('overview')!);
  const normal=await gaussianRenderer(pc,device,format,camera.uniform_buffer);
  await warmup(normal,2);normal.afterSubmit?.();
  await device.queue.onSubmittedWorkDone();
  const diff=difference(await pixels(),images.get(baselineId)!);
  checks.push({name:'Default viewer renderer matches benchmark baseline',passed:diff.maxChannelDifference<=2,...diff});
  normal.destroy();
  assertHealthy();
  if (checks.some(c=>!c.passed)) throw new Error('Benchmark validation failed.');
}
async function runAll(options: {rounds?: number;frames?: number;warmup?: number;validate?: boolean} = {}) {
  if(busy)throw new Error('An experiment is already running.');
  busy=true;
  const rounds=options.rounds??5, frames=options.frames??192,warmupFrames=options.warmup??64;
  try {
    await ready;
    if (options.validate !== false) await validateAll();
    phase='conditioning';
    announce('使用完整场景持续预热 3 秒，让 GPU 退出空闲频率');
    const conditioner=await select(configs.find(c=>c.id===baselineId)!);
    const conditioningStart=performance.now();
    let conditioningFrames=0;
    do {await warmup(conditioner,64);conditioningFrames+=64;} while(performance.now()-conditioningStart<3000);
    phase='running';
    const info=(adapter as any).info;
    result={schema:1,suite:precisionSuite?'precision':'basic',startedUtc:new Date().toISOString(),method:{width:WIDTH,height:HEIGHT,format,batchSize:BATCH,rounds,framesPerRun:frames,
      warmupFrames,conditioningFrames,conditioningWallMs:performance.now()-conditioningStart,seed:SEED,settings:{scale:1,sorting:true,shDegree:3},sortWorkgroup:256,
      timestamps:'GPU pass begin/end, nanoseconds; no browser precision flags; raw stamps retained',
      total:'preprocess begin to render end (point cloud: render only); excludes initial counter clears and final readback',
      readbacks:'one query resolve and map per batch, outside measured pass interval; no counter polling',
      target:'offscreen GPU texture, identical format/size/pipelines to viewer; no requestAnimationFrame or vsync',
      order:precisionSuite?'paired scenes, seeded scene permutation; alternate FP32/FP16 order each round':'seeded permutation, rotate by 5 positions each round; every configuration once per round',
      wallTime:'batch encoding + submit + query readback wait, not interactive application FPS'},
      environment:{userAgent:navigator.userAgent,adapter:info?{vendor:info.vendor,architecture:info.architecture,device:info.device,description:info.description}:null,
        adapterFeatures:[...adapter.features],deviceFeatures:[...device.features],limits:{maxComputeInvocationsPerWorkgroup:device.limits.maxComputeInvocationsPerWorkgroup,
          maxComputeWorkgroupSizeX:device.limits.maxComputeWorkgroupSizeX,maxComputeWorkgroupStorageSize:device.limits.maxComputeWorkgroupStorageSize}},
      configs,cameras,validation,checks,runs:[],boundaryControls:[],errors};
    const order=shuffle(configs,SEED);
    for(let round=0;round<rounds;++round) {
      const paired=precisionSuite?shuffle([0,2,4,6],SEED+round).flatMap((index,i)=>(round+i)%2?[configs[index+1],configs[index]]:[configs[index],configs[index+1]]):[];
      for(let i=0;i<order.length;++i) {
        const config=precisionSuite?paired[i]:order[(i+round*5)%order.length];
        result.runs.push(await runOne(config,round,frames,warmupFrames));
        announce(`已完成 ${result.runs.length}/${rounds*configs.length} 组`);
      }
      // Measure only outer timestamps to expose the effect of detailed profiling.
      for(const id of precisionSuite?['full-overview-f32','full-overview-f16']:['wg256','pointcloud']) {
        const renderer=await select(configs.find(c=>c.id===id)!);
        await warmup(renderer,warmupFrames);
        const batches=[];
        for(let i=0;i<frames;i+=BATCH) batches.push(await measuredBatch(renderer,Math.min(BATCH,frames-i),true));
        result.boundaryControls.push({config:id,round,batches});
      }
    }
    result.finishedUtc=new Date().toISOString();assertHealthy();
    phase='complete';announce(`${result.runs.length} 组实验完成，共 ${result.runs.length*frames} 帧 GPU 样本。`);
    document.querySelector<HTMLButtonElement>('#download')!.disabled=false;
    return {runs:result.runs.length,checks:checks.length,errors};
  } catch(error) {phase='failed';announce(String(error));throw error;}
  finally {busy=false;}
}

const ready=initialize().catch(error=>{phase='failed';announce(String(error));throw error;});
const api={ready:()=>ready,state:()=>({phase,progress,busy,runs:result?.runs.length??0,errors}),configs,
  runAll, result:()=>result, validateAll,
  show:async(id:string)=>{await ready;const config=configs.find(c=>c.id===id);if(!config)throw new Error(id);
    const renderer=await select(config);await warmup(renderer,1);canvas.getContext('2d')!.putImageData(new ImageData(await pixels(),WIDTH,HEIGHT),0,0);return validation[id];},
  validation:()=>({checks,validation,errors}),dispose};
(window as any).__gaussianBenchmark=api;
document.querySelector('#start')!.addEventListener('click',()=>{void runAll().catch(console.error);});
document.querySelector('#download')!.addEventListener('click',()=>{
  const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));
  const a=document.createElement('a');a.href=url;a.download='performance-raw.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
