// Development-only, deterministic validation of the actual production renderer.
import { Camera, load_camera_presets, CameraPreset } from './camera/camera';
import { load, PointCloud } from './utils/load';
import getRenderer, { GaussianRenderer, GaussianSettings } from './renderers/gaussian-renderer';
import type { PreprocessPrecision } from './renderers/preprocess-shader';

const ROOT='/output/playwright/';
type Case = {name:string;file:string;reference:string;camera:string;scale?:number;degree?:number;culling?:boolean;sorting?:boolean;size?:[number,number];capture?:boolean;gpu?:boolean};
const cases: Case[]=[];
function synthetic(name:string,reference=name,extra:Partial<Case>={}) {
  cases.push({name,file:`${ROOT}stage3/fixtures/${reference}.ply`,reference:`stage3/${reference}`,camera:'identity',...extra});
}
synthetic('shapes','shapes',{capture:true,gpu:true});
for(const scale of [0,.25,2,4])synthetic(`shapes-scale-${scale}`,'shapes',{scale,gpu:true});
synthetic('portrait','shapes',{size:[800,900],gpu:true});
for(const name of ['endpoints','overlap','culling'])synthetic(name,name,{capture:true,gpu:true});
synthetic('culling-off','culling',{culling:false,gpu:true});
synthetic('unsorted','overlap',{sorting:false,gpu:true});
for(const n of [1,255,256,257,3839,3840,3841,7681])synthetic(`count-${n}`,`count-${n}`);
synthetic('all-invisible','count-7681',{camera:'away'});
for(const degree of [0,1,2,3])synthetic(`sh-degree-${degree}`,'sh',{degree,gpu:true});
synthetic('sh-translated','sh',{camera:'translated',gpu:true});
for(const name of ['guards','near-plane','stress-257'])synthetic(name,name,{file:`${ROOT}stage5/fixtures/${name}.ply`,reference:`stage5/${name}`,capture:true,gpu:true});
for(const camera of ['overview','camera0','camera2','camera5'])cases.push({name:`bonsai-${camera}`,file:'/scenes/bonsai/bonsai.ply',reference:'stage3/bonsai',camera,capture:true,gpu:camera==='overview'||camera==='camera0'});
for(const scale of [.5,2])cases.push({name:`bonsai-scale-${scale}`,file:'/scenes/bonsai/bonsai.ply',reference:'stage3/bonsai',camera:'overview',scale,capture:true});
for(const degree of [0,1,2])cases.push({name:`bonsai-sh-${degree}`,file:'/scenes/bonsai/bonsai.ply',reference:'stage3/bonsai',camera:'overview',degree});
cases.push({name:'bonsai-no-cull',file:'/scenes/bonsai/bonsai.ply',reference:'stage3/bonsai',camera:'overview',culling:false});

const canvas=document.querySelector<HTMLCanvasElement>('#result')!;
const status=document.querySelector('#status')!;
const files=new Map<string,Blob>();
const errors:string[]=[];
const presets=new Map<string,CameraPreset>();
let adapter:GPUAdapter,device:GPUDevice,camera:Camera,texture:GPUTexture;
let current:{file:string;pc:PointCloud;renderer:GaussianRenderer}|undefined;
let busy=false,phase='initializing',progress='',report:any,disposed=false;
const update=(text:string)=>{progress=text;status.textContent=text;};
function check(name:string,passed:boolean,details:any={}) {report.checks.push({name,passed,...details});}
async function fetched(url:string) {const r=await fetch(url);if(!r.ok)throw new Error(`${url}: ${r.status}`);return r.arrayBuffer();}
function target(width:number,height:number) {
  texture?.destroy();canvas.width=width;canvas.height=height;
  canvas.style.width=`${width}px`;canvas.style.height=`${height}px`;
  texture=device.createTexture({size:[width,height],format:'bgra8unorm',usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC});
}
function destroy() {
  if(disposed)return;disposed=true;current?.renderer.destroy();current?.pc.destroy();
  texture?.destroy();camera?.uniform_buffer.destroy();device?.destroy();
}
window.addEventListener('pagehide',destroy);
window.addEventListener('error',e=>errors.push(e.message));
window.addEventListener('unhandledrejection',e=>errors.push(String(e.reason)));
async function initialize() {
  if(!import.meta.env.DEV)throw new Error('Validation is development-only.');
  adapter=(await navigator.gpu.requestAdapter())!;
  if(!adapter?.features.has('shader-f16'))throw new Error('Full FP16 validation requires shader-f16 support.');
  device=await adapter.requestDevice({requiredFeatures:['shader-f16'],requiredLimits:{
    maxComputeWorkgroupStorageSize:adapter.limits.maxComputeWorkgroupStorageSize,
    maxBufferSize:adapter.limits.maxBufferSize,maxStorageBufferBindingSize:adapter.limits.maxStorageBufferBindingSize}});
  device.addEventListener('uncapturederror',e=>errors.push((e as GPUUncapturedErrorEvent).error.message));
  void device.lost.then(i=>{if(!disposed)errors.push(`device lost: ${i.message}`);});
  camera=new Camera(canvas,device);target(1280,720);
  const generic=await load_camera_presets(await fetched(`${ROOT}stage3/fixtures/cameras.json`));
  ['identity','away','translated'].forEach((n,i)=>presets.set(n,generic[i]));
  presets.set('overview',(await load_camera_presets(await fetched(`${ROOT}stage3/overview-preset.json`)))[0]);
  const real=await load_camera_presets(await fetched('/scenes/bonsai/cameras.json'));
  for(const i of [0,2,5])presets.set(`camera${i}`,real[i]);
  phase='ready';update('已就绪，等待开始。');
}
async function select(c:Case) {
  if(current?.file!==c.file) {
    current?.renderer.destroy();current?.pc.destroy();current=undefined;
    const pc=await load(await fetched(c.file),device);
    try {current={file:c.file,pc,renderer:await getRenderer(pc,device,'bgra8unorm',camera.uniform_buffer,{collectStatistics:false})};}
    catch(e){pc.destroy();throw e;}
  }
  const size=c.size??[1280,720];if(canvas.width!==size[0]||canvas.height!==size[1])target(...size);
  camera.set_preset(presets.get(c.camera)!);
  return current!;
}
async function readPixels() {
  const stride=Math.ceil(canvas.width*4/256)*256;
  const buffer=device.createBuffer({size:stride*canvas.height,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try {
    const enc=device.createCommandEncoder();enc.copyTextureToBuffer({texture},{buffer,bytesPerRow:stride},[canvas.width,canvas.height]);
    device.queue.submit([enc.finish()]);await buffer.mapAsync(GPUMapMode.READ);
    const bytes=new Uint8Array(buffer.getMappedRange()),rgba=new Uint8ClampedArray(canvas.width*canvas.height*4);
    for(let y=0;y<canvas.height;++y)for(let x=0;x<canvas.width;++x){const a=y*stride+x*4,b=(y*canvas.width+x)*4;
      rgba[b]=bytes[a+2];rgba[b+1]=bytes[a+1];rgba[b+2]=bytes[a];rgba[b+3]=255;}
    return rgba;
  } finally {buffer.destroy();}
}
async function runMode(c:Case,precision:PreprocessPrecision) {
  const {renderer,pc}=current!;
  const settings:GaussianSettings={scaling:c.scale??1,culling:c.culling??true,sorting:c.sorting??true,mode:'gaussian',shDegree:c.degree??3,precision};
  renderer.setSettings(settings);
  for(let i=0;i<2;++i){const enc=device.createCommandEncoder();renderer.frame(enc,texture.createView());device.queue.submit([enc.finish()]);await device.queue.onSubmittedWorkDone();}
  const gpu=await renderer.inspect();const pixels=await readPixels();
  let finite=true,sorted=true,unique=true,halfSH=0,halfRotation=0;
  const ordered=gpu.indices.slice().sort();
  for(let i=0;i<gpu.indices.length;++i){
    if(i&&gpu.depthKeys[i]<gpu.depthKeys[i-1])sorted=false;
    if(i&&ordered[i]===ordered[i-1])unique=false;
    const id=gpu.indices[i];if(id>=pc.num_points)finite=false;
    for(let k=0;k<16;++k)if(!Number.isFinite(gpu.splats[id*16+k]))finite=false;
    const flags=gpu.splats[id*16+11];if(flags&1)halfSH++;if(flags&2)halfRotation++;
  }
  const metadata={count:gpu.indices.length,total:pc.num_points,settings:gpu.settings,camera:camera.snapshot(),precision:renderer.precision(),
    draw:gpu.draw,dispatch:gpu.dispatch,finite,sorted,unique,halfSH,halfRotation};
  check(`${c.name}/${precision}: GPU contracts`,finite&&unique&&(!settings.sorting||sorted)&&gpu.draw[0]===6&&gpu.draw[1]===gpu.indices.length&&gpu.dispatch[0]===Math.ceil(gpu.indices.length/3840),metadata);
  if(precision==='f32')check(`${c.name}: reference uses FP32`,halfSH===0&&halfRotation===0);
  if(c.capture){
    canvas.getContext('2d')!.putImageData(new ImageData(pixels,canvas.width,canvas.height),0,0);
    const blob=await new Promise<Blob>(resolve=>canvas.toBlob(b=>resolve(b!),'image/png'));
    files.set(`evidence/${c.name}-${precision}.png`,blob);
  }
  if(c.gpu){
    const blob=new Blob([gpu.indices,gpu.depthKeys,gpu.splats]);
    const compressed=await new Response(blob.stream().pipeThrough(new CompressionStream('gzip'))).blob();
    files.set(`gpu/${c.name}-${precision}.bin.gz`,compressed);
  }
  return {gpu,pixels,metadata,ordered};
}
function compare(a:Awaited<ReturnType<typeof runMode>>,b:Awaited<ReturnType<typeof runMode>>) {
  let square=0,absolute=0,maximum=0,changed=0;
  for(let i=0;i<a.pixels.length;i+=4)for(let k=0;k<3;++k){const delta=Math.abs(a.pixels[i+k]-b.pixels[i+k]);square+=delta*delta;absolute+=delta;maximum=Math.max(maximum,delta);if(delta>2)changed++;}
  const channels=a.pixels.length/4*3,rmse=Math.sqrt(square/channels);
  let missing=0,extra=0,maxCenterError=0,maxColorError=0,maxConicError=0,maxRadiusError=0,depthIdentity=true;
  const aIds=new Set(a.gpu.indices),bIds=new Set(b.gpu.indices);
  for(const id of aIds){if(!bIds.has(id)){missing++;continue;}
    for(let k=0;k<4;++k)maxCenterError=Math.max(maxCenterError,Math.abs(a.gpu.splats[id*16+k]-b.gpu.splats[id*16+k]));
    if(a.gpu.splats[id*16+3]!==b.gpu.splats[id*16+3])depthIdentity=false;
    for(let k=8;k<11;++k)maxColorError=Math.max(maxColorError,Math.abs(a.gpu.splats[id*16+k]-b.gpu.splats[id*16+k]));
    for(let k=4;k<7;++k)maxConicError=Math.max(maxConicError,Math.abs(a.gpu.splats[id*16+k]-b.gpu.splats[id*16+k]));
    maxRadiusError=Math.max(maxRadiusError,Math.abs(a.gpu.splats[id*16+14]-b.gpu.splats[id*16+14]));
  }
  for(const id of bIds)if(!aIds.has(id))extra++;
  return {meanAbsolute8bit:absolute/channels,rmse8bit:rmse,psnr:rmse===0?null:20*Math.log10(255/rmse),exact:rmse===0,
    max8bit:maximum,fractionChannelsAbove2:changed/channels,missing,extra,maxCenterError,maxColorError,maxConicError,maxRadiusError,depthIdentity};
}
async function runAll() {
  if(busy)throw new Error('Validation already running.');busy=true;files.clear();
  try {
    await ready;phase='running';
    report={stage:5,started:new Date().toISOString(),userAgent:navigator.userAgent,features:[...device.features],
      method:{pixelTargetPSNR:40,outputStride:64,rotationRatioLimit:8,shCoefficientLimit:256,reference:'FP32 same renderer; saved captures also checked against independent NumPy reference'},
      cases:[],checks:[],errors};
    for(const c of cases){
      update(`${report.cases.length+1}/${cases.length} · ${c.name}`);await select(c);
      const full=await runMode(c,'f32'),mixed=await runMode(c,'f16');const difference=compare(full,mixed);
      check(`${c.name}: image PSNR >= 40 dB`,difference.exact||difference.psnr!>=40,difference);
      check(`${c.name}: centre and sort depth preserved`,difference.maxCenterError===0&&difference.depthIdentity);
      if(c.name==='guards'){
        const flags=Array.from({length:6},(_,i)=>mixed.gpu.splats[i*16+11]);
        check('Large SH coefficients use FP32 fallback',(flags[0]&1)!==0&&(flags[1]&1)===0&&(flags[2]&1)===0,{flags});
        check('Anisotropic rotation uses FP32 fallback',(flags[3]&2)!==0&&(flags[4]&2)===0,{flags});
      }
      report.cases.push({case:c,f32:full.metadata,f16:mixed.metadata,difference});
      if(errors.length)throw new Error(errors.join('\n'));
    }
    // An actual second GPUDevice without shader-f16 tests renderer fallback.
    const fallbackAdapter=(await navigator.gpu.requestAdapter())!;
    const fallbackDevice=await fallbackAdapter.requestDevice({requiredLimits:{maxComputeWorkgroupStorageSize:fallbackAdapter.limits.maxComputeWorkgroupStorageSize}});
    const fallbackErrors:string[]=[];fallbackDevice.addEventListener('uncapturederror',e=>fallbackErrors.push((e as GPUUncapturedErrorEvent).error.message));
    const pc=await load(await fetched(`${ROOT}stage3/fixtures/shapes.ply`),fallbackDevice);
    const fallbackCamera=new Camera(canvas,fallbackDevice);fallbackCamera.set_preset(presets.get('identity')!);
    const fallbackTexture=fallbackDevice.createTexture({size:[canvas.width,canvas.height],format:'bgra8unorm',usage:GPUTextureUsage.RENDER_ATTACHMENT});
    const renderer=await getRenderer(pc,fallbackDevice,'bgra8unorm',fallbackCamera.uniform_buffer,{collectStatistics:false});
    renderer.setSettings({precision:'f16',scaling:1,culling:true,sorting:true,mode:'gaussian',shDegree:3});
    const enc=fallbackDevice.createCommandEncoder();renderer.frame(enc,fallbackTexture.createView());fallbackDevice.queue.submit([enc.finish()]);
    const snapshot=await renderer.inspect();
    check('Device without shader-f16 falls back to FP32',renderer.precision().active==='f32'&&!renderer.precision().supported&&snapshot.draw[1]===6&&!fallbackErrors.length,{precision:renderer.precision(),errors:fallbackErrors});
    renderer.destroy();renderer.destroy();pc.destroy();fallbackCamera.uniform_buffer.destroy();fallbackTexture.destroy();fallbackDevice.destroy();
    check('No GPU or JavaScript errors',errors.length===0,{errors});
    report.finished=new Date().toISOString();report.passed=report.checks.every((c:any)=>c.passed);
    files.set('acceptance-raw.json',new Blob([JSON.stringify(report,null,2)],{type:'application/json'}));
    phase=report.passed?'complete':'failed';update(`${phase}: ${report.cases.length} 个场景，${report.checks.length} 项检查。`);
    return {passed:report.passed,cases:report.cases.length,checks:report.checks.length,failures:report.checks.filter((c:any)=>!c.passed)};
  } catch(e){phase='failed';update(String(e));throw e;} finally {busy=false;}
}
const ready=initialize();
(window as any).__precisionAcceptance={ready:()=>ready,runAll,state:()=>({phase,progress,busy,cases:report?.cases.length??0,errors}),
  report:()=>report,files:()=>[...files.keys()],download:(path:string)=>{
    const blob=files.get(path);if(!blob)throw new Error(path);const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=path.split('/').pop()!;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  },dispose:destroy};
document.querySelector('#start')!.addEventListener('click',()=>void runAll().catch(console.error));
