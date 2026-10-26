import { Float16Array } from '@petamoriken/float16';
import { log, time, timeLog } from './simple-console';
import { decodeHeader, readRawVertex ,nShCoeffs} from './plyreader';

const c_size_float = 2;   // byte size of f16
// A conservative bound for the 16 SH terms at unit directions. With coefficients
// <=256, even a loose 64*256 bound stays far below the f16 finite maximum 65504.
export const SH_F16_COEFFICIENT_LIMIT = 256;

const c_size_3d_gaussian =
  3 * c_size_float  // x y z (position)
  + c_size_float    // opacity
  + 4 * c_size_float  // rotation
  + 4 * c_size_float  //scale
;

export type PointCloud = Awaited<ReturnType<typeof load>>;
export type FileData = Blob | ArrayBuffer | string;
export type Bounds = { min: number[]; max: number[] };

export async function readFileData(file: FileData): Promise<ArrayBuffer> {
  if (file instanceof Blob) return file.arrayBuffer();
  if (typeof file === 'string') return new TextEncoder().encode(file).buffer;
  return file;
}

export async function load(file: FileData, device: GPUDevice) {
  const arrayBuffer = await readFileData(file);

  const [vertexCount, propertyTypes, vertexData] = decodeHeader(arrayBuffer as ArrayBuffer);
  // figure out the SH degree from the number of coefficients
  var nRestCoeffs = 0;
  for (const propertyName in propertyTypes) {
      if (propertyName.startsWith('f_rest_')) {
          nRestCoeffs += 1;
      }
  }
  const nCoeffsPerColor = nRestCoeffs / 3;
  const sh_deg = Math.sqrt(nCoeffsPerColor + 1) - 1;
  const num_coefs = nShCoeffs(sh_deg);
  const max_num_coefs = 16;
  const requiredFields = ['x', 'y', 'z', 'opacity', 'rot_0', 'rot_1', 'rot_2', 'rot_3', 'scale_0', 'scale_1', 'scale_2', 'f_dc_0', 'f_dc_1', 'f_dc_2'];
  for (let i = 0; i < nRestCoeffs; ++i) requiredFields.push(`f_rest_${i}`);
  for (const name of requiredFields) {
    if (!Object.prototype.hasOwnProperty.call(propertyTypes, name)) throw new Error(`Missing Gaussian PLY property: ${name}`);
  }

  const c_size_sh_coef = 
    3 * max_num_coefs * c_size_float // 3 channels (RGB) x 16 coefs
  ;

  // figure out the order in which spherical harmonics should be read
  const shFeatureOrder = [];
  for (let rgb = 0; rgb < 3; ++rgb) {
      shFeatureOrder.push(`f_dc_${rgb}`);
  }
  for (let i = 0; i < nCoeffsPerColor; ++i) {
      for (let rgb = 0; rgb < 3; ++rgb) {
          shFeatureOrder.push(`f_rest_${rgb * nCoeffsPerColor + i}`);
      }
  }

  const num_points = vertexCount;
  const limit = Math.min(device.limits.maxBufferSize, device.limits.maxStorageBufferBindingSize);
  if (Math.max(num_points * c_size_3d_gaussian, num_points * c_size_sh_coef) > limit) {
    throw new Error(`Scene exceeds this GPU's per-buffer limit (${Math.round(limit / 1048576)} MiB).`);
  }
  const bounds: Bounds = { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] };

  log(`num points: ${num_points}`);
  log(`processing loaded attributes...`);
  time();

  // xyz (position), opacity, cov (from rot and scale)
  const gaussian_3d_buffer = device.createBuffer({
    label: 'ply input 3d gaussians data buffer',
    size: num_points * c_size_3d_gaussian,  // buffer size multiple of 4?
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  let sh_buffer: GPUBuffer | undefined;
  try {
  const gaussian = new Float16Array(gaussian_3d_buffer.getMappedRange());

  // Spherical harmonic function coeffs
  sh_buffer = device.createBuffer({
    label: 'PLY spherical harmonics',
    size: num_points * c_size_sh_coef,  // buffer size multiple of 4?
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  const sh = new Float16Array(sh_buffer.getMappedRange());

  var readOffset = 0;
  let shHalfFallbackPoints = 0;
  for (let i = 0; i < num_points; i++) {
    const [newReadOffset, rawVertex] = readRawVertex(readOffset, vertexData, propertyTypes);
    readOffset = newReadOffset;
    for (const name of requiredFields) {
      // Opacity is a logit: +/-Infinity represent the valid sigmoid endpoints 1/0.
      // The course Bonsai scene contains 32 positive-infinite opacity values.
      if (name === 'opacity' && (rawVertex[name] === Infinity || rawVertex[name] === -Infinity)) continue;
      if (!Number.isFinite(rawVertex[name]) || Math.abs(rawVertex[name]) > 65504) {
        throw new Error(`Vertex ${i}: ${name} cannot be stored as a finite f16 value.`);
      }
    }

    const o = i * (c_size_3d_gaussian / c_size_float);
    const output_offset = i * max_num_coefs * 3;
    
    let halfSafe = true;
    for (let order = 0; order < num_coefs; ++order) {
        const order_offset = order * 3;
        for (let j = 0; j < 3; ++j) {
            const coeffName = shFeatureOrder[order * 3 + j];
            sh[output_offset +order_offset+j]=rawVertex[coeffName];
          if (Math.abs(sh[output_offset +order_offset+j]) > SH_F16_COEFFICIENT_LIMIT) halfSafe = false;
        }
    }

    gaussian[o + 0] = rawVertex.x;
    gaussian[o + 1] = rawVertex.y;
    gaussian[o + 2] = rawVertex.z;
    gaussian[o + 3] = rawVertex.opacity;
    gaussian[o + 4] = rawVertex.rot_0;
    gaussian[o + 5] = rawVertex.rot_1;
    gaussian[o + 6] = rawVertex.rot_2;
    gaussian[o + 7] = rawVertex.rot_3;
    gaussian[o + 8] = rawVertex.scale_0;
    gaussian[o + 9] = rawVertex.scale_1;
    gaussian[o + 10] = rawVertex.scale_2;
    // Reuse the existing padding half; input stride remains 24 bytes.
    gaussian[o + 11] = halfSafe ? 1 : 0;
    if (!halfSafe) shHalfFallbackPoints++;
    for (let axis = 0; axis < 3; ++axis) {
      bounds.min[axis] = Math.min(bounds.min[axis], gaussian[o + axis]);
      bounds.max[axis] = Math.max(bounds.max[axis], gaussian[o + axis]);
    }
  }

  gaussian_3d_buffer.unmap(); 
  sh_buffer.unmap();

  timeLog();
  let destroyed = false;
  return {
    num_points: num_points,
    sh_deg: sh_deg,
    shHalfFallbackPoints,
    gaussian_3d_buffer,
    sh_buffer,
    bounds,
    sourceBytes: arrayBuffer.byteLength,
    destroy() {
      if (destroyed) return;
      destroyed = true;
      gaussian_3d_buffer.destroy();
      sh_buffer.destroy();
    },
  };
  } catch (error) {
    gaussian_3d_buffer.destroy();
    sh_buffer?.destroy();
    throw error;
  }
}
