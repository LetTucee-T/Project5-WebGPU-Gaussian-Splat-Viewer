import { Mat3, mat3, Mat4, mat4, Vec3, vec3, Vec2, vec2 } from 'wgpu-matrix';
import { log, time, timeLog } from '../utils/simple-console';
import { Bounds, FileData, readFileData } from '../utils/load';

interface CameraJson {
  id: number
  img_name: string
  width: number
  height: number
  position: number[]
  rotation: number[][]
  fx: number
  fy: number
};

function focal2fov(focal: number, pixels: number): number {
  return 2 * Math.atan(pixels / (2 * focal));
}

function fov2focal(fov: number, pixels: number): number {
  return pixels / (2 * Math.tan(fov * 0.5));
}

function get_view_matrix(r: Mat4, t: Vec3): Mat4 {
  const minus_t = vec3.mulScalar(t, -1);
  return mat4.translate(r, minus_t);
}


function get_projection_matrix(znear: number, zfar: number, fov_x: number, fov_y: number) {
  // return mat4.perspective(fov_y, 1, znear, zfar);

  const tan_half_fov_y = Math.tan(fov_y / 2.);
  const tan_half_fov_x = Math.tan(fov_x / 2.);

  const top = tan_half_fov_y * znear;
  const bottom = -top;
  const right = tan_half_fov_x * znear;
  const left = -right;

  const p = mat4.create();
  p[0] = 2.0 * znear / (right - left);
  // p[5] = 2.0 * znear / (top - bottom);
  p[5] = -2.0 * znear / (top - bottom);   // flip Y
  p[2] = (right + left) / (right - left);
  p[6] = (top + bottom) / (top - bottom);
  p[14] = 1.;
  p[10] = zfar / (zfar - znear);
  p[11] = -(zfar * znear) / (zfar - znear);
  mat4.transpose(p, p);
  
  // p[0] = 2.0 * znear / (right - left);
  // p[5] = 2.0 * znear / (top - bottom);
  // p[8] = (right + left) / (right - left);
  // p[9] = (top + bottom) / (top - bottom);
  // p[10] = zfar / (zfar - znear);
  // p[11] = -(zfar * znear) / (zfar - znear);
  // p[14] = 1.;
  // mat4.transpose(p, p);
  return p;
}

export interface CameraPreset {
  position: Vec3,
  rotation: Mat4,
  name: string,
  intrinsics?: { width: number; height: number; fx: number; fy: number },
}

export async function load_camera_presets(file: FileData): Promise<CameraPreset[]> {
  const arrayBuffer = await readFileData(file);
  const text = new TextDecoder().decode(arrayBuffer as ArrayBuffer);
  const json = JSON.parse(text);
  if (!Array.isArray(json) || json.length === 0) throw new Error('Camera JSON must be a nonempty array.');
  log(`loaded cameras count: ${json.length}`);

  return json.map((j: CameraJson, index: number): CameraPreset => {
    const finiteArray = (a: unknown, n: number): a is number[] => Array.isArray(a) && a.length === n && a.every(v => typeof v === 'number' && Number.isFinite(v));
    if (!j || !finiteArray(j.position, 3) || !Array.isArray(j.rotation) || j.rotation.length !== 3 || !j.rotation.every(row => finiteArray(row, 3))) {
      throw new Error(`Camera ${index}: expected a finite position and a 3 x 3 rotation.`);
    }
    const r = j.rotation;
    const determinant = r[0][0] * (r[1][1]*r[2][2]-r[1][2]*r[2][1]) - r[0][1] * (r[1][0]*r[2][2]-r[1][2]*r[2][0]) + r[0][2] * (r[1][0]*r[2][1]-r[1][1]*r[2][0]);
    // The course's hand-authored Bonsai camera 2 is not exactly orthonormal.
    // Preserve supplied matrices while rejecting singular, non-invertible data.
    if (!Number.isFinite(determinant) || Math.abs(determinant) < 1e-8) throw new Error(`Camera ${index}: rotation must be invertible.`);
    const keys = ['width', 'height', 'fx', 'fy'];
    let intrinsics: CameraPreset['intrinsics'];
    if (keys.some(key => j[key] !== undefined)) {
      if (!keys.every(key => typeof j[key] === 'number' && Number.isFinite(j[key]) && j[key] > 0)) throw new Error(`Camera ${index}: width, height, fx and fy must all be positive.`);
      intrinsics = { width: j.width, height: j.height, fx: j.fx, fy: j.fy };
    }
    const position = vec3.clone(j.position);
    // const rotation = mat3.create(...j.rotation.flat());
    // Camera-to-world rows become world-to-camera columns: this transposes R.
    const rotation = mat4.fromMat3(mat3.create(...j.rotation.flat()));

    return {
      position,
      rotation,
      name: String(j.img_name ?? j.id ?? index),
      intrinsics,
    };
  });
}


const c_size_vec2 = 4 * 2;
const c_size_mat4 = 4 * 16; // byte size of mat4 (i.e. Float32Array(16))
const c_size_camera_uniform = 4 * c_size_mat4 + 2 * c_size_vec2;
interface CameraUniform {
  view_matrix: Mat4,
  view_inv_matrix: Mat4,
  proj_matrix: Mat4,
  proj_inv_matrix: Mat4,

  viewport: Vec2,
  focal: Vec2,
}

export function create_camera_uniform_buffer(device: GPUDevice) {
  return device.createBuffer({
    label: 'camera uniform',
    size: c_size_camera_uniform,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
}

const intermediate_float_32_array = new Float32Array(c_size_camera_uniform / Float32Array.BYTES_PER_ELEMENT);

export class Camera {
  constructor(
    public readonly canvas: HTMLCanvasElement,
    private readonly device: GPUDevice,
  ) {
    this.uniform_buffer = create_camera_uniform_buffer(device);
    this.on_update_canvas();
  }

  on_update_canvas(): void {
    const width = Math.max(1, this.canvas.width), height = Math.max(1, this.canvas.height);
    if (this.intrinsics) {
      // Fit vertically and retain pixel aspect when the viewport changes shape.
      const scale = height / this.intrinsics.height;
      this.focal.set([this.intrinsics.fx * scale, this.intrinsics.fy * scale]);
      this.fovY = focal2fov(this.focal[1], height);
    } else {
      this.fovY = Math.PI / 4;
      const focal = 0.5 * height / Math.tan(this.fovY * 0.5);
      this.focal.set([focal, focal]);
    }
    this.fovX = focal2fov(this.focal[0], width);
    this.viewport.set([width, height]);
    // const viewport_ratio = this.canvas.width / this.canvas.height;

    this.update_buffer();
  }
  
  readonly uniform_buffer: GPUBuffer;
  
  position = vec3.create();
  rotation = mat4.identity();
  private intrinsics?: CameraPreset['intrinsics'];
  private far = 100;
  private fovY: number = 45 / 180 * Math.PI;
  private fovX: number;
  private focal: Vec2 = vec2.create();
  private viewport: Vec2 = vec2.create();

  private view_matrix: Mat4 = mat4.identity();
  private proj_matrix: Mat4 = mat4.identity();

  look = vec3.create(0, 0, 1);
  up = vec3.create(0, 1, 0);
  right = vec3.create(1, 0, 0);

  update_buffer(): void {
    let offset = 0;

    this.view_matrix = get_view_matrix(this.rotation, this.position);
    this.proj_matrix = get_projection_matrix(0.01, this.far, this.fovX, this.fovY);

    const inv_view_matrix = mat4.inverse(this.view_matrix);
    vec3.transformMat4Upper3x3(vec3.create(0, 0, 1), inv_view_matrix, this.look);
    vec3.normalize(this.look, this.look);

    vec3.transformMat4Upper3x3(vec3.create(0, 1, 0), inv_view_matrix, this.up);
    vec3.normalize(this.up, this.up);
    vec3.transformMat4Upper3x3(vec3.create(1, 0, 0), inv_view_matrix, this.right);
    vec3.normalize(this.right, this.right);

    intermediate_float_32_array.set(this.view_matrix, offset);
    offset += 16;
    intermediate_float_32_array.set(inv_view_matrix, offset);
    offset += 16;
    intermediate_float_32_array.set(this.proj_matrix, offset);
    offset += 16;
    intermediate_float_32_array.set(mat4.inverse(this.proj_matrix), offset);
    offset += 16;
    intermediate_float_32_array.set(this.viewport, offset);
    offset += 2;
    intermediate_float_32_array.set(this.focal, offset);
    offset += 2;

    this.device.queue.writeBuffer(this.uniform_buffer, 0, intermediate_float_32_array);
  }
  set_preset(preset: CameraPreset): void {
    vec3.copy(preset.position, this.position);
    mat4.copy(preset.rotation, this.rotation);
    this.intrinsics = preset.intrinsics;
    this.far = 100;
    this.on_update_canvas();
  }

  frame_bounds(bounds: Bounds): void {
    const center = bounds.min.map((v, i) => (v + bounds.max[i]) / 2);
    const radius = Math.max(.1, Math.hypot(...bounds.max.map((v, i) => (v - bounds.min[i]) / 2)));
    this.intrinsics = undefined;
    this.on_update_canvas();
    const tangent = Math.min(this.viewport[0] / (2*this.focal[0]), this.viewport[1] / (2*this.focal[1]));
    const distance = radius / Math.sin(Math.atan(tangent)) * 1.1;
    this.position.set([center[0], center[1], center[2] - distance]);
    mat4.identity(this.rotation);
    this.far = Math.max(100, distance + 2*radius);
    this.update_buffer();
  }

  snapshot() {
    return { position: Array.from(this.position), view: Array.from(this.view_matrix), projection: Array.from(this.proj_matrix),
      viewport: Array.from(this.viewport), focal: Array.from(this.focal), near: .01, far: this.far };
  }
};
