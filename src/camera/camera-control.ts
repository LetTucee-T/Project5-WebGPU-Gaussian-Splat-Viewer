import { vec3, mat4, quat } from 'wgpu-matrix';
import { Camera } from './camera';

export class CameraControl {
  private element: HTMLCanvasElement;
  private pointer: number | undefined;
  private panning = false;
  private lastX = 0;
  private lastY = 0;

  constructor(private camera: Camera) {
    this.element = camera.canvas;
    this.element.addEventListener('pointerdown', this.down);
    this.element.addEventListener('pointermove', this.move);
    this.element.addEventListener('pointerup', this.up);
    this.element.addEventListener('pointercancel', this.up);
    this.element.addEventListener('lostpointercapture', this.up);
    this.element.addEventListener('wheel', this.wheel, { passive: false });
    this.element.addEventListener('contextmenu', this.preventContext);
  }
  private down = (event: PointerEvent) => {
    if (!event.isPrimary || this.pointer !== undefined) return;
    event.preventDefault();
    this.pointer = event.pointerId;
    this.panning = event.button !== 0;
    this.lastX = event.clientX;
    this.lastY = event.clientY;
    this.element.setPointerCapture(event.pointerId);
  };
  private move = (event: PointerEvent) => {
    if (event.pointerId !== this.pointer) return;
    const dx = event.clientX - this.lastX, dy = event.clientY - this.lastY;
    this.lastX = event.clientX;
    this.lastY = event.clientY;
    if (this.panning) this.pan(dx, dy); else this.rotate(dx, dy);
  };
  private up = (event: PointerEvent) => {
    if (event.pointerId !== this.pointer) return;
    this.pointer = undefined;
    if (this.element.hasPointerCapture(event.pointerId)) this.element.releasePointerCapture(event.pointerId);
  };
  private preventContext = (event: Event) => event.preventDefault();
  private wheel = (event: WheelEvent) => {
    event.preventDefault();
    const units = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? this.element.clientHeight : 1;
    const delta = vec3.mulScalar(this.camera.look, -event.deltaY * units * .001);
    vec3.add(delta, this.camera.position, this.camera.position);
    this.camera.update_buffer();
  };
  rotate(dx: number, dy: number) {
    const rotation = mat4.fromQuat(quat.fromEuler(dy * .01, -dx * .01, 0, 'xyz'));
    mat4.mul(rotation, this.camera.rotation, this.camera.rotation);
    this.camera.update_buffer();
  }
  pan(dx: number, dy: number) {
    vec3.add(this.camera.position, vec3.mulScalar(this.camera.up, -dy * .01), this.camera.position);
    vec3.add(this.camera.position, vec3.mulScalar(this.camera.right, -dx * .01), this.camera.position);
    this.camera.update_buffer();
  }
  dispose() {
    if (this.pointer !== undefined && this.element.hasPointerCapture(this.pointer)) this.element.releasePointerCapture(this.pointer);
    this.pointer = undefined;
    this.element.removeEventListener('pointerdown', this.down);
    this.element.removeEventListener('pointermove', this.move);
    this.element.removeEventListener('pointerup', this.up);
    this.element.removeEventListener('pointercancel', this.up);
    this.element.removeEventListener('lostpointercapture', this.up);
    this.element.removeEventListener('wheel', this.wheel);
    this.element.removeEventListener('contextmenu', this.preventContext);
  }
}
