import common from '../shaders/preprocess.wgsl';
import full from '../shaders/preprocess-f32.wgsl';
import mixed from '../shaders/preprocess-f16.wgsl';

export type PreprocessPrecision = 'f32' | 'f16';
export function preprocessShader(precision: PreprocessPrecision): string {
  return (precision === 'f16' ? 'enable f16;\n' : '') + common + '\n' + (precision === 'f16' ? mixed : full);
}
