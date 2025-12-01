// MLS-MPM buffer schema and helpers.
// Layout is designed for peercompute compatibility: explicit offsets/strides, 16-byte alignment.

export const MPM_PARTICLE_STRIDE = 144; // bytes

// Offsets are in bytes relative to the start of the particle.
export const MPM_PARTICLE_FIELDS = {
  position: 0, // vec3<f32> (12B) + padding
  materialId: 12, // f32 (store int as float to keep alignment simple)
  velocity: 16, // vec3<f32> (12B) + padding
  phase: 28, // f32 (0=solid,1=liquid,2=gas)
  mass: 32, // f32
  volume: 36, // f32
  temperature: 40, // f32
  padding0: 44, // f32 reserved/padding
  F: 48, // mat3x3<f32> (48B: 3 columns * 16B alignment)
  C: 96 // mat3x3<f32> (48B: 3 columns * 16B alignment)
};

export const MPM_GRID_STRIDE = 16; // bytes
export const MPM_GRID_FIELDS = {
  velocity: 0, // vec3<f32> encoded as i32 when using fixed-point atomics
  mass: 12 // f32 encoded as i32 when using fixed-point atomics
};

// Baseline constants (from WebGPU-Ocean MLS-MPM)
export const MPM_WORKGROUP_SIZE = 64;
export const MPM_FIXED_POINT_SCALE = 1e7; // prototype scale for fixed-point atomics

export const DEFAULT_WATER_CONSTANTS = {
  stiffness: 3.0,
  restDensity: 4.0,
  dynamicViscosity: 0.1,
  dt: 0.2,
  fixedPointScale: MPM_FIXED_POINT_SCALE
};

export function particleBufferSize(particleCount) {
  return particleCount * MPM_PARTICLE_STRIDE;
}

export function gridBufferSize(cellCount) {
  return cellCount * MPM_GRID_STRIDE;
}

// Create typed views for a given particle index on a CPU-side ArrayBuffer.
// This is useful for initialization or headless tests; GPU buffers stay separate.
export function particleViews(buffer, index = 0) {
  const base = index * MPM_PARTICLE_STRIDE;
  return {
    position: new Float32Array(buffer, base + MPM_PARTICLE_FIELDS.position, 3),
    materialId: new Float32Array(buffer, base + MPM_PARTICLE_FIELDS.materialId, 1),
    velocity: new Float32Array(buffer, base + MPM_PARTICLE_FIELDS.velocity, 3),
    phase: new Float32Array(buffer, base + MPM_PARTICLE_FIELDS.phase, 1),
    mass: new Float32Array(buffer, base + MPM_PARTICLE_FIELDS.mass, 1),
    volume: new Float32Array(buffer, base + MPM_PARTICLE_FIELDS.volume, 1),
    temperature: new Float32Array(buffer, base + MPM_PARTICLE_FIELDS.temperature, 1),
    F: new Float32Array(buffer, base + MPM_PARTICLE_FIELDS.F, 9),
    C: new Float32Array(buffer, base + MPM_PARTICLE_FIELDS.C, 9)
  };
}

export function createParticleBuffer(device, particleCount, usage) {
  const size = particleBufferSize(particleCount);
  const bufferUsage = usage ?? (GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
  return device.createBuffer({
    label: "mpm-particles",
    size,
    usage: bufferUsage
  });
}

export function createGridBuffer(device, cellCount, usage) {
  const size = gridBufferSize(cellCount);
  const bufferUsage = usage ?? (GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
  return device.createBuffer({
    label: "mpm-grid",
    size,
    usage: bufferUsage
  });
}
