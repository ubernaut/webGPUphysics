import {
  MPM_PARTICLE_STRIDE,
  particleBufferSize,
  particleViews
} from "./schema.js";

const identity3 = () => [1, 0, 0, 0, 1, 0, 0, 0, 1];

/**
 * Create an ArrayBuffer for particles laid out in a block (dam-break style).
 * Options:
 * - count: number of particles (required)
 * - gridSize: {x,y,z} for clamping positions
 * - start: [x,y,z] starting corner (default [0,0,0])
 * - spacing: particle spacing (default 0.65 to mirror WebGPU-Ocean)
 * - jitter: random jitter magnitude (default 0.0)
 * - materialId: default 0
 * - mass: default 1.0
 * - temperature: default 273 K
 * - phase: default 1 (liquid)
 */
export function createBlockParticleData(options) {
  const {
    count,
    gridSize,
    start = [0, 0, 0],
    spacing = 0.65,
    jitter = 0.0,
    materialId = 0,
    mass = 1.0,
    temperature = 273.0,
    phase = 1
  } = options;
  if (!count || !gridSize) throw new Error("count and gridSize are required");

  const buf = new ArrayBuffer(particleBufferSize(count));
  let written = 0;

  for (let y = start[1]; y < gridSize.y && written < count; y += spacing) {
    for (let x = start[0]; x < gridSize.x && written < count; x += spacing) {
      for (let z = start[2]; z < gridSize.z && written < count; z += spacing) {
        const views = particleViews(buf, written);
        const jx = jitter ? (Math.random() * 2 - 1) * jitter : 0;
        const jy = jitter ? (Math.random() * 2 - 1) * jitter : 0;
        const jz = jitter ? (Math.random() * 2 - 1) * jitter : 0;
        views.position.set([x + jx, y + jy, z + jz]);
        views.materialId[0] = materialId;
        views.velocity.set([0, 0, 0]);
        views.phase[0] = phase;
        views.mass[0] = mass;
        views.volume[0] = 1.0; // will be recomputed in P2G2
        views.temperature[0] = temperature;
        views.F.set(identity3());
        views.C.fill(0);
        written += 1;
      }
    }
  }

  if (written < count) {
    throw new Error(`Could not place all particles; placed ${written} of ${count}`);
  }

  return buf;
}
