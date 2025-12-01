import {
  MPM_PARTICLE_STRIDE,
  MATERIAL_TYPE,
  MATERIAL_PRESETS,
  particleBufferSize,
  particleViews
} from "./schema.js";

// Identity matrix for mat3x3f (with column padding for GPU alignment)
// In WGSL std430, mat3x3f has 3 columns of vec4f (padded), so 12 floats total
const identity3Padded = () => [
  1, 0, 0, 0,  // column 0 + padding
  0, 1, 0, 0,  // column 1 + padding
  0, 0, 1, 0   // column 2 + padding
];

const zero3Padded = () => [
  0, 0, 0, 0,
  0, 0, 0, 0,
  0, 0, 0, 0
];

/**
 * Create an ArrayBuffer for particles laid out in a block (dam-break style).
 * Options:
 * - count: number of particles (required)
 * - gridSize: {x,y,z} for clamping positions
 * - start: [x,y,z] starting corner (default [0,0,0])
 * - spacing: particle spacing (default 0.65 to mirror WebGPU-Ocean)
 * - jitter: random jitter magnitude (default 0.0)
 * - materialType: MATERIAL_TYPE enum value (default LIQUID)
 * - mass: default 1.0
 * - temperature: default 273 K
 * - phase: 0=solid, 1=liquid, 2=gas (default based on materialType)
 * - mu: shear modulus (default from material preset)
 * - lambda: bulk modulus (default from material preset)
 * - restDensity: for volume calculation (default 1.0)
 */
export function createBlockParticleData(options) {
  const {
    count,
    gridSize,
    start = [0, 0, 0],
    spacing = 0.65,
    jitter = 0.0,
    materialType = MATERIAL_TYPE.LIQUID,
    mass = 1.0,
    temperature = 300.0,
    phase = null,  // Auto-determine from materialType if not specified
    mu = null,     // Auto-determine from materialType if not specified
    lambda = null, // Auto-determine from materialType if not specified
    restDensity = 1.0
  } = options;
  
  if (!count || !gridSize) throw new Error("count and gridSize are required");

  // Determine defaults based on material type
  let defaultPhase, defaultMu, defaultLambda;
  
  switch (materialType) {
    case MATERIAL_TYPE.BRITTLE_SOLID:
      defaultPhase = 0;  // Solid
      defaultMu = MATERIAL_PRESETS.ice.mu;
      defaultLambda = MATERIAL_PRESETS.ice.lambda;
      break;
    case MATERIAL_TYPE.ELASTIC_SOLID:
      defaultPhase = 0;  // Solid
      defaultMu = MATERIAL_PRESETS.rubber.mu;
      defaultLambda = MATERIAL_PRESETS.rubber.lambda;
      break;
    case MATERIAL_TYPE.LIQUID:
      defaultPhase = 1;  // Liquid
      defaultMu = 0.0;
      defaultLambda = MATERIAL_PRESETS.water.stiffness;
      break;
    case MATERIAL_TYPE.GAS:
      defaultPhase = 2;  // Gas
      defaultMu = 0.0;
      defaultLambda = MATERIAL_PRESETS.steam.gasConstant;
      break;
    case MATERIAL_TYPE.GRANULAR:
      defaultPhase = 0;  // Solid-ish
      defaultMu = 100.0;
      defaultLambda = 100.0;
      break;
    default:
      defaultPhase = 1;
      defaultMu = 0.0;
      defaultLambda = 50.0;
  }

  const actualPhase = phase !== null ? phase : defaultPhase;
  const actualMu = mu !== null ? mu : defaultMu;
  const actualLambda = lambda !== null ? lambda : defaultLambda;

  const buf = new ArrayBuffer(particleBufferSize(count));
  let written = 0;

  for (let y = start[1]; y < gridSize.y && written < count; y += spacing) {
    for (let x = start[0]; x < gridSize.x && written < count; x += spacing) {
      for (let z = start[2]; z < gridSize.z && written < count; z += spacing) {
        const views = particleViews(buf, written);
        
        // Position with optional jitter
        const jx = jitter ? (Math.random() * 2 - 1) * jitter : 0;
        const jy = jitter ? (Math.random() * 2 - 1) * jitter : 0;
        const jz = jitter ? (Math.random() * 2 - 1) * jitter : 0;
        views.position.set([x + jx, y + jy, z + jz]);
        
        // Material type (u32)
        views.materialType[0] = materialType;
        
        // Velocity
        views.velocity.set([0, 0, 0]);
        
        // Phase (u32)
        views.phase[0] = actualPhase;
        
        // Mass
        views.mass[0] = mass;
        
        // Initial/reference volume V0 = mass / rest_density
        views.volume0[0] = mass / restDensity;
        
        // Temperature
        views.temperature[0] = temperature;
        
        // Damage (0 = undamaged)
        views.damage[0] = 0.0;
        
        // Deformation gradient F = Identity
        views.F.set(identity3Padded());
        
        // APIC affine matrix C = 0
        views.C.set(zero3Padded());
        
        // Material properties
        views.mu[0] = actualMu;
        views.lambda[0] = actualLambda;
        
        written += 1;
      }
    }
  }

  if (written < count) {
    throw new Error(`Could not place all particles; placed ${written} of ${count}`);
  }

  return buf;
}

/**
 * Create ice block particle data
 */
export function createIceBlockData(options) {
  return createBlockParticleData({
    ...options,
    materialType: MATERIAL_TYPE.BRITTLE_SOLID,
    temperature: options.temperature ?? 260.0,  // Below freezing
    mu: options.mu ?? MATERIAL_PRESETS.ice.mu,
    lambda: options.lambda ?? MATERIAL_PRESETS.ice.lambda
  });
}

/**
 * Create water block particle data
 */
export function createWaterBlockData(options) {
  return createBlockParticleData({
    ...options,
    materialType: MATERIAL_TYPE.LIQUID,
    temperature: options.temperature ?? 300.0,  // Room temperature
  });
}

/**
 * Create steam/gas particle data
 */
export function createSteamBlockData(options) {
  return createBlockParticleData({
    ...options,
    materialType: MATERIAL_TYPE.GAS,
    temperature: options.temperature ?? 400.0,  // Above boiling
    restDensity: options.restDensity ?? 0.1     // Low density for gas
  });
}

/**
 * Create rubber/elastic block particle data
 */
export function createRubberBlockData(options) {
  return createBlockParticleData({
    ...options,
    materialType: MATERIAL_TYPE.ELASTIC_SOLID,
    temperature: options.temperature ?? 300.0,
    mu: options.mu ?? MATERIAL_PRESETS.rubber.mu,
    lambda: options.lambda ?? MATERIAL_PRESETS.rubber.lambda
  });
}

/**
 * Create mixed material scene with multiple blocks
 */
export function createMixedMaterialData(options) {
  const { gridSize, blocks } = options;
  
  // Calculate total particle count
  let totalCount = 0;
  for (const block of blocks) {
    totalCount += block.count;
  }
  
  const buf = new ArrayBuffer(particleBufferSize(totalCount));
  let offset = 0;
  
  for (const block of blocks) {
    const blockBuf = createBlockParticleData({
      count: block.count,
      gridSize,
      start: block.start,
      spacing: block.spacing ?? 0.65,
      jitter: block.jitter ?? 0.0,
      materialType: block.materialType ?? MATERIAL_TYPE.LIQUID,
      mass: block.mass ?? 1.0,
      temperature: block.temperature ?? 300.0,
      mu: block.mu,
      lambda: block.lambda,
      restDensity: block.restDensity ?? 1.0
    });
    
    // Copy block data into main buffer
    const src = new Uint8Array(blockBuf);
    const dst = new Uint8Array(buf, offset * MPM_PARTICLE_STRIDE);
    dst.set(src.subarray(0, block.count * MPM_PARTICLE_STRIDE));
    
    offset += block.count;
  }
  
  return buf;
}
