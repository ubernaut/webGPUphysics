// WGSL shader sources for MLS-MPM (multi-material constitutive framework + thermal diffusion).
// Supports: BRITTLE_SOLID (linear elastic + fracture), ELASTIC_SOLID (Neo-Hookean),
//           LIQUID (Tait EOS), GAS (Ideal Gas), GRANULAR (Drucker-Prager - future)
// Thermal: Heat transfer via grid, latent heat for phase transitions

// Material type constants (must match schema.js MATERIAL_TYPE enum)
const MATERIAL_CONSTANTS = /* wgsl */ `
const MATERIAL_BRITTLE_SOLID: u32 = 0u;
const MATERIAL_ELASTIC_SOLID: u32 = 1u;
const MATERIAL_LIQUID: u32 = 2u;
const MATERIAL_GAS: u32 = 3u;
const MATERIAL_GRANULAR: u32 = 4u;

// Phase transition temperatures (Kelvin)
const T_MELT: f32 = 273.0;
const T_BOIL: f32 = 373.0;
const T_MELT_LOW: f32 = 271.0;  // Hysteresis
const T_BOIL_HIGH: f32 = 375.0;

// Latent heats (scaled for simulation)
const LATENT_HEAT_MELT: f32 = 33.4;  // Scaled from 334 kJ/kg
const LATENT_HEAT_BOIL: f32 = 226.0; // Scaled from 2260 kJ/kg
const SPECIFIC_HEAT: f32 = 4.186;    // Scaled J/(kg·K)
`;

// Common particle struct (160 bytes, matches schema.js)
const PARTICLE_STRUCT = /* wgsl */ `
struct Particle {
  position: vec3f,
  materialType: u32,      // BRITTLE_SOLID, ELASTIC_SOLID, LIQUID, GAS, GRANULAR
  velocity: vec3f,
  phase: u32,             // Current phase: 0=solid, 1=liquid, 2=gas
  mass: f32,
  volume0: f32,           // Initial/reference volume
  temperature: f32,
  damage: f32,            // Fracture damage [0,1] for brittle materials
  F: mat3x3f,             // Deformation gradient
  C: mat3x3f,             // APIC affine matrix
  mu: f32,                // Per-particle shear modulus
  lambda: f32,            // Per-particle bulk modulus
  _pad0: f32,
  _pad1: f32,
};
`;

// Extended Cell struct (32 bytes, includes thermal fields)
const CELL_STRUCT = /* wgsl */ `
struct Cell {
  vx: i32,           // Velocity x (fixed-point)
  vy: i32,           // Velocity y (fixed-point)
  vz: i32,           // Velocity z (fixed-point)
  mass: i32,         // Mass (fixed-point)
  temperature: i32,  // Temperature * mass (fixed-point, for averaging)
  thermalMass: i32,  // Mass accumulator for temperature
  heatSource: i32,   // External heat flux (fixed-point)
  _pad: i32,         // Padding to 32 bytes
};
`;

const CELL_ATOMIC_STRUCT = /* wgsl */ `
struct CellAtomic {
  vx: atomic<i32>,
  vy: atomic<i32>,
  vz: atomic<i32>,
  mass: atomic<i32>,
  temperature: atomic<i32>,
  thermalMass: atomic<i32>,
  heatSource: atomic<i32>,
  _pad: i32,
};
`;

const FIXED_POINT_HELPERS = /* wgsl */ `
override fixed_point_multiplier: f32;

fn encodeFixedPoint(f: f32) -> i32 {
  return i32(f * fixed_point_multiplier);
}

fn decodeFixedPoint(v: i32) -> f32 {
  return f32(v) / fixed_point_multiplier;
}
`;

export const CLEAR_GRID_WGSL = /* wgsl */ `
${CELL_STRUCT}

@group(0) @binding(0) var<storage, read_write> cells: array<Cell>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < arrayLength(&cells)) {
    cells[id.x].mass = 0;
    cells[id.x].vx = 0;
    cells[id.x].vy = 0;
    cells[id.x].vz = 0;
    cells[id.x].temperature = 0;
    cells[id.x].thermalMass = 0;
    cells[id.x].heatSource = 0;
  }
}
`;

export const P2G1_WGSL = /* wgsl */ `
${MATERIAL_CONSTANTS}
${PARTICLE_STRUCT}
${CELL_ATOMIC_STRUCT}
${FIXED_POINT_HELPERS}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> cells: array<CellAtomic>;
@group(0) @binding(2) var<uniform> init_box_size: vec3f;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= arrayLength(&particles)) { return; }

  let p = particles[id.x];
  var weights: array<vec3f, 3>;

  let cell_idx: vec3f = floor(p.position);
  let cell_diff: vec3f = p.position - (cell_idx + 0.5);
  weights[0] = 0.5 * (0.5 - cell_diff) * (0.5 - cell_diff);
  weights[1] = 0.75 - cell_diff * cell_diff;
  weights[2] = 0.5 * (0.5 + cell_diff) * (0.5 + cell_diff);

  let C = p.C;

  for (var gx = 0; gx < 3; gx++) {
    for (var gy = 0; gy < 3; gy++) {
      for (var gz = 0; gz < 3; gz++) {
        let weight = weights[gx].x * weights[gy].y * weights[gz].z;
        let cell = vec3f(
          cell_idx.x + f32(gx) - 1.0,
          cell_idx.y + f32(gy) - 1.0,
          cell_idx.z + f32(gz) - 1.0
        );
        let cell_dist = (cell + 0.5) - p.position;
        let Q = C * cell_dist;

        let mass_contrib = weight * p.mass;
        let vel_contrib = mass_contrib * (p.velocity + Q);
        
        // Temperature contribution (weighted by mass for averaging)
        let temp_contrib = mass_contrib * p.temperature;

        let cell_index: i32 =
          i32(cell.x) * i32(init_box_size.y) * i32(init_box_size.z) +
          i32(cell.y) * i32(init_box_size.z) +
          i32(cell.z);
        
        atomicAdd(&cells[cell_index].mass, encodeFixedPoint(mass_contrib));
        atomicAdd(&cells[cell_index].vx, encodeFixedPoint(vel_contrib.x));
        atomicAdd(&cells[cell_index].vy, encodeFixedPoint(vel_contrib.y));
        atomicAdd(&cells[cell_index].vz, encodeFixedPoint(vel_contrib.z));
        
        // Scatter temperature (mass-weighted for proper averaging)
        atomicAdd(&cells[cell_index].temperature, encodeFixedPoint(temp_contrib));
        atomicAdd(&cells[cell_index].thermalMass, encodeFixedPoint(mass_contrib));
      }
    }
  }
}
`;

export const P2G2_WGSL = /* wgsl */ `
${MATERIAL_CONSTANTS}
${PARTICLE_STRUCT}

struct CellAtomic {
  vx: atomic<i32>,
  vy: atomic<i32>,
  vz: atomic<i32>,
  mass: i32,
  temperature: i32,
  thermalMass: i32,
  heatSource: i32,
  _pad: i32,
};

${FIXED_POINT_HELPERS}

override stiffness: f32;
override rest_density: f32;
override dynamic_viscosity: f32;
override dt: f32;
override tensile_strength: f32;
override damage_rate: f32;

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> cells: array<CellAtomic>;
@group(0) @binding(2) var<uniform> init_box_size: vec3f;

// Compute eigenvalues of a symmetric 3x3 matrix (for principal stresses)
fn eigenvalues_symmetric(m: mat3x3f) -> vec3f {
  let a = m[0][0]; let b = m[1][1]; let c = m[2][2];
  let d = m[0][1]; let e = m[1][2]; let f = m[0][2];
  
  let p1 = d*d + e*e + f*f;
  
  if (p1 < 1e-10) {
    return vec3f(a, b, c);
  }
  
  let q = (a + b + c) / 3.0;
  let p2 = (a - q)*(a - q) + (b - q)*(b - q) + (c - q)*(c - q) + 2.0*p1;
  let p = sqrt(p2 / 6.0);
  
  let B00 = (a - q) / p; let B11 = (b - q) / p; let B22 = (c - q) / p;
  let B01 = d / p; let B12 = e / p; let B02 = f / p;
  
  let r = 0.5 * (B00 * (B11*B22 - B12*B12) - B01 * (B01*B22 - B12*B02) + B02 * (B01*B12 - B11*B02));
  let r_clamped = clamp(r, -1.0, 1.0);
  let phi = acos(r_clamped) / 3.0;
  
  let eig0 = q + 2.0 * p * cos(phi);
  let eig2 = q + 2.0 * p * cos(phi + 2.0 * 3.14159265359 / 3.0);
  let eig1 = 3.0 * q - eig0 - eig2;
  
  return vec3f(eig0, eig1, eig2);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= arrayLength(&particles)) { return; }
  var p = particles[id.x];
  var weights: array<vec3f, 3>;

  let cell_idx = floor(p.position);
  let cell_diff = p.position - (cell_idx + 0.5);
  weights[0] = 0.5 * (0.5 - cell_diff) * (0.5 - cell_diff);
  weights[1] = 0.75 - cell_diff * cell_diff;
  weights[2] = 0.5 * (0.5 + cell_diff) * (0.5 + cell_diff);

  // Gather density from grid
  var density = 0.0;
  for (var gx = 0; gx < 3; gx++) {
    for (var gy = 0; gy < 3; gy++) {
      for (var gz = 0; gz < 3; gz++) {
        let weight = weights[gx].x * weights[gy].y * weights[gz].z;
        let cell = vec3f(
          cell_idx.x + f32(gx) - 1.0,
          cell_idx.y + f32(gy) - 1.0,
          cell_idx.z + f32(gz) - 1.0
        );
        let cell_index: i32 =
          i32(cell.x) * i32(init_box_size.y) * i32(init_box_size.z) +
          i32(cell.y) * i32(init_box_size.z) +
          i32(cell.z);
        density += decodeFixedPoint(cells[cell_index].mass) * weight;
      }
    }
  }

  // Constitutive model dispatch
  var stress = mat3x3f(vec3f(0.), vec3f(0.), vec3f(0.));
  var volume: f32;
  let I = mat3x3f(vec3f(1,0,0), vec3f(0,1,0), vec3f(0,0,1));

  switch (p.materialType) {
    case MATERIAL_BRITTLE_SOLID: {
      let F = p.F;
      let J = determinant(F);
      let clampedJ = clamp(J, 0.5, 2.0);  // Prevent extreme compression/expansion
      volume = p.volume0 * clampedJ;
      
      // Use Green-Lagrange strain (better for larger deformations)
      // E = 0.5 * (F^T * F - I)
      let FTF = transpose(F) * F;
      let E = 0.5 * (FTF - I);
      
      let effective_mu = p.mu * (1.0 - p.damage);
      let effective_lambda = p.lambda * (1.0 - p.damage);
      
      let trace_E = E[0][0] + E[1][1] + E[2][2];
      // Second Piola-Kirchhoff stress: S = λ·tr(E)·I + 2μ·E
      let S = effective_lambda * trace_E * I + 2.0 * effective_mu * E;
      // Convert to Cauchy stress: σ = (1/J) * F * S * F^T
      stress = (1.0 / clampedJ) * F * S * transpose(F);
      
      // Clamp stress magnitude to prevent explosion
      let stress_norm = sqrt(
        stress[0][0]*stress[0][0] + stress[1][1]*stress[1][1] + stress[2][2]*stress[2][2] +
        2.0*(stress[0][1]*stress[0][1] + stress[1][2]*stress[1][2] + stress[0][2]*stress[0][2])
      );
      let max_stress = 100.0;  // Maximum allowed stress magnitude
      if (stress_norm > max_stress) {
        stress = stress * (max_stress / stress_norm);
      }
      
      let principal = eigenvalues_symmetric(stress);
      let max_principal = max(max(principal.x, principal.y), principal.z);
      
      if (max_principal > tensile_strength && p.damage < 1.0) {
        let new_damage = p.damage + damage_rate * dt;
        p.damage = min(new_damage, 1.0);
      }
      
      break;
    }
    
    case MATERIAL_ELASTIC_SOLID: {
      let F = p.F;
      let J = determinant(F);
      let clampedJ = max(J, 0.1);
      volume = p.volume0 * clampedJ;
      
      let FFT = F * transpose(F);
      stress = (p.mu / clampedJ) * (FFT - I) + (p.lambda / clampedJ) * log(clampedJ) * I;
      
      break;
    }
    
    case MATERIAL_LIQUID: {
      volume = p.mass / max(density, 1e-6);
      
      let pressure = max(0.0, stiffness * (pow(density / rest_density, 7.0) - 1.0));
      stress = -pressure * I;
      
      let strain_rate = p.C + transpose(p.C);
      stress += dynamic_viscosity * strain_rate;
      
      break;
    }
    
    case MATERIAL_GAS: {
      volume = p.mass / max(density, 1e-8);
      
      let pressure = stiffness * 5.0 * (density / rest_density) * (p.temperature / 273.0);
      stress = -pressure * I;
      
      break;
    }
    
    case MATERIAL_GRANULAR: {
      let F = p.F;
      let J = determinant(F);
      volume = p.volume0 * max(J, 0.1);
      
      let FFT = F * transpose(F);
      let clampedJ = max(J, 0.1);
      stress = (p.mu / clampedJ) * (FFT - I) + (p.lambda / clampedJ) * log(clampedJ) * I;
      
      break;
    }
    
    default: {
      volume = p.mass / max(density, 1e-6);
    }
  }

  particles[id.x] = p;

  let factor = -volume * 4.0 * stress * dt;

  for (var gx = 0; gx < 3; gx++) {
    for (var gy = 0; gy < 3; gy++) {
      for (var gz = 0; gz < 3; gz++) {
        let weight = weights[gx].x * weights[gy].y * weights[gz].z;
        let cell = vec3f(
          cell_idx.x + f32(gx) - 1.0,
          cell_idx.y + f32(gy) - 1.0,
          cell_idx.z + f32(gz) - 1.0
        );
        let cell_dist = (cell + 0.5) - p.position;
        let cell_index: i32 =
          i32(cell.x) * i32(init_box_size.y) * i32(init_box_size.z) +
          i32(cell.y) * i32(init_box_size.z) +
          i32(cell.z);
        let momentum = factor * weight * cell_dist;
        atomicAdd(&cells[cell_index].vx, encodeFixedPoint(momentum.x));
        atomicAdd(&cells[cell_index].vy, encodeFixedPoint(momentum.y));
        atomicAdd(&cells[cell_index].vz, encodeFixedPoint(momentum.z));
      }
    }
  }
}
`;

export const UPDATE_GRID_WGSL = /* wgsl */ `
${CELL_STRUCT}

struct SimulationUniforms {
    gravity: vec3f,
    pad: f32,
};

${FIXED_POINT_HELPERS}
override dt: f32;
override thermal_diffusivity: f32;

@group(0) @binding(0) var<storage, read_write> cells: array<Cell>;
@group(0) @binding(1) var<uniform> real_box_size: vec3f;
@group(0) @binding(2) var<uniform> init_box_size: vec3f;
@group(0) @binding(3) var<uniform> sim_uniforms: SimulationUniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= arrayLength(&cells)) { return; }
  if (cells[id.x].mass <= 0) { return; }

  let cell_mass = decodeFixedPoint(cells[id.x].mass);
  
  // Velocity update
  var v = vec3f(
    decodeFixedPoint(cells[id.x].vx),
    decodeFixedPoint(cells[id.x].vy),
    decodeFixedPoint(cells[id.x].vz)
  );
  v /= cell_mass;
  v += sim_uniforms.gravity * dt;

  cells[id.x].vx = encodeFixedPoint(v.x);
  cells[id.x].vy = encodeFixedPoint(v.y);
  cells[id.x].vz = encodeFixedPoint(v.z);

  // Compute cell coordinates
  let x: i32 = i32(id.x) / i32(init_box_size.z) / i32(init_box_size.y);
  let y: i32 = (i32(id.x) / i32(init_box_size.z)) % i32(init_box_size.y);
  let z: i32 = i32(id.x) % i32(init_box_size.z);

  // Velocity boundary conditions
  if (x < 2 || x > i32(ceil(real_box_size.x) - 3.0)) { cells[id.x].vx = 0; }
  if (y < 2 || y > i32(ceil(real_box_size.y) - 3.0)) { cells[id.x].vy = 0; }
  if (z < 2 || z > i32(ceil(real_box_size.z) - 3.0)) { cells[id.x].vz = 0; }
  
  // Temperature averaging (divide accumulated T*m by total m)
  let thermal_mass = decodeFixedPoint(cells[id.x].thermalMass);
  if (thermal_mass > 1e-6) {
    let avg_temp = decodeFixedPoint(cells[id.x].temperature) / thermal_mass;
    // Add any heat sources
    let heat_flux = decodeFixedPoint(cells[id.x].heatSource);
    let new_temp = avg_temp + heat_flux * dt;
    // Store back as temperature * mass for proper interpolation
    cells[id.x].temperature = encodeFixedPoint(new_temp * thermal_mass);
  }
}
`;

// Thermal diffusion kernel - applies heat diffusion on the grid
export const DIFFUSE_TEMPERATURE_WGSL = /* wgsl */ `
${CELL_STRUCT}

${FIXED_POINT_HELPERS}
override dt: f32;
override thermal_diffusivity: f32;

@group(0) @binding(0) var<storage, read> cells_in: array<Cell>;
@group(0) @binding(1) var<storage, read_write> cells_out: array<Cell>;
@group(0) @binding(2) var<uniform> init_box_size: vec3f;

fn getCellIndex(x: i32, y: i32, z: i32) -> i32 {
  return x * i32(init_box_size.y) * i32(init_box_size.z) + y * i32(init_box_size.z) + z;
}

fn getCellTemp(x: i32, y: i32, z: i32) -> f32 {
  // Clamp to grid bounds
  let cx = clamp(x, 0, i32(init_box_size.x) - 1);
  let cy = clamp(y, 0, i32(init_box_size.y) - 1);
  let cz = clamp(z, 0, i32(init_box_size.z) - 1);
  let idx = getCellIndex(cx, cy, cz);
  
  let thermal_mass = decodeFixedPoint(cells_in[idx].thermalMass);
  if (thermal_mass < 1e-6) {
    return 0.0;
  }
  return decodeFixedPoint(cells_in[idx].temperature) / thermal_mass;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= arrayLength(&cells_in)) { return; }
  
  // Copy all non-temperature fields
  cells_out[id.x].vx = cells_in[id.x].vx;
  cells_out[id.x].vy = cells_in[id.x].vy;
  cells_out[id.x].vz = cells_in[id.x].vz;
  cells_out[id.x].mass = cells_in[id.x].mass;
  cells_out[id.x].thermalMass = cells_in[id.x].thermalMass;
  cells_out[id.x].heatSource = cells_in[id.x].heatSource;
  
  let thermal_mass = decodeFixedPoint(cells_in[id.x].thermalMass);
  if (thermal_mass < 1e-6) {
    cells_out[id.x].temperature = 0;
    return;
  }
  
  // Compute cell coordinates
  let x: i32 = i32(id.x) / i32(init_box_size.z) / i32(init_box_size.y);
  let y: i32 = (i32(id.x) / i32(init_box_size.z)) % i32(init_box_size.y);
  let z: i32 = i32(id.x) % i32(init_box_size.z);
  
  // Get temperatures of neighbors
  let T_center = getCellTemp(x, y, z);
  let T_xp = getCellTemp(x + 1, y, z);
  let T_xm = getCellTemp(x - 1, y, z);
  let T_yp = getCellTemp(x, y + 1, z);
  let T_ym = getCellTemp(x, y - 1, z);
  let T_zp = getCellTemp(x, y, z + 1);
  let T_zm = getCellTemp(x, y, z - 1);
  
  // Laplacian (discrete): ∇²T ≈ Σ(T_neighbor - T_center)
  let laplacian = (T_xp + T_xm + T_yp + T_ym + T_zp + T_zm - 6.0 * T_center);
  
  // Heat equation: ∂T/∂t = α∇²T
  let new_temp = T_center + thermal_diffusivity * dt * laplacian;
  
  // Store as T * mass
  cells_out[id.x].temperature = encodeFixedPoint(new_temp * thermal_mass);
}
`;

export const G2P_WGSL = /* wgsl */ `
${MATERIAL_CONSTANTS}
${PARTICLE_STRUCT}
${CELL_STRUCT}

struct MouseInteraction {
  point: vec3f,
  radius: f32,
  velocity: vec3f,     // For moving heat sources
  temperature: f32,    // Heat source temperature (0 = no thermal effect)
};

${FIXED_POINT_HELPERS}
override dt: f32;

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> cells: array<Cell>;
@group(0) @binding(2) var<uniform> real_box_size: vec3f;
@group(0) @binding(3) var<uniform> init_box_size: vec3f;
@group(0) @binding(4) var<uniform> mouse: MouseInteraction;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= arrayLength(&particles)) { return; }

  var p = particles[id.x];
  p.velocity = vec3f(0.0);
  var new_temperature = 0.0;
  var total_weight = 0.0;

  var weights: array<vec3f, 3>;
  let cell_idx = floor(p.position);
  let cell_diff = p.position - (cell_idx + 0.5);
  weights[0] = 0.5 * (0.5 - cell_diff) * (0.5 - cell_diff);
  weights[1] = 0.75 - cell_diff * cell_diff;
  weights[2] = 0.5 * (0.5 + cell_diff) * (0.5 + cell_diff);

  var B = mat3x3f(vec3f(0.0), vec3f(0.0), vec3f(0.0));
  
  for (var gx = 0; gx < 3; gx++) {
    for (var gy = 0; gy < 3; gy++) {
      for (var gz = 0; gz < 3; gz++) {
        let weight = weights[gx].x * weights[gy].y * weights[gz].z;
        let cell = vec3f(
          cell_idx.x + f32(gx) - 1.0,
          cell_idx.y + f32(gy) - 1.0,
          cell_idx.z + f32(gz) - 1.0
        );
        let cell_dist = (cell + 0.5) - p.position;
        let cell_index: i32 =
          i32(cell.x) * i32(init_box_size.y) * i32(init_box_size.z) +
          i32(cell.y) * i32(init_box_size.z) +
          i32(cell.z);
        
        let weighted_velocity = vec3f(
          decodeFixedPoint(cells[cell_index].vx),
          decodeFixedPoint(cells[cell_index].vy),
          decodeFixedPoint(cells[cell_index].vz)
        ) * weight;

        let term = mat3x3f(
          weighted_velocity * cell_dist.x,
          weighted_velocity * cell_dist.y,
          weighted_velocity * cell_dist.z
        );

        B += term;
        p.velocity += weighted_velocity;
        
        // Gather temperature
        let thermal_mass = decodeFixedPoint(cells[cell_index].thermalMass);
        if (thermal_mass > 1e-6) {
          let cell_temp = decodeFixedPoint(cells[cell_index].temperature) / thermal_mass;
          new_temperature += cell_temp * weight;
          total_weight += weight;
        }
      }
    }
  }

  p.C = B * 4.0;
  
  // Update temperature from grid
  if (total_weight > 0.0) {
    // Blend grid temperature with current particle temperature
    // This smooths out temperature changes
    let grid_temp = new_temperature / total_weight;
    p.temperature = mix(p.temperature, grid_temp, 0.5);
  }
  
  let I = mat3x3f(vec3f(1,0,0), vec3f(0,1,0), vec3f(0,0,1));
  
  // ==========================================
  // DEFORMATION GRADIENT UPDATE
  // ==========================================
  
  switch (p.materialType) {
    case MATERIAL_BRITTLE_SOLID, MATERIAL_ELASTIC_SOLID, MATERIAL_GRANULAR: {
      p.F = (I + dt * p.C) * p.F;
      
      if (p.materialType == MATERIAL_BRITTLE_SOLID && p.damage >= 1.0) {
        p.materialType = MATERIAL_LIQUID;
        p.phase = 1u;
        p.F = I;
        p.damage = 0.0;
      }
      
      break;
    }
    
    case MATERIAL_LIQUID, MATERIAL_GAS: {
      p.F = I;
      break;
    }
    
    default: {
      p.F = I;
    }
  }
  
  // ==========================================
  // PHASE TRANSITIONS (Temperature-based with latent heat)
  // ==========================================
  
  // Handle phase transitions for materials that support it
  if (p.materialType == MATERIAL_LIQUID || p.materialType == MATERIAL_GAS || 
      p.materialType == MATERIAL_BRITTLE_SOLID) {
    
    // Melting: Ice -> Water
    if (p.phase == 0u && p.temperature > T_MELT) {
      // Absorb latent heat
      let excess_temp = p.temperature - T_MELT;
      let latent_consumed = min(excess_temp * SPECIFIC_HEAT, LATENT_HEAT_MELT);
      p.temperature = T_MELT + (excess_temp * SPECIFIC_HEAT - latent_consumed) / SPECIFIC_HEAT;
      
      if (latent_consumed >= LATENT_HEAT_MELT * 0.9) {
        p.phase = 1u;
        p.materialType = MATERIAL_LIQUID;
        p.F = I;
        p.damage = 0.0;
      }
    }
    
    // Freezing: Water -> Ice
    if (p.phase == 1u && p.temperature < T_MELT_LOW) {
      // Release latent heat
      let deficit_temp = T_MELT - p.temperature;
      let latent_released = min(deficit_temp * SPECIFIC_HEAT, LATENT_HEAT_MELT);
      p.temperature = T_MELT - (deficit_temp * SPECIFIC_HEAT - latent_released) / SPECIFIC_HEAT;
      
      if (latent_released >= LATENT_HEAT_MELT * 0.9) {
        p.phase = 0u;
        p.materialType = MATERIAL_BRITTLE_SOLID;
        p.damage = 0.0;
        // Use soft stiffness for explicit integration stability
        p.mu = 50.0;
        p.lambda = 50.0;
      }
    }
    
    // Boiling: Water -> Steam
    if (p.phase == 1u && p.temperature > T_BOIL_HIGH) {
      let excess_temp = p.temperature - T_BOIL;
      let latent_consumed = min(excess_temp * SPECIFIC_HEAT, LATENT_HEAT_BOIL);
      p.temperature = T_BOIL + (excess_temp * SPECIFIC_HEAT - latent_consumed) / SPECIFIC_HEAT;
      
      if (latent_consumed >= LATENT_HEAT_BOIL * 0.9) {
        p.phase = 2u;
        p.materialType = MATERIAL_GAS;
        p.F = I;
      }
    }
    
    // Condensing: Steam -> Water
    if (p.phase == 2u && p.temperature < T_BOIL) {
      let deficit_temp = T_BOIL - p.temperature;
      let latent_released = min(deficit_temp * SPECIFIC_HEAT, LATENT_HEAT_BOIL);
      p.temperature = T_BOIL - (deficit_temp * SPECIFIC_HEAT - latent_released) / SPECIFIC_HEAT;
      
      if (latent_released >= LATENT_HEAT_BOIL * 0.9) {
        p.phase = 1u;
        p.materialType = MATERIAL_LIQUID;
        p.F = I;
      }
    }
  }

  // ==========================================
  // POSITION UPDATE & BOUNDARY CONDITIONS
  // ==========================================
  
  p.position += p.velocity * dt;
  p.position = vec3f(
    clamp(p.position.x, 1.0, real_box_size.x - 2.0),
    clamp(p.position.y, 1.0, real_box_size.y - 2.0),
    clamp(p.position.z, 1.0, real_box_size.z - 2.0)
  );

  // Soft wall boundaries
  let k = 3.0;
  let wall_stiffness = 0.3;
  let wall_min = vec3f(3.0);
  let wall_max = real_box_size - 4.0;
  let x_n = p.position + p.velocity * dt * k;
  if (x_n.x < wall_min.x) { p.velocity.x += wall_stiffness * (wall_min.x - x_n.x); }
  if (x_n.x > wall_max.x) { p.velocity.x += wall_stiffness * (wall_max.x - x_n.x); }
  if (x_n.y < wall_min.y) { p.velocity.y += wall_stiffness * (wall_min.y - x_n.y); }
  if (x_n.y > wall_max.y) { p.velocity.y += wall_stiffness * (wall_max.y - x_n.y); }
  if (x_n.z < wall_min.z) { p.velocity.z += wall_stiffness * (wall_min.z - x_n.z); }
  if (x_n.z > wall_max.z) { p.velocity.z += wall_stiffness * (wall_max.z - x_n.z); }

  // Heat source interaction (hot/cold sphere)
  if (mouse.radius > 0.0) {
    let diff = p.position - mouse.point;
    let dist = length(diff);
    if (dist < mouse.radius) {
      let normal = normalize(diff);
      let penetration = mouse.radius - dist;
      p.position += normal * penetration;
      
      let v_dot_n = dot(p.velocity, normal);
      if (v_dot_n < 0.0) {
        p.velocity -= 1.5 * v_dot_n * normal;
      }
      
      // Apply thermal effect (heat or cool particles near the sphere)
      if (mouse.temperature > 0.0) {
        let thermal_strength = 1.0 - dist / mouse.radius;
        p.temperature = mix(p.temperature, mouse.temperature, thermal_strength * 0.1);
      }
    }
  }

  particles[id.x] = p;
}
`;

export const COPY_POSITION_WGSL = /* wgsl */ `
${MATERIAL_CONSTANTS}
${PARTICLE_STRUCT}

struct PosVelData {
  position: vec3f,
  pad0: f32,
  velocity: vec3f,
  temperature: f32,
};

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> posvel: array<PosVelData>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= arrayLength(&particles)) { return; }
  let p = particles[id.x];
  posvel[id.x].position = p.position;
  posvel[id.x].pad0 = 0.0;
  posvel[id.x].velocity = p.velocity;
  posvel[id.x].temperature = p.temperature;
}
`;
