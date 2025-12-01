// WGSL shader sources for MLS-MPM (multi-material constitutive framework).
// Supports: BRITTLE_SOLID (linear elastic + fracture), ELASTIC_SOLID (Neo-Hookean),
//           LIQUID (Tait EOS), GAS (Ideal Gas), GRANULAR (Drucker-Prager - future)

// Material type constants (must match schema.js MATERIAL_TYPE enum)
const MATERIAL_CONSTANTS = /* wgsl */ `
const MATERIAL_BRITTLE_SOLID: u32 = 0u;
const MATERIAL_ELASTIC_SOLID: u32 = 1u;
const MATERIAL_LIQUID: u32 = 2u;
const MATERIAL_GAS: u32 = 3u;
const MATERIAL_GRANULAR: u32 = 4u;
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

const CELL_STRUCT = /* wgsl */ `
struct Cell {
  vx: i32,
  vy: i32,
  vz: i32,
  mass: i32,
};
`;

const CELL_ATOMIC_STRUCT = /* wgsl */ `
struct CellAtomic {
  vx: atomic<i32>,
  vy: atomic<i32>,
  vz: atomic<i32>,
  mass: atomic<i32>,
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
  }
}
`;

export const P2G1_WGSL = /* wgsl */ `
${MATERIAL_CONSTANTS}
${PARTICLE_STRUCT}

struct CellAtomic {
  vx: atomic<i32>,
  vy: atomic<i32>,
  vz: atomic<i32>,
  mass: atomic<i32>,
};

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

        let cell_index: i32 =
          i32(cell.x) * i32(init_box_size.y) * i32(init_box_size.z) +
          i32(cell.y) * i32(init_box_size.z) +
          i32(cell.z);
        atomicAdd(&cells[cell_index].mass, encodeFixedPoint(mass_contrib));
        atomicAdd(&cells[cell_index].vx, encodeFixedPoint(vel_contrib.x));
        atomicAdd(&cells[cell_index].vy, encodeFixedPoint(vel_contrib.y));
        atomicAdd(&cells[cell_index].vz, encodeFixedPoint(vel_contrib.z));
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
// Using analytical solution for symmetric matrices
fn eigenvalues_symmetric(m: mat3x3f) -> vec3f {
  let a = m[0][0]; let b = m[1][1]; let c = m[2][2];
  let d = m[0][1]; let e = m[1][2]; let f = m[0][2];
  
  let p1 = d*d + e*e + f*f;
  
  if (p1 < 1e-10) {
    // Matrix is diagonal
    return vec3f(a, b, c);
  }
  
  let q = (a + b + c) / 3.0;
  let p2 = (a - q)*(a - q) + (b - q)*(b - q) + (c - q)*(c - q) + 2.0*p1;
  let p = sqrt(p2 / 6.0);
  
  // B = (1/p) * (A - q*I)
  let B00 = (a - q) / p; let B11 = (b - q) / p; let B22 = (c - q) / p;
  let B01 = d / p; let B12 = e / p; let B02 = f / p;
  
  // det(B) / 2
  let r = 0.5 * (B00 * (B11*B22 - B12*B12) - B01 * (B01*B22 - B12*B02) + B02 * (B01*B12 - B11*B02));
  
  // Clamp r to [-1, 1] for numerical stability
  let r_clamped = clamp(r, -1.0, 1.0);
  let phi = acos(r_clamped) / 3.0;
  
  // Eigenvalues
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

  // ==========================================
  // CONSTITUTIVE MODEL DISPATCH
  // ==========================================
  var stress = mat3x3f(vec3f(0.), vec3f(0.), vec3f(0.));
  var volume: f32;
  let I = mat3x3f(vec3f(1,0,0), vec3f(0,1,0), vec3f(0,0,1));

  switch (p.materialType) {
    // ----------------------------------------
    // BRITTLE SOLID (Ice, Glass)
    // Linear Elastic + Fracture
    // ----------------------------------------
    case MATERIAL_BRITTLE_SOLID: {
      let F = p.F;
      let J = determinant(F);
      volume = p.volume0 * max(J, 0.1);
      
      // Small strain tensor: ε = (F + F^T)/2 - I
      let eps = 0.5 * (F + transpose(F)) - I;
      
      // Apply damage to material properties
      let effective_mu = p.mu * (1.0 - p.damage);
      let effective_lambda = p.lambda * (1.0 - p.damage);
      
      // Linear elastic stress: σ = λ·tr(ε)·I + 2μ·ε
      let trace_eps = eps[0][0] + eps[1][1] + eps[2][2];
      stress = effective_lambda * trace_eps * I + 2.0 * effective_mu * eps;
      
      // Principal stress fracture criterion
      // Compute principal stresses (eigenvalues of stress tensor)
      let principal = eigenvalues_symmetric(stress);
      let max_principal = max(max(principal.x, principal.y), principal.z);
      
      // Accumulate damage if tensile strength exceeded
      if (max_principal > tensile_strength && p.damage < 1.0) {
        let new_damage = p.damage + damage_rate * dt;
        p.damage = min(new_damage, 1.0);
      }
      
      break;
    }
    
    // ----------------------------------------
    // ELASTIC SOLID (Rubber)
    // Neo-Hookean
    // ----------------------------------------
    case MATERIAL_ELASTIC_SOLID: {
      let F = p.F;
      let J = determinant(F);
      let clampedJ = max(J, 0.1);
      volume = p.volume0 * clampedJ;
      
      // Neo-Hookean: σ = (μ/J)(FF^T - I) + (λ/J)ln(J)I
      let FFT = F * transpose(F);
      stress = (p.mu / clampedJ) * (FFT - I) + (p.lambda / clampedJ) * log(clampedJ) * I;
      
      break;
    }
    
    // ----------------------------------------
    // LIQUID (Water)
    // Tait Equation of State + Viscosity
    // ----------------------------------------
    case MATERIAL_LIQUID: {
      // Eulerian volume from grid density
      volume = p.mass / max(density, 1e-6);
      
      // Tait EOS: P = B * ((ρ/ρ₀)^γ - 1)
      let pressure = max(0.0, stiffness * (pow(density / rest_density, 7.0) - 1.0));
      stress = -pressure * I;
      
      // Viscosity: σ += μ·(∇v + ∇v^T)
      let strain_rate = p.C + transpose(p.C);
      stress += dynamic_viscosity * strain_rate;
      
      break;
    }
    
    // ----------------------------------------
    // GAS (Steam, Air)
    // Ideal Gas Law: P = ρRT
    // ----------------------------------------
    case MATERIAL_GAS: {
      // Eulerian volume from grid density
      volume = p.mass / max(density, 1e-8);
      
      // Ideal Gas: P ∝ ρ * T
      // Using temperature-proportional pressure for expansion
      let pressure = stiffness * 5.0 * (density / rest_density) * (p.temperature / 273.0);
      stress = -pressure * I;
      
      break;
    }
    
    // ----------------------------------------
    // GRANULAR (Sand, Snow) - Future
    // Drucker-Prager
    // ----------------------------------------
    case MATERIAL_GRANULAR: {
      // TODO: Implement Drucker-Prager elastoplasticity
      // For now, treat as elastic solid
      let F = p.F;
      let J = determinant(F);
      volume = p.volume0 * max(J, 0.1);
      
      let FFT = F * transpose(F);
      let clampedJ = max(J, 0.1);
      stress = (p.mu / clampedJ) * (FFT - I) + (p.lambda / clampedJ) * log(clampedJ) * I;
      
      break;
    }
    
    default: {
      // Unknown material - no stress
      volume = p.mass / max(density, 1e-6);
    }
  }

  // Write back updated particle (damage may have changed)
  particles[id.x] = p;

  // Apply stress to grid
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

@group(0) @binding(0) var<storage, read_write> cells: array<Cell>;
@group(0) @binding(1) var<uniform> real_box_size: vec3f;
@group(0) @binding(2) var<uniform> init_box_size: vec3f;
@group(0) @binding(3) var<uniform> sim_uniforms: SimulationUniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= arrayLength(&cells)) { return; }
  if (cells[id.x].mass <= 0) { return; }

  var v = vec3f(
    decodeFixedPoint(cells[id.x].vx),
    decodeFixedPoint(cells[id.x].vy),
    decodeFixedPoint(cells[id.x].vz)
  );
  v /= decodeFixedPoint(cells[id.x].mass);
  
  // Apply gravity
  v += sim_uniforms.gravity * dt;

  cells[id.x].vx = encodeFixedPoint(v.x);
  cells[id.x].vy = encodeFixedPoint(v.y);
  cells[id.x].vz = encodeFixedPoint(v.z);

  let x: i32 = i32(id.x) / i32(init_box_size.z) / i32(init_box_size.y);
  let y: i32 = (i32(id.x) / i32(init_box_size.z)) % i32(init_box_size.y);
  let z: i32 = i32(id.x) % i32(init_box_size.z);

  if (x < 2 || x > i32(ceil(real_box_size.x) - 3.0)) { cells[id.x].vx = 0; }
  if (y < 2 || y > i32(ceil(real_box_size.y) - 3.0)) { cells[id.x].vy = 0; }
  if (z < 2 || z > i32(ceil(real_box_size.z) - 3.0)) { cells[id.x].vz = 0; }
}
`;

export const G2P_WGSL = /* wgsl */ `
${MATERIAL_CONSTANTS}
${PARTICLE_STRUCT}
${CELL_STRUCT}

struct MouseInteraction {
  point: vec3f,
  radius: f32,
  pad0: vec3f,
  pad1: f32,
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
      }
    }
  }

  p.C = B * 4.0;
  
  let I = mat3x3f(vec3f(1,0,0), vec3f(0,1,0), vec3f(0,0,1));
  
  // ==========================================
  // DEFORMATION GRADIENT UPDATE
  // ==========================================
  
  // Update F based on material type
  switch (p.materialType) {
    case MATERIAL_BRITTLE_SOLID, MATERIAL_ELASTIC_SOLID, MATERIAL_GRANULAR: {
      // Solids: Evolve deformation gradient
      // F_new = (I + dt * C) * F
      p.F = (I + dt * p.C) * p.F;
      
      // For brittle solids with full damage, convert to liquid
      if (p.materialType == MATERIAL_BRITTLE_SOLID && p.damage >= 1.0) {
        p.materialType = MATERIAL_LIQUID;
        p.phase = 1u;  // Liquid phase
        p.F = I;       // Reset deformation gradient
        p.damage = 0.0;
      }
      
      break;
    }
    
    case MATERIAL_LIQUID, MATERIAL_GAS: {
      // Fluids: Reset F to identity (no elastic memory)
      p.F = I;
      break;
    }
    
    default: {
      p.F = I;
    }
  }
  
  // ==========================================
  // PHASE TRANSITIONS (Temperature-based)
  // ==========================================
  
  // Only apply phase transitions to materials that support it
  // (LIQUID, GAS, or fully damaged BRITTLE_SOLID)
  if (p.materialType == MATERIAL_LIQUID || p.materialType == MATERIAL_GAS) {
    // Hysteresis to prevent oscillation
    if (p.temperature < 271.0 && p.phase != 0u) {
      // Freeze: become ice
      p.phase = 0u;
      p.materialType = MATERIAL_BRITTLE_SOLID;
      p.damage = 0.0;
      // Set ice material properties
      p.mu = 1000.0;
      p.lambda = 1000.0;
    } else if (p.temperature > 375.0 && p.phase != 2u) {
      // Boil: become steam
      p.phase = 2u;
      p.materialType = MATERIAL_GAS;
      p.F = I;
    } else if (p.temperature >= 273.0 && p.temperature <= 373.0 && p.phase != 1u) {
      // Liquid range
      p.phase = 1u;
      p.materialType = MATERIAL_LIQUID;
      p.F = I;
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

  // Sphere interaction
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
  pad0: f32,             // Alignment padding
  velocity: vec3f,
  temperature: f32,      // For rendering coloring
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
