import{d as I}from"./device-CAsdAK37.js";const y={BRITTLE_SOLID:0,ELASTIC_SOLID:1,LIQUID:2,GAS:3,GRANULAR:4},A=160,u={position:0,materialType:12,velocity:16,phase:28,mass:32,volume0:36,temperature:40,damage:44,F:48,C:96,mu:144,lambda:148},H=32,w=64,q=1e5,b={ice:{mu:1e3,lambda:1e3},water:{stiffness:50},steam:{gasConstant:5},rubber:{mu:10,lambda:100}},U={stiffness:50,restDensity:1,dynamicViscosity:.1,dt:.1,subSteps:4,fixedPointScale:q,tensileStrength:10,damageRate:2,thermalDiffusivity:.05};function F(t){return t*A}function $(t){return t*H}function J(t,e=0){const i=e*A;return{position:new Float32Array(t,i+u.position,3),materialType:new Uint32Array(t,i+u.materialType,1),velocity:new Float32Array(t,i+u.velocity,3),phase:new Uint32Array(t,i+u.phase,1),mass:new Float32Array(t,i+u.mass,1),volume0:new Float32Array(t,i+u.volume0,1),temperature:new Float32Array(t,i+u.temperature,1),damage:new Float32Array(t,i+u.damage,1),F:new Float32Array(t,i+u.F,12),C:new Float32Array(t,i+u.C,12),mu:new Float32Array(t,i+u.mu,1),lambda:new Float32Array(t,i+u.lambda,1)}}function Q(t,e,i){const r=F(e),a=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC;return t.createBuffer({label:"mpm-particles",size:r,usage:a})}function W(t,e,i){const r=$(e),a=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC;return t.createBuffer({label:"mpm-grid",size:r,usage:a})}const D=(t,e)=>Math.ceil(t/e);class V{constructor(e,i={}){this.device=e,this.constants={...U,...i.constants??{}},this.iterations=i.iterations??1,this.pipelines={},this.bindGroups={},this.particleCount=0,this.gridCount=0}configure({pipelines:e,bindGroups:i}){this.pipelines={...e},this.bindGroups={...i}}setCounts({particleCount:e,gridCount:i}){this.particleCount=e??this.particleCount,this.gridCount=i??this.gridCount}step(e,i){if(!e)throw new Error("MpmDomain.step requires a command encoder");if(!this._hasPipelines())throw new Error("MpmDomain pipelines not configured");const r=D(this.particleCount,w),a=D(this.gridCount,w);for(let l=0;l<this.iterations;l+=1)this._runPass(e,"clearGrid",a),this._runPass(e,"p2g1",r),this._runPass(e,"p2g2",r),this._runPass(e,"updateGrid",a),this._runPass(e,"g2p",r),this.pipelines.copyPosition&&this.bindGroups.copyPosition&&this._runPass(e,"copyPosition",r)}_runPass(e,i,r){const a=this.pipelines[i],l=this.bindGroups[i];if(!a||!l)throw new Error(`Missing pipeline or bind group for ${i}`);const o=e.beginComputePass({label:`mpm-${i}`});o.setPipeline(a),o.setBindGroup(0,l),o.dispatchWorkgroups(r),o.end()}_hasPipelines(){return this.pipelines.clearGrid&&this.pipelines.p2g1&&this.pipelines.p2g2&&this.pipelines.updateGrid&&this.pipelines.g2p&&this.bindGroups.clearGrid&&this.bindGroups.p2g1&&this.bindGroups.p2g2&&this.bindGroups.updateGrid&&this.bindGroups.g2p}}const E=`
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
`,z=`
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
`,G=`
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
`,Y=`
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
`,P=`
override fixed_point_multiplier: f32;

fn encodeFixedPoint(f: f32) -> i32 {
  return i32(f * fixed_point_multiplier);
}

fn decodeFixedPoint(v: i32) -> f32 {
  return f32(v) / fixed_point_multiplier;
}
`,j=`
${G}

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
`,K=`
${E}
${z}
${Y}
${P}

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
`,X=`
${E}
${z}

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

${P}

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
      volume = p.volume0 * max(J, 0.1);
      
      let eps = 0.5 * (F + transpose(F)) - I;
      let effective_mu = p.mu * (1.0 - p.damage);
      let effective_lambda = p.lambda * (1.0 - p.damage);
      
      let trace_eps = eps[0][0] + eps[1][1] + eps[2][2];
      stress = effective_lambda * trace_eps * I + 2.0 * effective_mu * eps;
      
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
`,Z=`
${G}

struct SimulationUniforms {
    gravity: vec3f,
    pad: f32,
};

${P}
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
`,ee=`
${E}
${z}
${G}

struct MouseInteraction {
  point: vec3f,
  radius: f32,
  velocity: vec3f,     // For moving heat sources
  temperature: f32,    // Heat source temperature (0 = no thermal effect)
};

${P}
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
        p.mu = 1000.0;
        p.lambda = 1000.0;
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
`,te=`
${E}
${z}

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
`;function ie(t,e=U){const i=I(t,j,"mpm-clear-grid"),r=I(t,K,"mpm-p2g1"),a=I(t,X,"mpm-p2g2"),l=I(t,Z,"mpm-update-grid"),o=I(t,ee,"mpm-g2p"),c=I(t,te,"mpm-copy-position");return{clearGrid:t.createComputePipeline({label:"mpm-clear-grid",layout:"auto",compute:{module:i}}),p2g1:t.createComputePipeline({label:"mpm-p2g1",layout:"auto",compute:{module:r,constants:{fixed_point_multiplier:e.fixedPointScale}}}),p2g2:t.createComputePipeline({label:"mpm-p2g2",layout:"auto",compute:{module:a,constants:{fixed_point_multiplier:e.fixedPointScale,stiffness:e.stiffness,rest_density:e.restDensity,dynamic_viscosity:e.dynamicViscosity,dt:e.dt,tensile_strength:e.tensileStrength,damage_rate:e.damageRate}}}),updateGrid:t.createComputePipeline({label:"mpm-update-grid",layout:"auto",compute:{module:l,constants:{fixed_point_multiplier:e.fixedPointScale,dt:e.dt,thermal_diffusivity:e.thermalDiffusivity??.1}}}),g2p:t.createComputePipeline({label:"mpm-g2p",layout:"auto",compute:{module:o,constants:{fixed_point_multiplier:e.fixedPointScale,dt:e.dt}}}),copyPosition:t.createComputePipeline({label:"mpm-copy-position",layout:"auto",compute:{module:c}})}}function re(t,e,i){const{particleBuffer:r,gridBuffer:a,initBoxBuffer:l,realBoxBuffer:o,interactionBuffer:c,posVelBuffer:m}=i,d={clearGrid:t.createBindGroup({layout:e.clearGrid.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:a}}]}),p2g1:t.createBindGroup({layout:e.p2g1.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:r}},{binding:1,resource:{buffer:a}},{binding:2,resource:{buffer:l}}]}),p2g2:t.createBindGroup({layout:e.p2g2.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:r}},{binding:1,resource:{buffer:a}},{binding:2,resource:{buffer:l}}]}),updateGrid:t.createBindGroup({layout:e.updateGrid.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:a}},{binding:1,resource:{buffer:o}},{binding:2,resource:{buffer:l}},{binding:3,resource:{buffer:i.simUniformBuffer}}]}),g2p:t.createBindGroup({layout:e.g2p.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:r}},{binding:1,resource:{buffer:a}},{binding:2,resource:{buffer:o}},{binding:3,resource:{buffer:l}},{binding:4,resource:{buffer:c}}]})};return e.copyPosition&&m&&(d.copyPosition=t.createBindGroup({layout:e.copyPosition.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:r}},{binding:1,resource:{buffer:m}}]})),d}function M(t,e,i){const r=new Float32Array(4);r.set(e.slice(0,3));const a=t.createBuffer({label:i??"vec3-uniform",size:r.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});return t.queue.writeBuffer(a,0,r),a}function pe(t,e){const{particleCount:i,gridSize:r,posVelBuffer:a,interactionBuffer:l,constants:o,iterations:c}=e;if(!r)throw new Error("gridSize {x,y,z} is required");const m=Math.ceil(r.x)*Math.ceil(r.y)*Math.ceil(r.z),d=Q(t,i),x=W(t,m),f=M(t,[r.x,r.y,r.z],"mpm-init-box"),_=M(t,[r.x,r.y,r.z],"mpm-real-box"),n=M(t,[0,-.3,0],"mpm-sim-uniforms");let s=l;if(!s){s=t.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,label:"mpm-interaction-default"});const L=new Float32Array(8);L[3]=-1,t.queue.writeBuffer(s,0,L)}const p=ie(t,o),v=re(t,p,{particleBuffer:d,gridBuffer:x,initBoxBuffer:f,realBoxBuffer:_,simUniformBuffer:n,interactionBuffer:s,posVelBuffer:a}),T=new V(t,{constants:o,iterations:c});return T.configure({pipelines:p,bindGroups:v}),T.setCounts({particleCount:i,gridCount:m}),{domain:T,pipelines:p,bindGroups:v,buffers:{particleBuffer:d,gridBuffer:x,initBoxBuffer:f,realBoxBuffer:_,simUniformBuffer:n,interactionBuffer:s,posVelBuffer:a},dispatch:{particle:Math.ceil(i/w),grid:Math.ceil(m/w)}}}function de(t,e,i){const r=i.byteLength??i.length;if(r>e.size)throw new Error(`Particle data (${r}) exceeds buffer size (${e.size})`);t.queue.writeBuffer(e,0,i)}const ae=()=>[1,0,0,0,0,1,0,0,0,0,1,0],le=()=>[0,0,0,0,0,0,0,0,0,0,0,0];function se(t){const{count:e,gridSize:i,start:r=[0,0,0],spacing:a=.65,jitter:l=0,materialType:o=y.LIQUID,mass:c=1,temperature:m=300,phase:d=null,mu:x=null,lambda:f=null,restDensity:_=1}=t;if(!e||!i)throw new Error("count and gridSize are required");let n,s,p;switch(o){case y.BRITTLE_SOLID:n=0,s=b.ice.mu,p=b.ice.lambda;break;case y.ELASTIC_SOLID:n=0,s=b.rubber.mu,p=b.rubber.lambda;break;case y.LIQUID:n=1,s=0,p=b.water.stiffness;break;case y.GAS:n=2,s=0,p=b.steam.gasConstant;break;case y.GRANULAR:n=0,s=100,p=100;break;default:n=1,s=0,p=50}const v=d!==null?d:n,T=x!==null?x:s,L=f!==null?f:p,R=new ArrayBuffer(F(e));let h=0;for(let B=r[1];B<i.y&&h<e;B+=a)for(let S=r[0];S<i.x&&h<e;S+=a)for(let C=r[2];C<i.z&&h<e;C+=a){const g=J(R,h),O=l?(Math.random()*2-1)*l:0,k=l?(Math.random()*2-1)*l:0,N=l?(Math.random()*2-1)*l:0;g.position.set([S+O,B+k,C+N]),g.materialType[0]=o,g.velocity.set([0,0,0]),g.phase[0]=v,g.mass[0]=c,g.volume0[0]=c/_,g.temperature[0]=m,g.damage[0]=0,g.F.set(ae()),g.C.set(le()),g.mu[0]=T,g.lambda[0]=L,h+=1}if(h<e)throw new Error(`Could not place all particles; placed ${h} of ${e}`);return R}function oe(t){const{gridSize:e,blocks:i,totalCount:r,spacing:a=.65,jitter:l=0,restDensity:o=1}=t;if(i&&i.length>0){let f=0;for(const s of i)f+=s.count;const _=new ArrayBuffer(F(f));let n=0;for(const s of i){const p=se({count:s.count,gridSize:e,start:s.start,spacing:s.spacing??a,jitter:s.jitter??l,materialType:s.materialType??y.LIQUID,mass:s.mass??1,temperature:s.temperature??300,mu:s.mu,lambda:s.lambda,restDensity:s.restDensity??o}),v=new Uint8Array(p);new Uint8Array(_,n*A).set(v.subarray(0,s.count*A)),n+=s.count}return _}const c=Math.floor(r*.4),d=[{count:r-c,start:[2,2,2],spacing:a,jitter:l,materialType:y.LIQUID,temperature:300,restDensity:o}],x=[e.x*.25,e.y*.4,e.z*.25];return d.push({count:c,start:x,spacing:a,jitter:l*.5,materialType:y.BRITTLE_SOLID,temperature:260,mu:b.ice.mu,lambda:b.ice.lambda,restDensity:.92}),oe({gridSize:e,blocks:d,spacing:a,jitter:l,restDensity:o})}async function ne(t,e,i){var c;const r=i*A,a=t.createBuffer({label:"mpm-particle-staging",size:r,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),l=t.createCommandEncoder({label:"mpm-diagnostics-copy"});l.copyBufferToBuffer(e,0,a,0,r),t.queue.submit([l.finish()]),await a.mapAsync(GPUMapMode.READ);const o=a.getMappedRange().slice(0);return a.unmap(),(c=a.destroy)==null||c.call(a),o}async function ue(t,e,i){const r=await ne(t,e,i),a=u.mass/4,l=u.velocity/4,o=new Float32Array(r);let c=0,m=0,d=0,x=0;for(let f=0;f<i;f+=1){const _=A/4*f,n=o[_+a],s=o[_+l+0],p=o[_+l+1],v=o[_+l+2];c+=n,m+=n*s,d+=n*p,x+=n*v}return{mass:c,momentum:[m,d,x]}}export{ue as a,oe as b,se as c,pe as s,de as u};
