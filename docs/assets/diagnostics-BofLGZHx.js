import{d as y}from"./device-CAsdAK37.js";const z=144,u={position:0,materialId:12,velocity:16,phase:28,mass:32,volume:36,temperature:40,F:48,C:96},C=16,b=64,S=1e7,F={stiffness:3,restDensity:4,dynamicViscosity:.1,dt:.2,fixedPointScale:S};function G(e){return e*z}function M(e){return e*C}function I(e,i=0){const t=i*z;return{position:new Float32Array(e,t+u.position,3),materialId:new Float32Array(e,t+u.materialId,1),velocity:new Float32Array(e,t+u.velocity,3),phase:new Float32Array(e,t+u.phase,1),mass:new Float32Array(e,t+u.mass,1),volume:new Float32Array(e,t+u.volume,1),temperature:new Float32Array(e,t+u.temperature,1),F:new Float32Array(e,t+u.F,9),C:new Float32Array(e,t+u.C,9)}}function U(e,i,t){const r=G(i),o=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC;return e.createBuffer({label:"mpm-particles",size:r,usage:o})}function A(e,i,t){const r=M(i),o=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC;return e.createBuffer({label:"mpm-grid",size:r,usage:o})}const w=(e,i)=>Math.ceil(e/i);class L{constructor(i,t={}){this.device=i,this.constants={...F,...t.constants??{}},this.iterations=t.iterations??1,this.pipelines={},this.bindGroups={},this.particleCount=0,this.gridCount=0}configure({pipelines:i,bindGroups:t}){this.pipelines={...i},this.bindGroups={...t}}setCounts({particleCount:i,gridCount:t}){this.particleCount=i??this.particleCount,this.gridCount=t??this.gridCount}step(i,t){if(!i)throw new Error("MpmDomain.step requires a command encoder");if(!this._hasPipelines())throw new Error("MpmDomain pipelines not configured");const r=w(this.particleCount,b),o=w(this.gridCount,b);for(let s=0;s<this.iterations;s+=1)this._runPass(i,"clearGrid",o),this._runPass(i,"p2g1",r),this._runPass(i,"p2g2",r),this._runPass(i,"updateGrid",o),this._runPass(i,"g2p",r),this.pipelines.copyPosition&&this.bindGroups.copyPosition&&this._runPass(i,"copyPosition",r)}_runPass(i,t,r){const o=this.pipelines[t],s=this.bindGroups[t];if(!o||!s)throw new Error(`Missing pipeline or bind group for ${t}`);const l=i.beginComputePass({label:`mpm-${t}`});l.setPipeline(o),l.setBindGroup(0,s),l.dispatchWorkgroups(r),l.end()}_hasPipelines(){return this.pipelines.clearGrid&&this.pipelines.p2g1&&this.pipelines.p2g2&&this.pipelines.updateGrid&&this.pipelines.g2p&&this.bindGroups.clearGrid&&this.bindGroups.p2g1&&this.bindGroups.p2g2&&this.bindGroups.updateGrid&&this.bindGroups.g2p}}const E=`
struct Cell {
  vx: i32,
  vy: i32,
  vz: i32,
  mass: i32,
};

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
`,D=`
struct Particle {
  position: vec3f,
  materialId: f32,
  velocity: vec3f,
  phase: f32,
  mass: f32,
  volume: f32,
  temperature: f32,
  pad0: f32,
  F: mat3x3f,
  C: mat3x3f,
};
struct Cell {
  vx: atomic<i32>,
  vy: atomic<i32>,
  vz: atomic<i32>,
  mass: atomic<i32>,
};

override fixed_point_multiplier: f32;

fn encodeFixedPoint(f: f32) -> i32 {
  return i32(f * fixed_point_multiplier);
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> cells: array<Cell>;
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
`,T=`
struct Particle {
  position: vec3f,
  materialId: f32,
  velocity: vec3f,
  phase: f32,
  mass: f32,
  volume: f32,
  temperature: f32,
  pad0: f32,
  F: mat3x3f,
  C: mat3x3f,
};
struct Cell {
  vx: atomic<i32>,
  vy: atomic<i32>,
  vz: atomic<i32>,
  mass: i32,
};

override fixed_point_multiplier: f32;
override stiffness: f32;
override rest_density: f32;
override dynamic_viscosity: f32;
override dt: f32;

fn encodeFixedPoint(f: f32) -> i32 {
  return i32(f * fixed_point_multiplier);
}
fn decodeFixedPoint(v: i32) -> f32 {
  return f32(v) / fixed_point_multiplier;
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> cells: array<Cell>;
@group(0) @binding(2) var<uniform> init_box_size: vec3f;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= arrayLength(&particles)) { return; }
  let p = particles[id.x];
  var weights: array<vec3f, 3>;

  let cell_idx = floor(p.position);
  let cell_diff = p.position - (cell_idx + 0.5);
  weights[0] = 0.5 * (0.5 - cell_diff) * (0.5 - cell_diff);
  weights[1] = 0.75 - cell_diff * cell_diff;
  weights[2] = 0.5 * (0.5 + cell_diff) * (0.5 + cell_diff);

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

  // Phase 0: Solid (Ice) - Neo-Hookean Elasticity
  // Phase 1: Liquid (Water) - Tait EOS + Viscosity
  // Phase 2: Gas (Steam) - Ideal Gas EOS
  
  var stress = mat3x3f(vec3f(0.), vec3f(0.), vec3f(0.));
  var volume: f32;

  if (p.phase < 0.5) { // Solid
     // For solids, use Lagrangian volume: V = V0 * det(F)
     // p.volume stores V0 (initial volume)
     let F = p.F;
     let J = determinant(F);
     volume = p.volume * J;

     // Neo-Hookean
     // mu, lambda from stiffness
     // Solids need much higher stiffness than fluids to hold shape against gravity/impact
     let mu = stiffness * 100.0; 
     let lambda = stiffness * 100.0;
     
     // Simple Neo-Hookean: 
     // P = mu * (F - F^-T) + lambda * log(J) * F^-T
     // Stress = (1/J) * P * F^T
     // Stress = (mu/J) * (F*F^T - I) + (lambda/J) * log(J) * I
     
     // Approximate if F is close to I?
     // Let's use Corotated or simple approach.
     // For now, let's stick to a simpler elastic model or just high viscosity fluid?
     // No, need elasticity.
     
     // Let's calculate F*F^T
     let FFT = F * transpose(F);
     let I = mat3x3f(vec3f(1,0,0), vec3f(0,1,0), vec3f(0,0,1));
     
     // Clamp J to prevent infinite compression instability (especially with high jitter)
     let clampedJ = max(J, 0.2); 
     stress = (mu / clampedJ) * (FFT - I) + (lambda / clampedJ) * log(clampedJ) * I;
  } else {
     // For fluids/gas, use Eulerian volume from grid density
     volume = p.mass / max(density, 1e-6);
     
     if (p.phase < 1.5) { // Liquid
        let pressure = max(-0.0, stiffness * (pow(density / rest_density, 5.0) - 1.0));
        stress = mat3x3f(
        vec3f(-pressure, 0.0, 0.0),
        vec3f(0.0, -pressure, 0.0),
        vec3f(0.0, 0.0, -pressure)
     );
        let dudv = p.C;
        let strain = dudv + transpose(dudv);
        stress += dynamic_viscosity * strain;
     } else { // Gas
        // Ideal Gas-like EOS: P ~ density * temperature
        // Boost pressure to ensure visual expansion against gravity
        let pressure = stiffness * 5.0 * (density / rest_density) * (p.temperature / 273.0);
        stress = mat3x3f(
            vec3f(-pressure, 0.0, 0.0),
            vec3f(0.0, -pressure, 0.0),
            vec3f(0.0, 0.0, -pressure)
        );
     }
  }

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
`,O=`
struct Cell {
  vx: i32,
  vy: i32,
  vz: i32,
  mass: i32,
};

struct SimulationUniforms {
    gravity: vec3f,
    pad: f32,
};

override fixed_point_multiplier: f32;
override dt: f32;

@group(0) @binding(0) var<storage, read_write> cells: array<Cell>;
@group(0) @binding(1) var<uniform> real_box_size: vec3f;
@group(0) @binding(2) var<uniform> init_box_size: vec3f;
@group(0) @binding(3) var<uniform> sim_uniforms: SimulationUniforms;

fn encodeFixedPoint(f: f32) -> i32 {
  return i32(f * fixed_point_multiplier);
}
fn decodeFixedPoint(v: i32) -> f32 {
  return f32(v) / fixed_point_multiplier;
}

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
`,R=`
struct Particle {
  position: vec3f,
  materialId: f32,
  velocity: vec3f,
  phase: f32,
  mass: f32,
  volume: f32,
  temperature: f32,
  pad0: f32,
  F: mat3x3f,
  C: mat3x3f,
};
struct Cell {
  vx: i32,
  vy: i32,
  vz: i32,
  mass: i32,
};

struct MouseInteraction {
  point: vec3f,
  radius: f32, // if <= 0, disabled
  pad0: vec3f, // padding to 32 bytes?
  pad1: f32,
};

override fixed_point_multiplier: f32;
override dt: f32;

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> cells: array<Cell>;
@group(0) @binding(2) var<uniform> real_box_size: vec3f;
@group(0) @binding(3) var<uniform> init_box_size: vec3f;
@group(0) @binding(4) var<uniform> mouse: MouseInteraction;

fn decodeFixedPoint(v: i32) -> f32 {
  return f32(v) / fixed_point_multiplier;
}

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
  
  // Update Deformation Gradient (Elasticity)
  // F_new = (I + dt * C) * F
  let I = mat3x3f(vec3f(1,0,0), vec3f(0,1,0), vec3f(0,0,1));
  p.F = (I + dt * p.C) * p.F;
  
  // Phase Logic
  // Simple Temp Thresholds
  if (p.temperature < 273.0) {
     p.phase = 0.0; // Ice
     // Maybe harden?
  } else if (p.temperature > 373.0) {
     p.phase = 2.0; // Steam
  } else {
     p.phase = 1.0; // Water
  }
  
  // Plasticity/Fracture handling for solids?
  if (p.phase < 0.5) {
     // Clamp F to avoid infinite stretching?
     // Simple plasticity: if determinant(F) is too wild, reset it?
     // Or Clamp singular values.
  } else {
     // Fluid/Gas: Reset F to Identity to avoid elastic memory
     p.F = I;
  }

  p.position += p.velocity * dt;
  p.position = vec3f(
    clamp(p.position.x, 1.0, real_box_size.x - 2.0),
    clamp(p.position.y, 1.0, real_box_size.y - 2.0),
    clamp(p.position.z, 1.0, real_box_size.z - 2.0)
  );

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
      // Push out
      p.position += normal * penetration;
      
      // Reflect velocity? Or just push?
      // Simple bounce:
      let v_dot_n = dot(p.velocity, normal);
      if (v_dot_n < 0.0) {
        p.velocity -= 1.5 * v_dot_n * normal; // 1.5 restitution?
      }
      
      // Friction?
      // let tangent = p.velocity - dot(p.velocity, normal) * normal;
      // p.velocity -= tangent * 0.1;
    }
  }

  particles[id.x] = p;
}
`,k=`
struct Particle {
  position: vec3f,
  materialId: f32,
  velocity: vec3f,
  phase: f32,
  mass: f32,
  volume: f32,
  temperature: f32,
  pad0: f32,
  F: mat3x3f,
  C: mat3x3f,
};
struct PosVel {
  position: vec3f,
  velocity: vec3f,
};

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> posvel: array<PosVel>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= arrayLength(&particles)) { return; }
  posvel[id.x].position = particles[id.x].position;
  posvel[id.x].velocity = particles[id.x].velocity;
}
`;function J(e,i=F){const t=y(e,E,"mpm-clear-grid"),r=y(e,D,"mpm-p2g1"),o=y(e,T,"mpm-p2g2"),s=y(e,O,"mpm-update-grid"),l=y(e,R,"mpm-g2p"),a=y(e,k,"mpm-copy-position");return{clearGrid:e.createComputePipeline({label:"mpm-clear-grid",layout:"auto",compute:{module:t}}),p2g1:e.createComputePipeline({label:"mpm-p2g1",layout:"auto",compute:{module:r,constants:{fixed_point_multiplier:i.fixedPointScale}}}),p2g2:e.createComputePipeline({label:"mpm-p2g2",layout:"auto",compute:{module:o,constants:{fixed_point_multiplier:i.fixedPointScale,stiffness:i.stiffness,rest_density:i.restDensity,dynamic_viscosity:i.dynamicViscosity,dt:i.dt}}}),updateGrid:e.createComputePipeline({label:"mpm-update-grid",layout:"auto",compute:{module:s,constants:{fixed_point_multiplier:i.fixedPointScale,dt:i.dt}}}),g2p:e.createComputePipeline({label:"mpm-g2p",layout:"auto",compute:{module:l,constants:{fixed_point_multiplier:i.fixedPointScale,dt:i.dt}}}),copyPosition:e.createComputePipeline({label:"mpm-copy-position",layout:"auto",compute:{module:a}})}}function V(e,i,t){const{particleBuffer:r,gridBuffer:o,initBoxBuffer:s,realBoxBuffer:l,interactionBuffer:a,posVelBuffer:f}=t,m={clearGrid:e.createBindGroup({layout:i.clearGrid.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:o}}]}),p2g1:e.createBindGroup({layout:i.p2g1.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:r}},{binding:1,resource:{buffer:o}},{binding:2,resource:{buffer:s}}]}),p2g2:e.createBindGroup({layout:i.p2g2.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:r}},{binding:1,resource:{buffer:o}},{binding:2,resource:{buffer:s}}]}),updateGrid:e.createBindGroup({layout:i.updateGrid.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:o}},{binding:1,resource:{buffer:l}},{binding:2,resource:{buffer:s}},{binding:3,resource:{buffer:t.simUniformBuffer}}]}),g2p:e.createBindGroup({layout:i.g2p.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:r}},{binding:1,resource:{buffer:o}},{binding:2,resource:{buffer:l}},{binding:3,resource:{buffer:s}},{binding:4,resource:{buffer:a}}]})};return i.copyPosition&&f&&(m.copyPosition=e.createBindGroup({layout:i.copyPosition.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:r}},{binding:1,resource:{buffer:f}}]})),m}function P(e,i,t){const r=new Float32Array(4);r.set(i.slice(0,3));const o=e.createBuffer({label:t??"vec3-uniform",size:r.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});return e.queue.writeBuffer(o,0,r),o}function Y(e,i){const{particleCount:t,gridSize:r,posVelBuffer:o,interactionBuffer:s,constants:l,iterations:a}=i;if(!r)throw new Error("gridSize {x,y,z} is required");const f=Math.ceil(r.x)*Math.ceil(r.y)*Math.ceil(r.z),m=U(e,t),x=A(e,f),g=P(e,[r.x,r.y,r.z],"mpm-init-box"),n=P(e,[r.x,r.y,r.z],"mpm-real-box"),d=P(e,[0,-.3,0],"mpm-sim-uniforms");let p=s;if(!p){p=e.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,label:"mpm-interaction-default"});const h=new Float32Array(8);h[3]=-1,e.queue.writeBuffer(p,0,h)}const _=J(e,l),c=V(e,_,{particleBuffer:m,gridBuffer:x,initBoxBuffer:g,realBoxBuffer:n,simUniformBuffer:d,interactionBuffer:p,posVelBuffer:o}),v=new L(e,{constants:l,iterations:a});return v.configure({pipelines:_,bindGroups:c}),v.setCounts({particleCount:t,gridCount:f}),{domain:v,pipelines:_,bindGroups:c,buffers:{particleBuffer:m,gridBuffer:x,initBoxBuffer:g,realBoxBuffer:n,simUniformBuffer:d,interactionBuffer:p,posVelBuffer:o},dispatch:{particle:Math.ceil(t/b),grid:Math.ceil(f/b)}}}function j(e,i,t){const r=t.byteLength??t.length;if(r>i.size)throw new Error(`Particle data (${r}) exceeds buffer size (${i.size})`);e.queue.writeBuffer(i,0,t)}const W=()=>[1,0,0,0,1,0,0,0,1];function $(e){const{count:i,gridSize:t,start:r=[0,0,0],spacing:o=.65,jitter:s=0,materialId:l=0,mass:a=1,temperature:f=273,phase:m=1,restDensity:x=4}=e;if(!i||!t)throw new Error("count and gridSize are required");const g=new ArrayBuffer(G(i));let n=0;for(let d=r[1];d<t.y&&n<i;d+=o)for(let p=r[0];p<t.x&&n<i;p+=o)for(let _=r[2];_<t.z&&n<i;_+=o){const c=I(g,n),v=s?(Math.random()*2-1)*s:0,h=s?(Math.random()*2-1)*s:0,B=s?(Math.random()*2-1)*s:0;c.position.set([p+v,d+h,_+B]),c.materialId[0]=l,c.velocity.set([0,0,0]),c.phase[0]=m,c.mass[0]=a,c.volume[0]=a/x,c.temperature[0]=f,c.F.set(W()),c.C.fill(0),n+=1}if(n<i)throw new Error(`Could not place all particles; placed ${n} of ${i}`);return g}async function N(e,i,t){var a;const r=t*z,o=e.createBuffer({label:"mpm-particle-staging",size:r,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),s=e.createCommandEncoder({label:"mpm-diagnostics-copy"});s.copyBufferToBuffer(i,0,o,0,r),e.queue.submit([s.finish()]),await o.mapAsync(GPUMapMode.READ);const l=o.getMappedRange().slice(0);return o.unmap(),(a=o.destroy)==null||a.call(o),l}async function H(e,i,t){const r=await N(e,i,t),o=u.mass/4,s=u.velocity/4,l=new Float32Array(r);let a=0,f=0,m=0,x=0;for(let g=0;g<t;g+=1){const n=z/4*g,d=l[n+o],p=l[n+s+0],_=l[n+s+1],c=l[n+s+2];a+=d,f+=d*p,m+=d*_,x+=d*c}return{mass:a,momentum:[f,m,x]}}export{H as a,$ as c,Y as s,j as u};
