import{d as y}from"./device-CAsdAK37.js";const h=144,g={position:0,materialId:12,velocity:16,phase:28,mass:32,volume:36,temperature:40,F:48,C:96},C=16,z=64,F=1e7,G={stiffness:3,restDensity:4,dynamicViscosity:.1,dt:.2,fixedPointScale:F};function B(e){return e*h}function M(e){return e*C}function S(e,i=0){const t=i*h;return{position:new Float32Array(e,t+g.position,3),materialId:new Float32Array(e,t+g.materialId,1),velocity:new Float32Array(e,t+g.velocity,3),phase:new Float32Array(e,t+g.phase,1),mass:new Float32Array(e,t+g.mass,1),volume:new Float32Array(e,t+g.volume,1),temperature:new Float32Array(e,t+g.temperature,1),F:new Float32Array(e,t+g.F,9),C:new Float32Array(e,t+g.C,9)}}function U(e,i,t){const r=B(i),o=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC;return e.createBuffer({label:"mpm-particles",size:r,usage:o})}function A(e,i,t){const r=M(i),o=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC;return e.createBuffer({label:"mpm-grid",size:r,usage:o})}const P=(e,i)=>Math.ceil(e/i);class D{constructor(i,t={}){this.device=i,this.constants={...G,...t.constants??{}},this.iterations=t.iterations??1,this.pipelines={},this.bindGroups={},this.particleCount=0,this.gridCount=0}configure({pipelines:i,bindGroups:t}){this.pipelines={...i},this.bindGroups={...t}}setCounts({particleCount:i,gridCount:t}){this.particleCount=i??this.particleCount,this.gridCount=t??this.gridCount}step(i,t){if(!i)throw new Error("MpmDomain.step requires a command encoder");if(!this._hasPipelines())throw new Error("MpmDomain pipelines not configured");const r=P(this.particleCount,z),o=P(this.gridCount,z);for(let l=0;l<this.iterations;l+=1)this._runPass(i,"clearGrid",o),this._runPass(i,"p2g1",r),this._runPass(i,"p2g2",r),this._runPass(i,"updateGrid",o),this._runPass(i,"g2p",r),this.pipelines.copyPosition&&this.bindGroups.copyPosition&&this._runPass(i,"copyPosition",r)}_runPass(i,t,r){const o=this.pipelines[t],l=this.bindGroups[t];if(!o||!l)throw new Error(`Missing pipeline or bind group for ${t}`);const s=i.beginComputePass({label:`mpm-${t}`});s.setPipeline(o),s.setBindGroup(0,l),s.dispatchWorkgroups(r),s.end()}_hasPipelines(){return this.pipelines.clearGrid&&this.pipelines.p2g1&&this.pipelines.p2g2&&this.pipelines.updateGrid&&this.pipelines.g2p&&this.bindGroups.clearGrid&&this.bindGroups.p2g1&&this.bindGroups.p2g2&&this.bindGroups.updateGrid&&this.bindGroups.g2p}}const I=`
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
`,L=`
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
`,E=`
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

  let volume = p.mass / max(density, 1e-6);
  let pressure = max(-0.0, stiffness * (pow(density / rest_density, 5.0) - 1.0));

  var stress = mat3x3f(
    vec3f(-pressure, 0.0, 0.0),
    vec3f(0.0, -pressure, 0.0),
    vec3f(0.0, 0.0, -pressure)
  );
  let dudv = p.C;
  let strain = dudv + transpose(dudv);
  stress += dynamic_viscosity * strain;

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
`;function k(e,i=G){const t=y(e,I,"mpm-clear-grid"),r=y(e,L,"mpm-p2g1"),o=y(e,E,"mpm-p2g2"),l=y(e,O,"mpm-update-grid"),s=y(e,R,"mpm-g2p"),c=y(e,T,"mpm-copy-position");return{clearGrid:e.createComputePipeline({label:"mpm-clear-grid",layout:"auto",compute:{module:t}}),p2g1:e.createComputePipeline({label:"mpm-p2g1",layout:"auto",compute:{module:r,constants:{fixed_point_multiplier:i.fixedPointScale}}}),p2g2:e.createComputePipeline({label:"mpm-p2g2",layout:"auto",compute:{module:o,constants:{fixed_point_multiplier:i.fixedPointScale,stiffness:i.stiffness,rest_density:i.restDensity,dynamic_viscosity:i.dynamicViscosity,dt:i.dt}}}),updateGrid:e.createComputePipeline({label:"mpm-update-grid",layout:"auto",compute:{module:l,constants:{fixed_point_multiplier:i.fixedPointScale,dt:i.dt}}}),g2p:e.createComputePipeline({label:"mpm-g2p",layout:"auto",compute:{module:s,constants:{fixed_point_multiplier:i.fixedPointScale,dt:i.dt}}}),copyPosition:e.createComputePipeline({label:"mpm-copy-position",layout:"auto",compute:{module:c}})}}function W(e,i,t){const{particleBuffer:r,gridBuffer:o,initBoxBuffer:l,realBoxBuffer:s,interactionBuffer:c,posVelBuffer:f}=t,m={clearGrid:e.createBindGroup({layout:i.clearGrid.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:o}}]}),p2g1:e.createBindGroup({layout:i.p2g1.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:r}},{binding:1,resource:{buffer:o}},{binding:2,resource:{buffer:l}}]}),p2g2:e.createBindGroup({layout:i.p2g2.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:r}},{binding:1,resource:{buffer:o}},{binding:2,resource:{buffer:l}}]}),updateGrid:e.createBindGroup({layout:i.updateGrid.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:o}},{binding:1,resource:{buffer:s}},{binding:2,resource:{buffer:l}},{binding:3,resource:{buffer:t.simUniformBuffer}}]}),g2p:e.createBindGroup({layout:i.g2p.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:r}},{binding:1,resource:{buffer:o}},{binding:2,resource:{buffer:s}},{binding:3,resource:{buffer:l}},{binding:4,resource:{buffer:c}}]})};return i.copyPosition&&f&&(m.copyPosition=e.createBindGroup({layout:i.copyPosition.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:r}},{binding:1,resource:{buffer:f}}]})),m}function w(e,i,t){const r=new Float32Array(4);r.set(i.slice(0,3));const o=e.createBuffer({label:t??"vec3-uniform",size:r.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});return e.queue.writeBuffer(o,0,r),o}function N(e,i){const{particleCount:t,gridSize:r,posVelBuffer:o,interactionBuffer:l,constants:s,iterations:c}=i;if(!r)throw new Error("gridSize {x,y,z} is required");const f=Math.ceil(r.x)*Math.ceil(r.y)*Math.ceil(r.z),m=U(e,t),_=A(e,f),a=w(e,[r.x,r.y,r.z],"mpm-init-box"),d=w(e,[r.x,r.y,r.z],"mpm-real-box"),u=w(e,[0,-.3,0],"mpm-sim-uniforms");let p=l;if(!p){p=e.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,label:"mpm-interaction-default"});const b=new Float32Array(8);b[3]=-1,e.queue.writeBuffer(p,0,b)}const n=k(e,s),x=W(e,n,{particleBuffer:m,gridBuffer:_,initBoxBuffer:a,realBoxBuffer:d,simUniformBuffer:u,interactionBuffer:p,posVelBuffer:o}),v=new D(e,{constants:s,iterations:c});return v.configure({pipelines:n,bindGroups:x}),v.setCounts({particleCount:t,gridCount:f}),{domain:v,pipelines:n,bindGroups:x,buffers:{particleBuffer:m,gridBuffer:_,initBoxBuffer:a,realBoxBuffer:d,simUniformBuffer:u,interactionBuffer:p,posVelBuffer:o},dispatch:{particle:Math.ceil(t/z),grid:Math.ceil(f/z)}}}function $(e,i,t){const r=t.byteLength??t.length;if(r>i.size)throw new Error(`Particle data (${r}) exceeds buffer size (${i.size})`);e.queue.writeBuffer(i,0,t)}const V=()=>[1,0,0,0,1,0,0,0,1];function j(e){const{count:i,gridSize:t,start:r=[0,0,0],spacing:o=.65,jitter:l=0,materialId:s=0,mass:c=1,temperature:f=273,phase:m=1}=e;if(!i||!t)throw new Error("count and gridSize are required");const _=new ArrayBuffer(B(i));let a=0;for(let d=r[1];d<t.y&&a<i;d+=o)for(let u=r[0];u<t.x&&a<i;u+=o)for(let p=r[2];p<t.z&&a<i;p+=o){const n=S(_,a),x=l?(Math.random()*2-1)*l:0,v=l?(Math.random()*2-1)*l:0,b=l?(Math.random()*2-1)*l:0;n.position.set([u+x,d+v,p+b]),n.materialId[0]=s,n.velocity.set([0,0,0]),n.phase[0]=m,n.mass[0]=c,n.volume[0]=1,n.temperature[0]=f,n.F.set(V()),n.C.fill(0),a+=1}if(a<i)throw new Error(`Could not place all particles; placed ${a} of ${i}`);return _}async function Y(e,i,t){var c;const r=t*h,o=e.createBuffer({label:"mpm-particle-staging",size:r,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),l=e.createCommandEncoder({label:"mpm-diagnostics-copy"});l.copyBufferToBuffer(i,0,o,0,r),e.queue.submit([l.finish()]),await o.mapAsync(GPUMapMode.READ);const s=o.getMappedRange().slice(0);return o.unmap(),(c=o.destroy)==null||c.call(o),s}async function Q(e,i,t){const r=await Y(e,i,t),o=g.mass/4,l=g.velocity/4,s=new Float32Array(r);let c=0,f=0,m=0,_=0;for(let a=0;a<t;a+=1){const d=h/4*a,u=s[d+o],p=s[d+l+0],n=s[d+l+1],x=s[d+l+2];c+=u,f+=u*p,m+=u*n,_+=u*x}return{mass:c,momentum:[f,m,_]}}export{Q as a,j as c,N as s,$ as u};
