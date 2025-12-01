import{d as m}from"./device-CAsdAK37.js";const v=128,u={position:0,materialId:12,velocity:16,phase:28,mass:32,volume:36,temperature:40,F:48,C:84},B=16,y=64,F=1e7,w={stiffness:3,restDensity:4,dynamicViscosity:.1,dt:.2,fixedPointScale:F};function P(i){return i*v}function M(i){return i*B}function S(i,e=0){const t=e*v;return{position:new Float32Array(i,t+u.position,3),materialId:new Float32Array(i,t+u.materialId,1),velocity:new Float32Array(i,t+u.velocity,3),phase:new Float32Array(i,t+u.phase,1),mass:new Float32Array(i,t+u.mass,1),volume:new Float32Array(i,t+u.volume,1),temperature:new Float32Array(i,t+u.temperature,1),F:new Float32Array(i,t+u.F,9),C:new Float32Array(i,t+u.C,9)}}function A(i,e,t){const r=P(e),l=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC;return i.createBuffer({label:"mpm-particles",size:r,usage:l})}function U(i,e,t){const r=M(e),l=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC;return i.createBuffer({label:"mpm-grid",size:r,usage:l})}const b=(i,e)=>Math.ceil(i/e);class L{constructor(e,t={}){this.device=e,this.constants={...w,...t.constants??{}},this.iterations=t.iterations??1,this.pipelines={},this.bindGroups={},this.particleCount=0,this.gridCount=0}configure({pipelines:e,bindGroups:t}){this.pipelines={...e},this.bindGroups={...t}}setCounts({particleCount:e,gridCount:t}){this.particleCount=e??this.particleCount,this.gridCount=t??this.gridCount}step(e,t){if(!e)throw new Error("MpmDomain.step requires a command encoder");if(!this._hasPipelines())throw new Error("MpmDomain pipelines not configured");const r=b(this.particleCount,y),l=b(this.gridCount,y);for(let o=0;o<this.iterations;o+=1)this._runPass(e,"clearGrid",l),this._runPass(e,"p2g1",r),this._runPass(e,"p2g2",r),this._runPass(e,"updateGrid",l),this._runPass(e,"g2p",r),this.pipelines.copyPosition&&this.bindGroups.copyPosition&&this._runPass(e,"copyPosition",r)}_runPass(e,t,r){const l=this.pipelines[t],o=this.bindGroups[t];if(!l||!o)throw new Error(`Missing pipeline or bind group for ${t}`);const s=e.beginComputePass({label:`mpm-${t}`});s.setPipeline(l),s.setBindGroup(0,o),s.dispatchWorkgroups(r),s.end()}_hasPipelines(){return this.pipelines.clearGrid&&this.pipelines.p2g1&&this.pipelines.p2g2&&this.pipelines.updateGrid&&this.pipelines.g2p&&this.bindGroups.clearGrid&&this.bindGroups.p2g1&&this.bindGroups.p2g2&&this.bindGroups.updateGrid&&this.bindGroups.g2p}}const D=`
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
  pad1: vec2f,
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
`,I=`
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
  pad1: vec2f,
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
`,R=`
struct Cell {
  vx: i32,
  vy: i32,
  vz: i32,
  mass: i32,
};

override fixed_point_multiplier: f32;
override dt: f32;

@group(0) @binding(0) var<storage, read_write> cells: array<Cell>;
@group(0) @binding(1) var<uniform> real_box_size: vec3f;
@group(0) @binding(2) var<uniform> init_box_size: vec3f;

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
  cells[id.x].vx = encodeFixedPoint(v.x);
  cells[id.x].vy = encodeFixedPoint(v.y + (-0.3 * dt));
  cells[id.x].vz = encodeFixedPoint(v.z);

  let x: i32 = i32(id.x) / i32(init_box_size.z) / i32(init_box_size.y);
  let y: i32 = (i32(id.x) / i32(init_box_size.z)) % i32(init_box_size.y);
  let z: i32 = i32(id.x) % i32(init_box_size.z);

  if (x < 2 || x > i32(ceil(real_box_size.x) - 3.0)) { cells[id.x].vx = 0; }
  if (y < 2 || y > i32(ceil(real_box_size.y) - 3.0)) { cells[id.x].vy = 0; }
  if (z < 2 || z > i32(ceil(real_box_size.z) - 3.0)) { cells[id.x].vz = 0; }
}
`,O=`
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
  pad1: vec2f,
};
struct Cell {
  vx: i32,
  vy: i32,
  vz: i32,
  mass: i32,
};

override fixed_point_multiplier: f32;
override dt: f32;

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> cells: array<Cell>;
@group(0) @binding(2) var<uniform> real_box_size: vec3f;
@group(0) @binding(3) var<uniform> init_box_size: vec3f;

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
  pad1: vec2f,
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
`;function k(i,e=w){const t=m(i,D,"mpm-clear-grid"),r=m(i,E,"mpm-p2g1"),l=m(i,I,"mpm-p2g2"),o=m(i,R,"mpm-update-grid"),s=m(i,O,"mpm-g2p"),n=m(i,T,"mpm-copy-position");return{clearGrid:i.createComputePipeline({label:"mpm-clear-grid",layout:"auto",compute:{module:t}}),p2g1:i.createComputePipeline({label:"mpm-p2g1",layout:"auto",compute:{module:r,constants:{fixed_point_multiplier:e.fixedPointScale}}}),p2g2:i.createComputePipeline({label:"mpm-p2g2",layout:"auto",compute:{module:l,constants:{fixed_point_multiplier:e.fixedPointScale,stiffness:e.stiffness,rest_density:e.restDensity,dynamic_viscosity:e.dynamicViscosity,dt:e.dt}}}),updateGrid:i.createComputePipeline({label:"mpm-update-grid",layout:"auto",compute:{module:o,constants:{fixed_point_multiplier:e.fixedPointScale,dt:e.dt}}}),g2p:i.createComputePipeline({label:"mpm-g2p",layout:"auto",compute:{module:s,constants:{fixed_point_multiplier:e.fixedPointScale,dt:e.dt}}}),copyPosition:i.createComputePipeline({label:"mpm-copy-position",layout:"auto",compute:{module:n}})}}function W(i,e,t){const{particleBuffer:r,gridBuffer:l,initBoxBuffer:o,realBoxBuffer:s,posVelBuffer:n}=t,d={clearGrid:i.createBindGroup({layout:e.clearGrid.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:l}}]}),p2g1:i.createBindGroup({layout:e.p2g1.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:r}},{binding:1,resource:{buffer:l}},{binding:2,resource:{buffer:o}}]}),p2g2:i.createBindGroup({layout:e.p2g2.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:r}},{binding:1,resource:{buffer:l}},{binding:2,resource:{buffer:o}}]}),updateGrid:i.createBindGroup({layout:e.updateGrid.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:l}},{binding:1,resource:{buffer:s}},{binding:2,resource:{buffer:o}}]}),g2p:i.createBindGroup({layout:e.g2p.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:r}},{binding:1,resource:{buffer:l}},{binding:2,resource:{buffer:s}},{binding:3,resource:{buffer:o}}]})};return e.copyPosition&&n&&(d.copyPosition=i.createBindGroup({layout:e.copyPosition.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:r}},{binding:1,resource:{buffer:n}}]})),d}function h(i,e,t){const r=new Float32Array(4);r.set(e.slice(0,3));const l=i.createBuffer({label:t??"vec3-uniform",size:r.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});return i.queue.writeBuffer(l,0,r),l}function $(i,e){const{particleCount:t,gridSize:r,posVelBuffer:l,constants:o,iterations:s}=e;if(!r)throw new Error("gridSize {x,y,z} is required");const n=Math.ceil(r.x)*Math.ceil(r.y)*Math.ceil(r.z),d=A(i,t),x=U(i,n),g=h(i,[r.x,r.y,r.z],"mpm-init-box"),a=h(i,[r.x,r.y,r.z],"mpm-real-box"),c=k(i,o),f=W(i,c,{particleBuffer:d,gridBuffer:x,initBoxBuffer:g,realBoxBuffer:a,posVelBuffer:l}),_=new L(i,{constants:o,iterations:s});return _.configure({pipelines:c,bindGroups:f}),_.setCounts({particleCount:t,gridCount:n}),{domain:_,pipelines:c,bindGroups:f,buffers:{particleBuffer:d,gridBuffer:x,initBoxBuffer:g,realBoxBuffer:a,posVelBuffer:l},dispatch:{particle:Math.ceil(t/y),grid:Math.ceil(n/y)}}}function N(i,e,t){const r=t.byteLength??t.length;if(r>e.size)throw new Error(`Particle data (${r}) exceeds buffer size (${e.size})`);i.queue.writeBuffer(e,0,t)}const V=()=>[1,0,0,0,1,0,0,0,1];function j(i){const{count:e,gridSize:t,start:r=[0,0,0],spacing:l=.65,jitter:o=0,materialId:s=0,mass:n=1,temperature:d=273,phase:x=1}=i;if(!e||!t)throw new Error("count and gridSize are required");const g=new ArrayBuffer(P(e));let a=0;for(let c=r[1];c<t.y&&a<e;c+=l)for(let f=r[0];f<t.x&&a<e;f+=l)for(let _=r[2];_<t.z&&a<e;_+=l){const p=S(g,a),z=o?(Math.random()*2-1)*o:0,G=o?(Math.random()*2-1)*o:0,C=o?(Math.random()*2-1)*o:0;p.position.set([f+z,c+G,_+C]),p.materialId[0]=s,p.velocity.set([0,0,0]),p.phase[0]=x,p.mass[0]=n,p.volume[0]=1,p.temperature[0]=d,p.F.set(V()),p.C.fill(0),a+=1}if(a<e)throw new Error(`Could not place all particles; placed ${a} of ${e}`);return g}async function Y(i,e,t){var n;const r=t*v,l=i.createBuffer({label:"mpm-particle-staging",size:r,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),o=i.createCommandEncoder({label:"mpm-diagnostics-copy"});o.copyBufferToBuffer(e,0,l,0,r),i.queue.submit([o.finish()]),await l.mapAsync(GPUMapMode.READ);const s=l.getMappedRange().slice(0);return l.unmap(),(n=l.destroy)==null||n.call(l),s}async function Q(i,e,t){const r=await Y(i,e,t),l=u.mass/4,o=u.velocity/4,s=new Float32Array(r);let n=0,d=0,x=0,g=0;for(let a=0;a<t;a+=1){const c=v/4*a,f=s[c+l],_=s[c+o+0],p=s[c+o+1],z=s[c+o+2];n+=f,d+=f*_,x+=f*p,g+=f*z}return{mass:n,momentum:[d,x,g]}}export{Q as a,j as c,$ as s,N as u};
