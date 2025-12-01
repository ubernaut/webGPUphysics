import { initWebGPU, mpm } from "../src/index.js";
import { OrbitCamera } from "./shared/orbitControls.js";
import { createSphereGeometry } from "./shared/geometries.js";
import { FluidRenderer } from "./shared/fluidRenderer.js";

const canvas = document.getElementById("canvas");
const toggleBtn = document.getElementById("toggleBtn");
const statusEl = document.getElementById("status");
const particleCountEl = document.getElementById("particleCount");
const fpsEl = document.getElementById("fps");
const massEl = document.getElementById("mass");
const momentumEl = document.getElementById("momentum");
const errorEl = document.getElementById("error");

let device, context, format;
let depthTexture, depthView;
let renderer; // Particle renderer
let fluidRenderer; // Fluid renderer
let sphereRenderer; // For interaction sphere
let domain;
let buffers; // Keep track of current buffers
let running = true;
let lastTime = performance.now();
let fps = 0;
let lastDiag = 0;
let animationFrameId;
let currentParticleCount = 0;
let initializing = false;
let initialOrientation = null;

const params = {
  renderMode: 'Fluid', // 'Particles' or 'Fluid'
  particleCount: 30000,
  gridSizeX: 64,
  gridSizeY: 64,
  gridSizeZ: 64,
  spacing: 0.65,
  jitter: 0.5,
  temperature: 273.0,
  dt: 0.1,
  stiffness: 2.5,
  restDensity: 4.0,
  dynamicViscosity: 0.08,
  iterations: 1,
  fixedPointScale: 1e5,
  visualRadius: 0.2, // For particles
  fluidRadius: 1,   // For fluid (larger for overlap)
  
  // Interaction
  interactionRadius: 9.0,
  interactionX: 32,
  interactionY: 0,
  interactionZ: 32,
  interactionActive: true,
};

toggleBtn.addEventListener("click", () => {
  running = !running;
  toggleBtn.textContent = running ? "Pause" : "Resume";
  statusEl.textContent = running ? "running" : "paused";
});

function setError(err) {
  console.error(err);
  errorEl.textContent = err?.stack || String(err);
  statusEl.textContent = "error";
  running = false;
  toggleBtn.disabled = true;
}

function resize() {
  const devicePixelRatio = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.floor(canvas.clientWidth * devicePixelRatio));
  const height = Math.max(1, Math.floor(canvas.clientHeight * devicePixelRatio));
  canvas.width = width;
  canvas.height = height;
  context.configure({
    device,
    format,
    alphaMode: "opaque"
  });
  if (depthTexture) depthTexture.destroy();
  depthTexture = device.createTexture({
    size: [width, height],
    format: "depth24plus",
    usage: GPUTextureUsage.RENDER_ATTACHMENT
  });
  depthView = depthTexture.createView();
  if (renderer) renderer.resize(width, height);
  if (fluidRenderer) fluidRenderer.resize(width, height);
}

class MpmRenderer {
  constructor(device) {
    this.device = device;
    this.canvasAspect = 1;
    const { vertices, indices } = createSphereGeometry(1, 8, 6);
    this.vertexBuffer = device.createBuffer({
      size: vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(this.vertexBuffer, 0, vertices);
    this.indexBuffer = device.createBuffer({
      size: indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(this.indexBuffer, 0, indices);
    this.indexCount = indices.length;

    this.uniformBuffer = device.createBuffer({
      size: 256,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    const shaderCode = /* wgsl */ `
      struct Uniforms {
        viewProj: mat4x4<f32>,
        radius: f32,
        pad0: vec3<f32>,
      };
      struct PosVel {
        position: vec3<f32>,
        velocity: vec3<f32>,
        temperature: f32,
      };
      @group(0) @binding(0) var<uniform> uniforms: Uniforms;
      @group(0) @binding(1) var<storage, read> posvel: array<PosVel>;

      struct VSOut {
        @builtin(position) position: vec4<f32>,
        @location(0) normal: vec3<f32>,
        @location(1) color: vec3<f32>,
      };

      @vertex
      fn vs_main(
        @location(0) localPos: vec3<f32>,
        @location(1) normal: vec3<f32>,
        @builtin(instance_index) instanceId: u32
      ) -> VSOut {
        let p = posvel[instanceId].position;
        let t = posvel[instanceId].temperature;
        let worldPos = p + localPos * uniforms.radius;
        
        // Temperature mapping: 273K (Blue) -> 373K (Red)
        let tempT = clamp((t - 273.0) / 100.0, 0.0, 1.0);
        let colorCold = vec3<f32>(0.2, 0.7, 0.95);
        let colorHot = vec3<f32>(0.95, 0.2, 0.2);
        let color = mix(colorCold, colorHot, tempT);

        var out: VSOut;
        out.position = uniforms.viewProj * vec4<f32>(worldPos, 1.0);
        out.normal = normal;
        out.color = color;
        return out;
      }

      @fragment
      fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
        let lightDir = normalize(vec3<f32>(0.3, 1.0, 0.2));
        let ambient = 0.25;
        let diffuse = max(dot(in.normal, lightDir), 0.0);
        let lighting = ambient + diffuse * 1.2;
        return vec4<f32>(in.color * lighting, 1.0);
      }
    `;

    const shaderModule = device.createShaderModule({ code: shaderCode });
    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } }
      ]
    });

    this.pipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      vertex: {
        module: shaderModule,
        entryPoint: "vs_main",
        buffers: [{
          arrayStride: 24, // Stride for vertex attributes (geometry), not instance buffer
          attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x3" },
            { shaderLocation: 1, offset: 12, format: "float32x3" }
          ]
        }]
      },
      fragment: {
        module: shaderModule,
        entryPoint: "fs_main",
        targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }]
      },
      primitive: { topology: "triangle-list", cullMode: "back" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" }
    });

    this.bindGroup = null;
  }

  resize(width, height) {
    this.canvasAspect = width / height;
  }

  updateBindGroup(posVelBuffer) {
    this.bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: posVelBuffer } }
      ]
    });
  }

  updateUniforms(viewProj, radius) {
    const data = new Float32Array(20);
    data.set(viewProj, 0);
    data[16] = radius;
    this.device.queue.writeBuffer(this.uniformBuffer, 0, data);
  }

  record(pass, particleCount) {
    if (!this.bindGroup) return;
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.setVertexBuffer(0, this.vertexBuffer);
    pass.setIndexBuffer(this.indexBuffer, "uint16");
    pass.drawIndexed(this.indexCount, particleCount);
  }
}

// Simple Renderer for Interaction Sphere (Red wireframe or solid)
class SphereRenderer {
    constructor(device) {
        this.device = device;
        const { vertices, indices } = createSphereGeometry(1, 16, 12);
        this.vertexBuffer = device.createBuffer({ size: vertices.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(this.vertexBuffer, 0, vertices);
        this.indexBuffer = device.createBuffer({ size: indices.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(this.indexBuffer, 0, indices);
        this.indexCount = indices.length;
        
        this.uniformBuffer = device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        
        const shaderCode = /* wgsl */ `
            struct Uniforms {
                viewProj: mat4x4<f32>,
                position: vec3<f32>,
                radius: f32,
            };
            @group(0) @binding(0) var<uniform> uniforms: Uniforms;
            
            struct VSOut {
                @builtin(position) position: vec4<f32>,
                @location(0) normal: vec3<f32>,
            };
            
            @vertex
            fn vs(@location(0) pos: vec3<f32>, @location(1) norm: vec3<f32>) -> VSOut {
                var out: VSOut;
                let worldPos = uniforms.position + pos * uniforms.radius;
                out.position = uniforms.viewProj * vec4<f32>(worldPos, 1.0);
                out.normal = norm;
                return out;
            }
            
            @fragment
            fn fs(in: VSOut) -> @location(0) vec4<f32> {
                let light = vec3<f32>(0.5, 1.0, 0.2);
                let diff = max(dot(in.normal, normalize(light)), 0.2);
                return vec4<f32>(1.0, 0.4, 0.4, 1.0) * diff;
            }
        `;
        
        const module = device.createShaderModule({ code: shaderCode });
        this.pipeline = device.createRenderPipeline({
            layout: 'auto',
            vertex: { module, entryPoint: 'vs', buffers: [{ arrayStride: 24, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }, { shaderLocation: 1, offset: 12, format: 'float32x3' }] }] },
            fragment: { module, entryPoint: 'fs', targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }] },
            primitive: { topology: 'triangle-list', cullMode: 'back' },
            depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' }
        });
        
        this.bindGroup = device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [{ binding: 0, resource: { buffer: this.uniformBuffer } }]
        });
    }
    
    update(viewProj, pos, radius) {
        const data = new Float32Array(20);
        data.set(viewProj, 0);
        data.set(pos, 16); // vec3
        data[19] = radius; // float (packed after vec3? 16+12=28? No. 16+3*4 = 28. Float aligned 4. But vec3 align 16? In uniform, yes. So pos is 16..28. radius at 28?)
        // Standard uniform layout:
        // mat4 (0..64)
        // vec3 (64..76)
        // f32 (76..80)
        // My code says `data.set(pos, 16)`. That's offset 16 FLOATs (64 bytes). Correct.
        // `data[19]`. That's offset 19 floats (76 bytes). Correct.
        this.device.queue.writeBuffer(this.uniformBuffer, 0, data);
    }
    
    record(pass) {
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        pass.setVertexBuffer(0, this.vertexBuffer);
        pass.setIndexBuffer(this.indexBuffer, 'uint16');
        pass.drawIndexed(this.indexCount);
    }
}

async function initSimulation() {
  if (initializing) return;
  initializing = true;
  statusEl.textContent = "initializing...";

  try {
    const particleCount = params.particleCount;
    const gridSize = { x: params.gridSizeX, y: params.gridSizeY, z: params.gridSizeZ };
    const blockOptions = { 
        start: [2, 2, 2], 
        gridSize, 
        jitter: params.jitter, 
        spacing: params.spacing,
        temperature: params.temperature,
        restDensity: params.restDensity
    };

    // Sub-stepping for stability
    // Fix physics step to a stable value (e.g. 5ms) and iterate to match requested dt
    const physics_dt = 0.005;
    const iterations = Math.ceil(params.dt / physics_dt);

    const constants = {
        stiffness: params.stiffness,
        restDensity: params.restDensity,
        dynamicViscosity: params.dynamicViscosity,
        dt: physics_dt,
        fixedPointScale: params.fixedPointScale
    };

    // Create buffers
    // Safety size for PosVel struct stride 32
    const posVelBuffer = device.createBuffer({
        size: particleCount * 32, 
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    });
    
    const interactionBuffer = device.createBuffer({
        size: 32,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    const setup = mpm.setupMpmDomain(device, {
        particleCount,
        gridSize,
        iterations: iterations, // Use calculated sub-steps
        posVelBuffer,
        interactionBuffer,
        constants
    });

    // Initialize particles
    const data = mpm.createBlockParticleData({ count: particleCount, gridSize, ...blockOptions });
    mpm.uploadParticleData(device, setup.buffers.particleBuffer, data);

    domain = setup.domain;
    buffers = setup.buffers;
    currentParticleCount = particleCount;

    if (renderer) renderer.updateBindGroup(posVelBuffer);
    if (fluidRenderer) fluidRenderer.updateBindGroup(posVelBuffer);
    
    particleCountEl.textContent = particleCount.toString();
    statusEl.textContent = "running";
  } catch(e) {
    console.warn("Init error:", e);
    setError(e);
  } finally {
    initializing = false;
  }
}

async function setup() {
  try {
    const gpu = await initWebGPU();
    device = gpu.device;
    context = canvas.getContext("webgpu");
    format = navigator.gpu.getPreferredCanvasFormat();
    
    renderer = new MpmRenderer(device);
    fluidRenderer = new FluidRenderer(device, format);
    sphereRenderer = new SphereRenderer(device);
    
    resize();
    window.addEventListener("resize", resize);

    // Center camera on the default grid (64x64x64)
    // Center: 32, 32, 32
    // Radius: ~1.5 * max_dim to fit in view
    const camera = new OrbitCamera(canvas, { target: [32, 32, 32], radius: 120 });

    // Initial Sim
    await initSimulation();

    // GUI Setup
    const gui = new window.lil.GUI({ title: "MLS-MPM Controls" });
    
    // Collapse on mobile
    const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) || window.innerWidth < 768;
    if (isMobile) {
        gui.close();
    }

    gui.add(params, "renderMode", ['Particles', 'Fluid']).name("Render Mode");
    
    const viewFolder = gui.addFolder("Visuals");
    viewFolder.add(params, "visualRadius", 0.05, 1.0, 0.05).name("Particle Radius");
    viewFolder.add(params, "fluidRadius", 0.05, 1.0, 0.05).name("Fluid Radius");

    const simFolder = gui.addFolder("Simulation");
    simFolder.add(params, "particleCount", 100, 800000, 1000).name("Particle Count").onFinishChange(initSimulation);
    
    // Box Size
    simFolder.add(params, "gridSizeX", 16, 256, 16).name("Box X").onFinishChange(initSimulation);
    simFolder.add(params, "gridSizeY", 16, 256, 16).name("Box Y").onFinishChange(initSimulation);
    simFolder.add(params, "gridSizeZ", 16, 256, 16).name("Box Z").onFinishChange(initSimulation);
    
    simFolder.add(params, "spacing", 0.1, 2.0, 0.05).name("Spacing").onFinishChange(initSimulation);
    simFolder.add(params, "jitter", 0.0, 1.0, 0.1).name("Jitter").onFinishChange(initSimulation);
    simFolder.add(params, "temperature", 0, 500, 1).name("Temperature (K)").onFinishChange(initSimulation);
    
    const physFolder = gui.addFolder("Physics Constants");
    physFolder.add(params, "dt", 0.001, 0.2, 0.001).name("Time Step (dt)").onChange(() => initSimulation());
    physFolder.add(params, "stiffness", 0.1, 50.0, 0.1).onFinishChange(initSimulation);
    physFolder.add(params, "restDensity", 0.1, 10.0, 0.1).onFinishChange(initSimulation);
    physFolder.add(params, "dynamicViscosity", 0.0, 5.0, 0.01).onFinishChange(initSimulation);
    
    const interactFolder = gui.addFolder("Interaction Sphere");
    interactFolder.add(params, "interactionActive").name("Active");
    interactFolder.add(params, "interactionRadius", 0.1, 20.0).name("Radius");
    interactFolder.add(params, "interactionX", 0, 100).name("X");
    interactFolder.add(params, "interactionY", 0, 100).name("Y");
    interactFolder.add(params, "interactionZ", 0, 100).name("Z");
    
    gui.add({ reset: initSimulation }, "reset").name("Reset Simulation");
    gui.add({ calibrate: () => initialOrientation = null }, "calibrate").name("Calibrate Sensors");

    statusEl.textContent = "running";

    // Mouse Interaction
    let dragging = false;
    let planeY = 0;
    
    canvas.addEventListener('pointerdown', e => {
        if(e.button === 0 && e.shiftKey) { // Shift+Click to drag sphere
            dragging = true;
            planeY = params.interactionY;
            e.stopImmediatePropagation(); // Prevent orbit
        }
    });
    window.addEventListener('pointerup', () => dragging = false);
    
    // Device Orientation
    window.addEventListener("deviceorientation", (event) => {
        if (!buffers || !buffers.simUniformBuffer) return;
        
        if (!initialOrientation) {
            initialOrientation = { beta: event.beta, gamma: event.gamma };
        }

        const g = 0.3;
        // Calculate relative tilt
        const beta = (event.beta - initialOrientation.beta) * (Math.PI / 180);
        const gamma = (event.gamma - initialOrientation.gamma) * (Math.PI / 180);
        
        // Logic:
        // Initial state (relative 0,0): Gravity is straight down (-Y).
        // Tilting device (changing beta/gamma) rotates the gravity vector.
        
        // Rotate vector (0, -1, 0) by Euler angles?
        // Simplified:
        // Gamma (left/right tilt) affects X.
        // Beta (front/back tilt) affects Z.
        
        let gx = Math.sin(gamma) * g;
        let gz = Math.sin(beta) * g;
        // Ensure y component maintains magnitude roughly
        let gy = -Math.sqrt(Math.max(0, g*g - gx*gx - gz*gz));
        
        const data = new Float32Array(4);
        data.set([gx, gy, gz], 0); 
        device.queue.writeBuffer(buffers.simUniformBuffer, 0, data);
    }, true);

    canvas.addEventListener('pointermove', e => {
        if (dragging && params.interactionActive) {
            // Raycast logic
            const rect = canvas.getBoundingClientRect();
            const x = (e.clientX - rect.left) / rect.width * 2 - 1;
            const y = -(e.clientY - rect.top) / rect.height * 2 + 1;
            
            // Get camera ray
            const mats = camera.getMatrices(rect.width/rect.height);
            const invView = mats.invView;
            const invProj = mats.invProj;
            
            const clip = [x, y, 0.5, 1.0];
            // View space
            let viewX = invProj[0]*x + invProj[4]*y + invProj[8]*0.5 + invProj[12];
            let viewY = invProj[1]*x + invProj[5]*y + invProj[9]*0.5 + invProj[13];
            let viewZ = invProj[2]*x + invProj[6]*y + invProj[10]*0.5 + invProj[14];
            let viewW = invProj[3]*x + invProj[7]*y + invProj[11]*0.5 + invProj[15];
            viewX/=viewW; viewY/=viewW; viewZ/=viewW;
            
            // World space
            let worldX = invView[0]*viewX + invView[4]*viewY + invView[8]*viewZ; // No translation for dir?
            let worldY = invView[1]*viewX + invView[5]*viewY + invView[9]*viewZ;
            let worldZ = invView[2]*viewX + invView[6]*viewY + invView[10]*viewZ;
            // Camera pos
            const eye = mats.eye;
            const dir = [worldX, worldY, worldZ]; // Vector from eye? 
            // Correct way: unproject (x,y,0) and (x,y,1) or just use direction from eye.
            // Simplified: Unproject near plane point
            const worldPos = [
                invView[0]*viewX + invView[4]*viewY + invView[8]*viewZ + invView[12],
                invView[1]*viewX + invView[5]*viewY + invView[9]*viewZ + invView[13],
                invView[2]*viewX + invView[6]*viewY + invView[10]*viewZ + invView[14]
            ];
            
            const rayDir = [worldPos[0]-eye[0], worldPos[1]-eye[1], worldPos[2]-eye[2]];
            // Normalize
            const len = Math.hypot(rayDir[0], rayDir[1], rayDir[2]);
            rayDir[0]/=len; rayDir[1]/=len; rayDir[2]/=len;
            
            // Intersect plane Y = planeY
            // P = O + t*D. P.y = planeY.
            // planeY = O.y + t*D.y => t = (planeY - O.y) / D.y
            if (Math.abs(rayDir[1]) > 0.001) {
                const t = (planeY - eye[1]) / rayDir[1];
                if (t > 0) {
                    params.interactionX = eye[0] + t * rayDir[0];
                    params.interactionZ = eye[2] + t * rayDir[2];
                    gui.controllers.forEach(c => c.updateDisplay());
                }
            }
        }
    });

    async function frame(now) {
      const dtSeconds = (now - lastTime) / 1000;
      lastTime = now;
      const smooth = 0.9;
      if (dtSeconds > 0) {
        const instFps = 1 / dtSeconds;
        fps = fps * smooth + instFps * (1 - smooth);
        fpsEl.textContent = fps.toFixed(1);
      }

      if (initializing) {
        animationFrameId = requestAnimationFrame(frame);
        return;
      }

      // Update Interaction Buffer
      if (buffers && buffers.interactionBuffer) {
          const iData = new Float32Array(8);
          if (params.interactionActive) {
              iData.set([params.interactionX, params.interactionY, params.interactionZ], 0);
              iData[3] = params.interactionRadius;
          } else {
              iData[3] = -1.0;
          }
          device.queue.writeBuffer(buffers.interactionBuffer, 0, iData);
      }

      const encoder = device.createCommandEncoder({ label: "mpm-visual-frame" });
      if (running && domain) {
        domain.step(encoder, params.dt);
      }

      // Render
      if (params.renderMode === 'Fluid') {
        const matrices = camera.getMatrices(canvas.width / canvas.height);
        fluidRenderer.updateUniforms(matrices, params.fluidRadius); 
        
        const textureView = context.getCurrentTexture().createView();
        const clearPass = encoder.beginRenderPass({
            colorAttachments: [{
                view: textureView,
                clearValue: { r: 0.05, g: 0.08, b: 0.14, a: 1 },
                loadOp: 'clear',
                storeOp: 'store'
            }],
            depthStencilAttachment: {
                view: depthView,
                depthLoadOp: 'clear',
                depthStoreOp: 'store',
                depthClearValue: 1.0
            }
        });
        // Draw interaction sphere in fluid mode? Maybe as solid?
        // Let's draw it before fluid or after?
        // If we draw it before, fluid covers it.
        // If after, it's on top.
        // Let's render sphere inside fluid renderer? No, separate.
        // Render sphere to screen.
        if (params.interactionActive) {
            sphereRenderer.update(matrices.viewProj, [params.interactionX, params.interactionY, params.interactionZ], params.interactionRadius);
            sphereRenderer.record(clearPass);
        }
        clearPass.end();
        
        fluidRenderer.render(encoder, textureView, depthView, currentParticleCount);

      } else {
        const viewProj = camera.getViewProj(canvas.width / canvas.height);
        renderer.updateUniforms(viewProj, params.visualRadius);

        const textureView = context.getCurrentTexture().createView();
        const pass = encoder.beginRenderPass({
          colorAttachments: [{
            view: textureView,
            loadOp: "clear",
            storeOp: "store",
            clearValue: { r: 0.05, g: 0.08, b: 0.14, a: 1 }
          }],
          depthStencilAttachment: {
            view: depthView,
            depthLoadOp: "clear",
            depthStoreOp: "store",
            depthClearValue: 1.0
          }
        });
        
        if (params.interactionActive) {
            sphereRenderer.update(viewProj, [params.interactionX, params.interactionY, params.interactionZ], params.interactionRadius);
            sphereRenderer.record(pass);
        }
        
        renderer.record(pass, currentParticleCount);
        pass.end();
      }

      device.queue.submit([encoder.finish()]);

      // Occasional diagnostics
      if (now - lastDiag > 500 && running && buffers) {
        lastDiag = now;
        mpm.computeMassMomentum(device, buffers.particleBuffer, currentParticleCount)
          .then(({ mass, momentum }) => {
            massEl.textContent = mass.toFixed(3);
            momentumEl.textContent = momentum.map((v) => v.toFixed(3)).join(", ");
          })
          .catch(err => console.warn(err));
      }

      animationFrameId = requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  } catch (err) {
    setError(err);
  }
}

setup();
