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
let domain;
let buffers; // Keep track of current buffers
let running = true;
let lastTime = performance.now();
let fps = 0;
let lastDiag = 0;
let animationFrameId;
let currentParticleCount = 0;
let initializing = false;

const params = {
  renderMode: 'Particles', // 'Particles' or 'Fluid'
  particleCount: 4000,
  gridSizeX: 32,
  gridSizeY: 32,
  gridSizeZ: 32,
  spacing: 0.65,
  jitter: 0.5,
  dt: 0.05,
  stiffness: 2.5,
  restDensity: 4.0,
  dynamicViscosity: 0.08,
  iterations: 1,
  fixedPointScale: 1e7,
  visualRadius: 0.25, // For particles
  fluidRadius: 0.4,   // For fluid (larger for overlap)
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
      };
      @group(0) @binding(0) var<uniform> uniforms: Uniforms;
      @group(0) @binding(1) var<storage, read> posvel: array<PosVel>;

      struct VSOut {
        @builtin(position) position: vec4<f32>,
        @location(0) normal: vec3<f32>,
      };

      @vertex
      fn vs_main(
        @location(0) localPos: vec3<f32>,
        @location(1) normal: vec3<f32>,
        @builtin(instance_index) instanceId: u32
      ) -> VSOut {
        let p = posvel[instanceId].position;
        let worldPos = p + localPos * uniforms.radius;
        var out: VSOut;
        out.position = uniforms.viewProj * vec4<f32>(worldPos, 1.0);
        out.normal = normal;
        return out;
      }

      @fragment
      fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
        let lightDir = normalize(vec3<f32>(0.3, 1.0, 0.2));
        let ambient = 0.25;
        let diffuse = max(dot(in.normal, lightDir), 0.0);
        let color = vec3<f32>(0.2, 0.7, 0.95);
        let lighting = ambient + diffuse * 1.2;
        return vec4<f32>(color * lighting, 1.0);
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
          arrayStride: 24,
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
        spacing: params.spacing 
    };

    const constants = {
        stiffness: params.stiffness,
        restDensity: params.restDensity,
        dynamicViscosity: params.dynamicViscosity,
        dt: params.dt,
        fixedPointScale: params.fixedPointScale
    };

    // Create buffers
    // Safety size for PosVel struct stride 32
    const posVelBuffer = device.createBuffer({
        size: particleCount * 32, 
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    });

    const setup = mpm.setupMpmDomain(device, {
        particleCount,
        gridSize,
        iterations: params.iterations,
        posVelBuffer,
        constants
    });

    // Initialize particles
    const data = mpm.createBlockParticleData({ count: particleCount, gridSize, ...blockOptions });
    mpm.uploadParticleData(device, setup.buffers.particleBuffer, data);

    // Atomic update of state
    if (domain) {
        // cleanup old if needed? WebGPU objects are GC'd or we should destroy them.
        // For now rely on GC.
    }
    
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
    
    resize();
    window.addEventListener("resize", resize);

    const camera = new OrbitCamera(canvas, { target: [16, 16, 16], radius: 40 });

    // Initial Sim
    await initSimulation();

    // GUI Setup
    const gui = new window.lil.GUI({ title: "MLS-MPM Controls" });
    
    gui.add(params, "renderMode", ['Particles', 'Fluid']).name("Render Mode");
    
    const viewFolder = gui.addFolder("Visuals");
    viewFolder.add(params, "visualRadius", 0.05, 1.0, 0.05).name("Particle Radius");
    viewFolder.add(params, "fluidRadius", 0.05, 1.0, 0.05).name("Fluid Radius");

    const simFolder = gui.addFolder("Simulation");
    simFolder.add(params, "particleCount", 100, 50000, 100).name("Particle Count").onFinishChange(initSimulation);
    simFolder.add(params, "gridSizeX", 16, 128, 16).name("Grid X").onFinishChange(initSimulation);
    simFolder.add(params, "spacing", 0.1, 2.0, 0.05).name("Spacing").onFinishChange(initSimulation);
    simFolder.add(params, "jitter", 0.0, 1.0, 0.1).name("Jitter").onFinishChange(initSimulation);
    
    const physFolder = gui.addFolder("Physics Constants");
    physFolder.add(params, "dt", 0.001, 0.2, 0.001).name("Time Step (dt)").onChange(v => {
      initSimulation();
    });
    physFolder.add(params, "stiffness", 0.1, 50.0, 0.1).onFinishChange(initSimulation);
    physFolder.add(params, "restDensity", 0.1, 10.0, 0.1).onFinishChange(initSimulation);
    physFolder.add(params, "dynamicViscosity", 0.0, 5.0, 0.01).onFinishChange(initSimulation);
    
    gui.add({ reset: initSimulation }, "reset").name("Reset Simulation");

    statusEl.textContent = "running";

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
        // Skip frame if initializing to avoid buffer mismatch
        animationFrameId = requestAnimationFrame(frame);
        return;
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
            }]
        });
        clearPass.end();
        
        fluidRenderer.render(encoder, textureView, currentParticleCount);

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
        renderer.record(pass, currentParticleCount);
        pass.end();
      }

      device.queue.submit([encoder.finish()]);

      // Occasional diagnostics
      if (now - lastDiag > 500 && running && buffers) {
        lastDiag = now;
        // Use currentParticleCount here too
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
