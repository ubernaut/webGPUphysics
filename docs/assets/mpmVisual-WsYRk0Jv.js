import{i as de}from"./device-CAsdAK37.js";import{a as fe,s as le,c as pe,u as he}from"./diagnostics-BofLGZHx.js";import{O as me,c as ne}from"./orbitControls-D071pLfL.js";const H=`
struct VertexOutput {
  @builtin(position) position : vec4f,
  @location(0) uv : vec2f,
  @location(1) iuv : vec2f,
}

override screenWidth: f32;
override screenHeight: f32;

@vertex
fn vs(@builtin(vertex_index) vertex_index : u32) -> VertexOutput {
    var out: VertexOutput;
    var pos = array(
        vec2( 1.0,  1.0), vec2( 1.0, -1.0), vec2(-1.0, -1.0),
        vec2( 1.0,  1.0), vec2(-1.0, -1.0), vec2(-1.0,  1.0),
    );
    var uv = array(
        vec2(1.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0),
        vec2(1.0, 0.0), vec2(0.0, 1.0), vec2(0.0, 0.0),
    );
    out.position = vec4(pos[vertex_index], 0.0, 1.0);
    out.uv = uv[vertex_index];
    out.iuv = out.uv * vec2f(screenWidth, screenHeight);
    return out;
}
`,Z=`
struct VertexOutput {
    @builtin(position) position: vec4f, 
    @location(0) uv: vec2f, 
    @location(1) view_position: vec3f, 
}
struct FragmentInput {
    @location(0) uv: vec2f, 
    @location(1) view_position: vec3f, 
}
struct FragmentOutput {
    @location(0) frag_color: vec4f, 
    @builtin(frag_depth) frag_depth: f32, 
}
struct RenderUniforms {
    texel_size: vec2f, 
    sphere_size: f32, 
    inv_projection_matrix: mat4x4f, 
    projection_matrix: mat4x4f, 
    view_matrix: mat4x4f, 
    inv_view_matrix: mat4x4f, 
}
struct PosVel {
    position: vec3f, 
    v: vec3f, 
}
@group(0) @binding(0) var<storage> particles: array<PosVel>;
@group(0) @binding(1) var<uniform> uniforms: RenderUniforms;

@vertex
fn vs(@builtin(vertex_index) vertex_index: u32, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    var corner_positions = array(
        vec2(0.5, 0.5), vec2(0.5, -0.5), vec2(-0.5, -0.5),
        vec2(0.5, 0.5), vec2(-0.5, -0.5), vec2(-0.5, 0.5),
    );
    let corner = vec3(corner_positions[vertex_index] * uniforms.sphere_size, 0.0);
    let uv = corner_positions[vertex_index] + 0.5;
    let real_position = particles[instance_index].position;
    let view_position = (uniforms.view_matrix * vec4f(real_position, 1.0)).xyz;
    let out_position = uniforms.projection_matrix * vec4f(view_position + corner, 1.0);
    return VertexOutput(out_position, uv, view_position);
}

@fragment
fn fs(input: FragmentInput) -> FragmentOutput {
    var out: FragmentOutput;
    var normalxy: vec2f = input.uv * 2.0 - 1.0;
    var r2: f32 = dot(normalxy, normalxy);
    if (r2 > 1.0) { discard; }
    var normalz = sqrt(1.0 - r2);
    var normal = vec3(normalxy, normalz);
    var radius = uniforms.sphere_size / 2.0;
    var real_view_pos: vec4f = vec4f(input.view_position + normal * radius, 1.0);
    var clip_space_pos: vec4f = uniforms.projection_matrix * real_view_pos;
    out.frag_depth = clip_space_pos.z / clip_space_pos.w;
    out.frag_color = vec4(real_view_pos.z, 0.0, 0.0, 1.0);
    return out;
}
`,K=`
@group(0) @binding(1) var texture: texture_2d<f32>;
@group(0) @binding(2) var<uniform> uniforms: FilterUniforms;

struct FragmentInput {
    @location(0) uv: vec2f,  
    @location(1) iuv: vec2f
}
override depth_threshold: f32;
override projected_particle_constant: f32;
override max_filter_size: f32;
struct FilterUniforms {
    blur_dir: vec2f,
}

@fragment
fn fs(input: FragmentInput) -> @location(0) vec4f {
    var depth: f32 = abs(textureLoad(texture, vec2u(input.iuv), 0).r);
    if (depth >= 1e4 || depth <= 0.0) {
        return vec4f(vec3f(depth), 1.0);
    }
    var filter_size: i32 = min(i32(max_filter_size), i32(ceil(projected_particle_constant / depth)));
    var sigma: f32 = f32(filter_size) / 3.0; 
    var two_sigma: f32 = 2.0 * sigma * sigma;
    var sigma_depth: f32 = depth_threshold / 3.0;
    var two_sigma_depth: f32 = 2.0 * sigma_depth * sigma_depth;

    var sum: f32 = 0.0;
    var wsum: f32 = 0.0;
    for (var x: i32 = -filter_size; x <= filter_size; x++) {
        var coords: vec2f = vec2f(f32(x));
        var sampled_depth: f32 = abs(textureLoad(texture, vec2u(input.iuv + coords * uniforms.blur_dir), 0).r);
        var rr: f32 = dot(coords, coords);
        var w: f32 = exp(-rr / two_sigma);
        var r_depth: f32 = sampled_depth - depth;
        var wd: f32 = exp(-r_depth * r_depth / two_sigma_depth);
        sum += sampled_depth * w * wd;
        wsum += w * wd;
    }
    if (wsum > 0.0) { sum /= wsum; }
    return vec4f(sum, 0.0, 0.0, 1.0);
}
`,J=`
struct RenderUniforms {
    texel_size: vec2f, sphere_size: f32,
    inv_projection_matrix: mat4x4f, projection_matrix: mat4x4f,
    view_matrix: mat4x4f, inv_view_matrix: mat4x4f,
}
struct VertexOutput {
    @builtin(position) position: vec4f, 
    @location(0) uv: vec2f, 
}
struct FragmentInput { @location(0) uv: vec2f, }
struct PosVel { position: vec3f, v: vec3f, }
@group(0) @binding(0) var<storage> particles: array<PosVel>;
@group(0) @binding(1) var<uniform> uniforms: RenderUniforms;

@vertex
fn vs(@builtin(vertex_index) vertex_index: u32, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    var corner_positions = array(
        vec2(0.5, 0.5), vec2(0.5, -0.5), vec2(-0.5, -0.5),
        vec2(0.5, 0.5), vec2(-0.5, -0.5), vec2(-0.5, 0.5),
    );
    let corner = vec3(corner_positions[vertex_index] * uniforms.sphere_size, 0.0);
    let uv = corner_positions[vertex_index] + 0.5;
    let real_position = particles[instance_index].position;
    let view_position = (uniforms.view_matrix * vec4f(real_position, 1.0)).xyz;
    let out_position = uniforms.projection_matrix * vec4f(view_position + corner, 1.0);
    return VertexOutput(out_position, uv);
}

@fragment
fn fs(input: FragmentInput) -> @location(0) vec4f {
    var normalxy: vec2f = input.uv * 2.0 - 1.0;
    var r2: f32 = dot(normalxy, normalxy);
    if (r2 > 1.0) { discard; }
    var thickness: f32 = sqrt(1.0 - r2);
    let particle_alpha = 0.05;
    return vec4f(vec3f(particle_alpha * thickness), 1.0);
}
`,Q=`
@group(0) @binding(1) var texture: texture_2d<f32>;
@group(0) @binding(2) var<uniform> uniforms: FilterUniforms;
struct FragmentInput { @location(0) uv: vec2f, @location(1) iuv: vec2f }
struct FilterUniforms { blur_dir: vec2f, }

@fragment
fn fs(input: FragmentInput) -> @location(0) vec4f {
    var thickness: f32 = textureLoad(texture, vec2u(input.iuv), 0).r;
    if (thickness == 0.0) { return vec4f(0.0, 0.0, 0.0, 1.0); }
    var filter_size: i32 = 30; 
    var sigma: f32 = f32(filter_size) / 3.0;
    var two_sigma: f32 = 2.0 * sigma * sigma;
    var sum = 0.0; var wsum = 0.0;
    for (var x: i32 = -filter_size; x <= filter_size; x++) {
        var coords: vec2f = vec2f(f32(x));
        var sampled_thickness: f32 = textureLoad(texture, vec2u(input.iuv + uniforms.blur_dir * coords), 0).r;
        var w: f32 = exp(-coords.x * coords.x / two_sigma);
        sum += sampled_thickness * w;
        wsum += w;
    }
    sum /= wsum;
    return vec4f(sum, 0.0, 0.0, 1.0);
}
`,$=`
@group(0) @binding(1) var texture: texture_2d<f32>;
@group(0) @binding(2) var<uniform> uniforms: RenderUniforms;
@group(0) @binding(3) var thickness_texture: texture_2d<f32>;

struct RenderUniforms {
    texel_size: vec2f, sphere_size: f32,
    inv_projection_matrix: mat4x4f, projection_matrix: mat4x4f,
    view_matrix: mat4x4f, inv_view_matrix: mat4x4f,
}
struct FragmentInput { @location(0) uv: vec2f, @location(1) iuv: vec2f, }

struct FragmentOutput {
    @location(0) color: vec4f,
    @builtin(frag_depth) depth: f32,
}

fn computeViewPosFromUVDepth(tex_coord: vec2f, depth: f32) -> vec3f {
    var ndc: vec4f = vec4f(tex_coord.x * 2.0 - 1.0, 1.0 - 2.0 * tex_coord.y, 0.0, 1.0);
    ndc.z = -uniforms.projection_matrix[2].z + uniforms.projection_matrix[3].z / depth;
    ndc.w = 1.0;
    var eye_pos: vec4f = uniforms.inv_projection_matrix * ndc;
    return eye_pos.xyz / eye_pos.w;
}

fn getViewPosFromTexCoord(tex_coord: vec2f, iuv: vec2f) -> vec3f {
    var depth: f32 = abs(textureLoad(texture, vec2u(iuv), 0).x);
    return computeViewPosFromUVDepth(tex_coord, depth);
}

@fragment
fn fs(input: FragmentInput) -> FragmentOutput {
    var depth: f32 = abs(textureLoad(texture, vec2u(input.iuv), 0).r);
    // Make background lighter (greyish blue)
    let bgColor: vec3f = vec3f(0.7, 0.75, 0.8); 

    if (depth >= 1e4 || depth <= 0.0) { discard; }

    var viewPos: vec3f = computeViewPosFromUVDepth(input.uv, depth);
    
    // Finite difference for normals
    var ddx = getViewPosFromTexCoord(input.uv + vec2f(uniforms.texel_size.x, 0.), input.iuv + vec2f(1.0, 0.0)) - viewPos; 
    var ddy = getViewPosFromTexCoord(input.uv + vec2f(0., uniforms.texel_size.y), input.iuv + vec2f(0.0, 1.0)) - viewPos; 
    var ddx2 = viewPos - getViewPosFromTexCoord(input.uv + vec2f(-uniforms.texel_size.x, 0.), input.iuv + vec2f(-1.0, 0.0));
    var ddy2 = viewPos - getViewPosFromTexCoord(input.uv + vec2f(0., -uniforms.texel_size.y), input.iuv + vec2f(0.0, -1.0));
    if (abs(ddx.z) > abs(ddx2.z)) { ddx = ddx2; }
    if (abs(ddy.z) > abs(ddy2.z)) { ddy = ddy2; }

    var normal: vec3f = -normalize(cross(ddx, ddy)); 
    var rayDir = normalize(viewPos);
    // Light direction (top-down)
    var lightDir = normalize((uniforms.view_matrix * vec4f(0.2, 1.0, 0.2, 0.0)).xyz);
    
    var H = normalize(lightDir - rayDir);
    var specular = pow(max(0.0, dot(H, normal)), 250.0);
    
    var density = 0.5; // Reduced density for more transparency
    var thickness = textureLoad(thickness_texture, vec2u(input.iuv), 0).r;
    // Water color (Blue-ish)
    var diffuseColor = vec3f(0.1, 0.6, 0.9); 
    
    // Beer's law for attenuation
    var transmittance = exp(-density * thickness * (1.0 - diffuseColor)); 
    var refractionColor = bgColor * transmittance;

    let F0 = 0.02;
    var fresnel = clamp(F0 + (1.0 - F0) * pow(1.0 - dot(normal, -rayDir), 5.0), 0.0, 1.0);

    var reflectionDir = reflect(rayDir, normal);
    var reflectionDirWorld = (uniforms.inv_view_matrix * vec4f(reflectionDir, 0.0)).xyz;
    
    // Brighter Fake environment map
    // Sky gradient: Light blue at top, white/grey at horizon/bottom
    var skyTop = vec3f(0.6, 0.8, 1.0);
    var skyBottom = vec3f(0.9, 0.95, 1.0);
    var reflectionColor = mix(skyBottom, skyTop, clamp(reflectionDirWorld.y * 0.5 + 0.5, 0.0, 1.0));
    
    var finalColor = 1.0 * specular + mix(refractionColor, reflectionColor, fresnel);
    
    // Gamma correction
    var color = vec4f(pow(finalColor, vec3f(1.0/2.2)), 1.0);
    
    // Compute clip space depth for depth test
    var clipPos = uniforms.projection_matrix * vec4f(viewPos, 1.0);
    var fragDepth = clipPos.z / clipPos.w;

    return FragmentOutput(color, fragDepth);
}
`;class ve{constructor(e,i){this.device=e,this.canvasFormat=i,this.width=1,this.height=1,this.radius=.25;const r=e.createSampler({magFilter:"linear",minFilter:"linear"});this.sampler=r,this.renderUniformBuffer=e.createBuffer({size:272,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});const c=16;this.filterXUniformBuffer=e.createBuffer({size:c,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),this.filterYUniformBuffer=e.createBuffer({size:c,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(this.filterXUniformBuffer,0,new Float32Array([1,0])),e.queue.writeBuffer(this.filterYUniformBuffer,0,new Float32Array([0,1]));const s=e.createShaderModule({code:H});this.depthMapPipeline=e.createRenderPipeline({layout:"auto",vertex:{module:e.createShaderModule({code:Z}),entryPoint:"vs"},fragment:{module:e.createShaderModule({code:Z}),entryPoint:"fs",targets:[{format:"r32float"}]},primitive:{topology:"triangle-list"},depthStencil:{depthWriteEnabled:!0,depthCompare:"less",format:"depth32float"}}),this.thicknessMapPipeline=e.createRenderPipeline({layout:"auto",vertex:{module:e.createShaderModule({code:J}),entryPoint:"vs"},fragment:{module:e.createShaderModule({code:J}),entryPoint:"fs",targets:[{format:"r16float",blend:{color:{operation:"add",srcFactor:"one",dstFactor:"one"},alpha:{operation:"add",srcFactor:"one",dstFactor:"one"}}}]},primitive:{topology:"triangle-list"}}),this.depthFilterPipeline=e.createRenderPipeline({layout:"auto",vertex:{module:s,entryPoint:"vs",constants:{screenWidth:1,screenHeight:1}},fragment:{module:e.createShaderModule({code:K}),entryPoint:"fs",targets:[{format:"r32float"}],constants:{depth_threshold:.1,projected_particle_constant:100,max_filter_size:100}},primitive:{topology:"triangle-list"}}),this.thicknessFilterPipeline=e.createRenderPipeline({layout:"auto",vertex:{module:s,entryPoint:"vs",constants:{screenWidth:1,screenHeight:1}},fragment:{module:e.createShaderModule({code:Q}),entryPoint:"fs",targets:[{format:"r16float"}]},primitive:{topology:"triangle-list"}}),this.fluidPipeline=e.createRenderPipeline({layout:"auto",vertex:{module:s,entryPoint:"vs",constants:{screenWidth:1,screenHeight:1}},fragment:{module:e.createShaderModule({code:$}),entryPoint:"fs",targets:[{format:i,blend:{color:{srcFactor:"src-alpha",dstFactor:"one-minus-src-alpha"},alpha:{srcFactor:"one",dstFactor:"one"}}}]},primitive:{topology:"triangle-list"},depthStencil:{format:"depth24plus",depthWriteEnabled:!1,depthCompare:"always"}})}resize(e,i){this.width=e,this.height=i;const r=GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING;this.depthMapTexture=this.device.createTexture({size:[e,i],format:"r32float",usage:r}),this.tmpDepthMapTexture=this.device.createTexture({size:[e,i],format:"r32float",usage:r}),this.thicknessTexture=this.device.createTexture({size:[e,i],format:"r16float",usage:r}),this.tmpThicknessTexture=this.device.createTexture({size:[e,i],format:"r16float",usage:r}),this.depthTestTexture=this.device.createTexture({size:[e,i],format:"depth32float",usage:GPUTextureUsage.RENDER_ATTACHMENT}),this.depthMapTextureView=this.depthMapTexture.createView(),this.tmpDepthMapTextureView=this.tmpDepthMapTexture.createView(),this.thicknessTextureView=this.thicknessTexture.createView(),this.tmpThicknessTextureView=this.tmpThicknessTexture.createView(),this.depthTestTextureView=this.depthTestTexture.createView(),this.recreatePipelines(e,i),this.posVelBuffer&&this.updateBindGroups()}recreatePipelines(e,i){const r=this.device.createShaderModule({code:H}),c={screenWidth:e,screenHeight:i},s=10,f=Math.PI/4,_=12*(2*this.radius)*.05*(i/2)/Math.tan(f/2),P={depth_threshold:this.radius*s,max_filter_size:100,projected_particle_constant:_};this.depthFilterPipeline=this.device.createRenderPipeline({layout:"auto",vertex:{module:r,entryPoint:"vs",constants:c},fragment:{module:this.device.createShaderModule({code:K}),entryPoint:"fs",targets:[{format:"r32float"}],constants:P},primitive:{topology:"triangle-list"}}),this.thicknessFilterPipeline=this.device.createRenderPipeline({layout:"auto",vertex:{module:r,entryPoint:"vs",constants:c},fragment:{module:this.device.createShaderModule({code:Q}),entryPoint:"fs",targets:[{format:"r16float"}]},primitive:{topology:"triangle-list"}}),this.fluidPipeline=this.device.createRenderPipeline({layout:"auto",vertex:{module:r,entryPoint:"vs",constants:c},fragment:{module:this.device.createShaderModule({code:$}),entryPoint:"fs",targets:[{format:this.canvasFormat,blend:{color:{srcFactor:"src-alpha",dstFactor:"one-minus-src-alpha"},alpha:{srcFactor:"one",dstFactor:"one"}}}]},primitive:{topology:"triangle-list"},depthStencil:{format:"depth24plus",depthWriteEnabled:!0,depthCompare:"less"}})}updateBindGroup(e){this.posVelBuffer=e,this.updateBindGroups()}updateBindGroups(){this.depthMapBindGroup=this.device.createBindGroup({layout:this.depthMapPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.posVelBuffer}},{binding:1,resource:{buffer:this.renderUniformBuffer}}]}),this.thicknessMapBindGroup=this.device.createBindGroup({layout:this.thicknessMapPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.posVelBuffer}},{binding:1,resource:{buffer:this.renderUniformBuffer}}]}),this.depthFilterBindGroups=[this.device.createBindGroup({layout:this.depthFilterPipeline.getBindGroupLayout(0),entries:[{binding:1,resource:this.depthMapTextureView},{binding:2,resource:{buffer:this.filterXUniformBuffer}}]}),this.device.createBindGroup({layout:this.depthFilterPipeline.getBindGroupLayout(0),entries:[{binding:1,resource:this.tmpDepthMapTextureView},{binding:2,resource:{buffer:this.filterYUniformBuffer}}]})],this.thicknessFilterBindGroups=[this.device.createBindGroup({layout:this.thicknessFilterPipeline.getBindGroupLayout(0),entries:[{binding:1,resource:this.thicknessTextureView},{binding:2,resource:{buffer:this.filterXUniformBuffer}}]}),this.device.createBindGroup({layout:this.thicknessFilterPipeline.getBindGroupLayout(0),entries:[{binding:1,resource:this.tmpThicknessTextureView},{binding:2,resource:{buffer:this.filterYUniformBuffer}}]})],this.fluidBindGroup=this.device.createBindGroup({layout:this.fluidPipeline.getBindGroupLayout(0),entries:[{binding:1,resource:this.depthMapTextureView},{binding:2,resource:{buffer:this.renderUniformBuffer}},{binding:3,resource:this.thicknessTextureView}]})}updateUniforms(e,i){this.radius=i;const r=new Float32Array(272/4);r.set([1/this.width,1/this.height],0),r[2]=i*2,r.set(e.invProj,4),r.set(e.proj,20),r.set(e.view,36),r.set(e.invView,52),this.device.queue.writeBuffer(this.renderUniformBuffer,0,r)}record(e,i){console.error("FluidRenderer.record expects CommandEncoder, not RenderPassEncoder")}render(e,i,r,c){if(!this.depthMapBindGroup)return;const s=e.beginRenderPass({colorAttachments:[{view:this.depthMapTextureView,clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}],depthStencilAttachment:{view:this.depthTestTextureView,depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"store"}});s.setPipeline(this.depthMapPipeline),s.setBindGroup(0,this.depthMapBindGroup),s.draw(6,c),s.end();const f=e.beginRenderPass({colorAttachments:[{view:this.tmpDepthMapTextureView,clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}]});f.setPipeline(this.depthFilterPipeline),f.setBindGroup(0,this.depthFilterBindGroups[0]),f.draw(6),f.end();const h=e.beginRenderPass({colorAttachments:[{view:this.depthMapTextureView,clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}]});h.setPipeline(this.depthFilterPipeline),h.setBindGroup(0,this.depthFilterBindGroups[1]),h.draw(6),h.end();const m=e.beginRenderPass({colorAttachments:[{view:this.thicknessTextureView,clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}]});m.setPipeline(this.thicknessMapPipeline),m.setBindGroup(0,this.thicknessMapBindGroup),m.draw(6,c),m.end();const _=e.beginRenderPass({colorAttachments:[{view:this.tmpThicknessTextureView,clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}]});_.setPipeline(this.thicknessFilterPipeline),_.setBindGroup(0,this.thicknessFilterBindGroups[0]),_.draw(6),_.end();const P=e.beginRenderPass({colorAttachments:[{view:this.thicknessTextureView,clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}]});P.setPipeline(this.thicknessFilterPipeline),P.setBindGroup(0,this.thicknessFilterBindGroups[1]),P.draw(6),P.end();const d=e.beginRenderPass({colorAttachments:[{view:i,loadOp:"load",storeOp:"store"}],depthStencilAttachment:{view:r,depthLoadOp:"load",depthStoreOp:"store"}});d.setPipeline(this.fluidPipeline),d.setBindGroup(0,this.fluidBindGroup),d.draw(6),d.end()}}const x=document.getElementById("canvas"),W=document.getElementById("toggleBtn"),M=document.getElementById("status"),ge=document.getElementById("particleCount"),xe=document.getElementById("fps"),_e=document.getElementById("mass"),Pe=document.getElementById("momentum"),we=document.getElementById("error");let l,O,q,R,D,G,C,V,N,B,S=!0,ee=performance.now(),X=0,te=0,ie,k=0,E=!1,z=null;const t={renderMode:"Fluid",particleCount:3e4,gridSizeX:64,gridSizeY:64,gridSizeZ:64,spacing:.65,jitter:.5,temperature:273,dt:.1,stiffness:2.5,restDensity:4,dynamicViscosity:.08,iterations:1,fixedPointScale:1e5,visualRadius:.2,fluidRadius:1,interactionRadius:9,interactionX:32,interactionY:0,interactionZ:32,interactionActive:!0};W.addEventListener("click",()=>{S=!S,W.textContent=S?"Pause":"Resume",M.textContent=S?"running":"paused"});function oe(u){console.error(u),we.textContent=(u==null?void 0:u.stack)||String(u),M.textContent="error",S=!1,W.disabled=!0}function re(){const u=window.devicePixelRatio||1,e=Math.max(1,Math.floor(x.clientWidth*u)),i=Math.max(1,Math.floor(x.clientHeight*u));x.width=e,x.height=i,O.configure({device:l,format:q,alphaMode:"opaque"}),R&&R.destroy(),R=l.createTexture({size:[e,i],format:"depth24plus",usage:GPUTextureUsage.RENDER_ATTACHMENT}),D=R.createView(),G&&G.resize(e,i),C&&C.resize(e,i)}class ye{constructor(e){this.device=e,this.canvasAspect=1;const{vertices:i,indices:r}=ne(1,8,6);this.vertexBuffer=e.createBuffer({size:i.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(this.vertexBuffer,0,i),this.indexBuffer=e.createBuffer({size:r.byteLength,usage:GPUBufferUsage.INDEX|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(this.indexBuffer,0,r),this.indexCount=r.length,this.uniformBuffer=e.createBuffer({size:256,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});const s=e.createShaderModule({code:`
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
    `}),f=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.VERTEX,buffer:{type:"uniform"}},{binding:1,visibility:GPUShaderStage.VERTEX,buffer:{type:"read-only-storage"}}]});this.pipeline=e.createRenderPipeline({layout:e.createPipelineLayout({bindGroupLayouts:[f]}),vertex:{module:s,entryPoint:"vs_main",buffers:[{arrayStride:24,attributes:[{shaderLocation:0,offset:0,format:"float32x3"},{shaderLocation:1,offset:12,format:"float32x3"}]}]},fragment:{module:s,entryPoint:"fs_main",targets:[{format:navigator.gpu.getPreferredCanvasFormat()}]},primitive:{topology:"triangle-list",cullMode:"back"},depthStencil:{format:"depth24plus",depthWriteEnabled:!0,depthCompare:"less"}}),this.bindGroup=null}resize(e,i){this.canvasAspect=e/i}updateBindGroup(e){this.bindGroup=this.device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.uniformBuffer}},{binding:1,resource:{buffer:e}}]})}updateUniforms(e,i){const r=new Float32Array(20);r.set(e,0),r[16]=i,this.device.queue.writeBuffer(this.uniformBuffer,0,r)}record(e,i){this.bindGroup&&(e.setPipeline(this.pipeline),e.setBindGroup(0,this.bindGroup),e.setVertexBuffer(0,this.vertexBuffer),e.setIndexBuffer(this.indexBuffer,"uint16"),e.drawIndexed(this.indexCount,i))}}class Be{constructor(e){this.device=e;const{vertices:i,indices:r}=ne(1,16,12);this.vertexBuffer=e.createBuffer({size:i.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(this.vertexBuffer,0,i),this.indexBuffer=e.createBuffer({size:r.byteLength,usage:GPUBufferUsage.INDEX|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(this.indexBuffer,0,r),this.indexCount=r.length,this.uniformBuffer=e.createBuffer({size:80,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});const s=e.createShaderModule({code:`
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
        `});this.pipeline=e.createRenderPipeline({layout:"auto",vertex:{module:s,entryPoint:"vs",buffers:[{arrayStride:24,attributes:[{shaderLocation:0,offset:0,format:"float32x3"},{shaderLocation:1,offset:12,format:"float32x3"}]}]},fragment:{module:s,entryPoint:"fs",targets:[{format:navigator.gpu.getPreferredCanvasFormat()}]},primitive:{topology:"triangle-list",cullMode:"back"},depthStencil:{format:"depth24plus",depthWriteEnabled:!0,depthCompare:"less"}}),this.bindGroup=e.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.uniformBuffer}}]})}update(e,i,r){const c=new Float32Array(20);c.set(e,0),c.set(i,16),c[19]=r,this.device.queue.writeBuffer(this.uniformBuffer,0,c)}record(e){e.setPipeline(this.pipeline),e.setBindGroup(0,this.bindGroup),e.setVertexBuffer(0,this.vertexBuffer),e.setIndexBuffer(this.indexBuffer,"uint16"),e.drawIndexed(this.indexCount)}}async function g(){if(!E){E=!0,M.textContent="initializing...";try{const u=t.particleCount,e={x:t.gridSizeX,y:t.gridSizeY,z:t.gridSizeZ},i={start:[2,2,2],gridSize:e,jitter:t.jitter,spacing:t.spacing,temperature:t.temperature,restDensity:t.restDensity},r=.005,c=Math.ceil(t.dt/r),s={stiffness:t.stiffness,restDensity:t.restDensity,dynamicViscosity:t.dynamicViscosity,dt:r,fixedPointScale:t.fixedPointScale},f=l.createBuffer({size:u*32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC}),h=l.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),m=le(l,{particleCount:u,gridSize:e,iterations:c,posVelBuffer:f,interactionBuffer:h,constants:s}),_=pe({count:u,gridSize:e,...i});he(l,m.buffers.particleBuffer,_),N=m.domain,B=m.buffers,k=u,G&&G.updateBindGroup(f),C&&C.updateBindGroup(f),ge.textContent=u.toString(),M.textContent="running"}catch(u){console.warn("Init error:",u),oe(u)}finally{E=!1}}}async function be(){try{l=(await de()).device,O=x.getContext("webgpu"),q=navigator.gpu.getPreferredCanvasFormat(),G=new ye(l),C=new ve(l,q),V=new Be(l),re(),window.addEventListener("resize",re);const e=new me(x,{target:[32,32,32],radius:120});await g();const i=new window.lil.GUI({title:"MLS-MPM Controls"});(/Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent)||window.innerWidth<768)&&i.close(),i.add(t,"renderMode",["Particles","Fluid"]).name("Render Mode");const c=i.addFolder("Visuals");c.add(t,"visualRadius",.05,1,.05).name("Particle Radius"),c.add(t,"fluidRadius",.05,1,.05).name("Fluid Radius");const s=i.addFolder("Simulation");s.add(t,"particleCount",100,8e5,1e3).name("Particle Count").onFinishChange(g),s.add(t,"gridSizeX",16,256,16).name("Box X").onFinishChange(g),s.add(t,"gridSizeY",16,256,16).name("Box Y").onFinishChange(g),s.add(t,"gridSizeZ",16,256,16).name("Box Z").onFinishChange(g),s.add(t,"spacing",.1,2,.05).name("Spacing").onFinishChange(g),s.add(t,"jitter",0,1,.1).name("Jitter").onFinishChange(g),s.add(t,"temperature",0,500,1).name("Temperature (K)").onFinishChange(g);const f=i.addFolder("Physics Constants");f.add(t,"dt",.001,.2,.001).name("Time Step (dt)").onChange(()=>g()),f.add(t,"stiffness",.1,50,.1).onFinishChange(g),f.add(t,"restDensity",.1,10,.1).onFinishChange(g),f.add(t,"dynamicViscosity",0,5,.01).onFinishChange(g);const h=i.addFolder("Interaction Sphere");h.add(t,"interactionActive").name("Active"),h.add(t,"interactionRadius",.1,20).name("Radius"),h.add(t,"interactionX",0,100).name("X"),h.add(t,"interactionY",0,100).name("Y"),h.add(t,"interactionZ",0,100).name("Z"),i.add({reset:g},"reset").name("Reset Simulation"),i.add({calibrate:()=>z=null},"calibrate").name("Calibrate Sensors"),M.textContent="running";let m=!1,_=0;x.addEventListener("pointerdown",d=>{d.button===0&&d.shiftKey&&(m=!0,_=t.interactionY,d.stopImmediatePropagation())}),window.addEventListener("pointerup",()=>m=!1),window.addEventListener("deviceorientation",d=>{if(!B||!B.simUniformBuffer)return;z||(z={beta:d.beta,gamma:d.gamma});const p=.3,w=(d.beta-z.beta)*(Math.PI/180),v=(d.gamma-z.gamma)*(Math.PI/180);let a=Math.sin(v)*p,n=Math.sin(w)*p,o=-Math.sqrt(Math.max(0,p*p-a*a-n*n));const L=new Float32Array(4);L.set([a,o,n],0),l.queue.writeBuffer(B.simUniformBuffer,0,L)},!0),x.addEventListener("pointermove",d=>{if(m&&t.interactionActive){const p=x.getBoundingClientRect(),w=(d.clientX-p.left)/p.width*2-1,v=-(d.clientY-p.top)/p.height*2+1,a=e.getMatrices(p.width/p.height),n=a.invView,o=a.invProj,L=[w,v,.5,1];let b=o[0]*w+o[4]*v+o[8]*.5+o[12],F=o[1]*w+o[5]*v+o[9]*.5+o[13],U=o[2]*w+o[6]*v+o[10]*.5+o[14],I=o[3]*w+o[7]*v+o[11]*.5+o[15];b/=I,F/=I,U/=I;let se=n[0]*b+n[4]*F+n[8]*U,ae=n[1]*b+n[5]*F+n[9]*U,ue=n[2]*b+n[6]*F+n[10]*U;const T=a.eye,Fe=[se,ae,ue],A=[n[0]*b+n[4]*F+n[8]*U+n[12],n[1]*b+n[5]*F+n[9]*U+n[13],n[2]*b+n[6]*F+n[10]*U+n[14]],y=[A[0]-T[0],A[1]-T[1],A[2]-T[2]],j=Math.hypot(y[0],y[1],y[2]);if(y[0]/=j,y[1]/=j,y[2]/=j,Math.abs(y[1])>.001){const Y=(_-T[1])/y[1];Y>0&&(t.interactionX=T[0]+Y*y[0],t.interactionZ=T[2]+Y*y[2],i.controllers.forEach(ce=>ce.updateDisplay()))}}});async function P(d){const p=(d-ee)/1e3;ee=d;const w=.9;if(p>0){const a=1/p;X=X*w+a*(1-w),xe.textContent=X.toFixed(1)}if(E){ie=requestAnimationFrame(P);return}if(B&&B.interactionBuffer){const a=new Float32Array(8);t.interactionActive?(a.set([t.interactionX,t.interactionY,t.interactionZ],0),a[3]=t.interactionRadius):a[3]=-1,l.queue.writeBuffer(B.interactionBuffer,0,a)}const v=l.createCommandEncoder({label:"mpm-visual-frame"});if(S&&N&&N.step(v,t.dt),t.renderMode==="Fluid"){const a=e.getMatrices(x.width/x.height);C.updateUniforms(a,t.fluidRadius);const n=O.getCurrentTexture().createView(),o=v.beginRenderPass({colorAttachments:[{view:n,clearValue:{r:.05,g:.08,b:.14,a:1},loadOp:"clear",storeOp:"store"}],depthStencilAttachment:{view:D,depthLoadOp:"clear",depthStoreOp:"store",depthClearValue:1}});t.interactionActive&&(V.update(a.viewProj,[t.interactionX,t.interactionY,t.interactionZ],t.interactionRadius),V.record(o)),o.end(),C.render(v,n,D,k)}else{const a=e.getViewProj(x.width/x.height);G.updateUniforms(a,t.visualRadius);const n=O.getCurrentTexture().createView(),o=v.beginRenderPass({colorAttachments:[{view:n,loadOp:"clear",storeOp:"store",clearValue:{r:.05,g:.08,b:.14,a:1}}],depthStencilAttachment:{view:D,depthLoadOp:"clear",depthStoreOp:"store",depthClearValue:1}});t.interactionActive&&(V.update(a,[t.interactionX,t.interactionY,t.interactionZ],t.interactionRadius),V.record(o)),G.record(o,k),o.end()}l.queue.submit([v.finish()]),d-te>500&&S&&B&&(te=d,fe(l,B.particleBuffer,k).then(({mass:a,momentum:n})=>{_e.textContent=a.toFixed(3),Pe.textContent=n.map(o=>o.toFixed(3)).join(", ")}).catch(a=>console.warn(a))),ie=requestAnimationFrame(P)}requestAnimationFrame(P)}catch(u){oe(u)}}be();
