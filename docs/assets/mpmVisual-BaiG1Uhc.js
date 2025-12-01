import{i as pe}from"./device-CAsdAK37.js";import{a as me,s as he,b as ve,c as ge,u as xe}from"./diagnostics-BsSKq56m.js";import{O as _e,c as se}from"./orbitControls-D071pLfL.js";const J=`
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
`,Q=`
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
`,$=`
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
`,ee=`
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
`,te=`
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
`,ie=`
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
`;class Pe{constructor(e,i){this.device=e,this.canvasFormat=i,this.width=1,this.height=1,this.radius=.25;const r=e.createSampler({magFilter:"linear",minFilter:"linear"});this.sampler=r,this.renderUniformBuffer=e.createBuffer({size:272,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});const c=16;this.filterXUniformBuffer=e.createBuffer({size:c,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),this.filterYUniformBuffer=e.createBuffer({size:c,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(this.filterXUniformBuffer,0,new Float32Array([1,0])),e.queue.writeBuffer(this.filterYUniformBuffer,0,new Float32Array([0,1]));const a=e.createShaderModule({code:J});this.depthMapPipeline=e.createRenderPipeline({layout:"auto",vertex:{module:e.createShaderModule({code:Q}),entryPoint:"vs"},fragment:{module:e.createShaderModule({code:Q}),entryPoint:"fs",targets:[{format:"r32float"}]},primitive:{topology:"triangle-list"},depthStencil:{depthWriteEnabled:!0,depthCompare:"less",format:"depth32float"}}),this.thicknessMapPipeline=e.createRenderPipeline({layout:"auto",vertex:{module:e.createShaderModule({code:ee}),entryPoint:"vs"},fragment:{module:e.createShaderModule({code:ee}),entryPoint:"fs",targets:[{format:"r16float",blend:{color:{operation:"add",srcFactor:"one",dstFactor:"one"},alpha:{operation:"add",srcFactor:"one",dstFactor:"one"}}}]},primitive:{topology:"triangle-list"}}),this.depthFilterPipeline=e.createRenderPipeline({layout:"auto",vertex:{module:a,entryPoint:"vs",constants:{screenWidth:1,screenHeight:1}},fragment:{module:e.createShaderModule({code:$}),entryPoint:"fs",targets:[{format:"r32float"}],constants:{depth_threshold:.1,projected_particle_constant:100,max_filter_size:100}},primitive:{topology:"triangle-list"}}),this.thicknessFilterPipeline=e.createRenderPipeline({layout:"auto",vertex:{module:a,entryPoint:"vs",constants:{screenWidth:1,screenHeight:1}},fragment:{module:e.createShaderModule({code:te}),entryPoint:"fs",targets:[{format:"r16float"}]},primitive:{topology:"triangle-list"}}),this.fluidPipeline=e.createRenderPipeline({layout:"auto",vertex:{module:a,entryPoint:"vs",constants:{screenWidth:1,screenHeight:1}},fragment:{module:e.createShaderModule({code:ie}),entryPoint:"fs",targets:[{format:i,blend:{color:{srcFactor:"src-alpha",dstFactor:"one-minus-src-alpha"},alpha:{srcFactor:"one",dstFactor:"one"}}}]},primitive:{topology:"triangle-list"},depthStencil:{format:"depth24plus",depthWriteEnabled:!1,depthCompare:"always"}})}resize(e,i){this.width=e,this.height=i;const r=GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING;this.depthMapTexture=this.device.createTexture({size:[e,i],format:"r32float",usage:r}),this.tmpDepthMapTexture=this.device.createTexture({size:[e,i],format:"r32float",usage:r}),this.thicknessTexture=this.device.createTexture({size:[e,i],format:"r16float",usage:r}),this.tmpThicknessTexture=this.device.createTexture({size:[e,i],format:"r16float",usage:r}),this.depthTestTexture=this.device.createTexture({size:[e,i],format:"depth32float",usage:GPUTextureUsage.RENDER_ATTACHMENT}),this.depthMapTextureView=this.depthMapTexture.createView(),this.tmpDepthMapTextureView=this.tmpDepthMapTexture.createView(),this.thicknessTextureView=this.thicknessTexture.createView(),this.tmpThicknessTextureView=this.tmpThicknessTexture.createView(),this.depthTestTextureView=this.depthTestTexture.createView(),this.recreatePipelines(e,i),this.posVelBuffer&&this.updateBindGroups()}recreatePipelines(e,i){const r=this.device.createShaderModule({code:J}),c={screenWidth:e,screenHeight:i},a=10,h=Math.PI/4,p=12*(2*this.radius)*.05*(i/2)/Math.tan(h/2),g={depth_threshold:this.radius*a,max_filter_size:100,projected_particle_constant:p};this.depthFilterPipeline=this.device.createRenderPipeline({layout:"auto",vertex:{module:r,entryPoint:"vs",constants:c},fragment:{module:this.device.createShaderModule({code:$}),entryPoint:"fs",targets:[{format:"r32float"}],constants:g},primitive:{topology:"triangle-list"}}),this.thicknessFilterPipeline=this.device.createRenderPipeline({layout:"auto",vertex:{module:r,entryPoint:"vs",constants:c},fragment:{module:this.device.createShaderModule({code:te}),entryPoint:"fs",targets:[{format:"r16float"}]},primitive:{topology:"triangle-list"}}),this.fluidPipeline=this.device.createRenderPipeline({layout:"auto",vertex:{module:r,entryPoint:"vs",constants:c},fragment:{module:this.device.createShaderModule({code:ie}),entryPoint:"fs",targets:[{format:this.canvasFormat,blend:{color:{srcFactor:"src-alpha",dstFactor:"one-minus-src-alpha"},alpha:{srcFactor:"one",dstFactor:"one"}}}]},primitive:{topology:"triangle-list"},depthStencil:{format:"depth24plus",depthWriteEnabled:!0,depthCompare:"less"}})}updateBindGroup(e){this.posVelBuffer=e,this.updateBindGroups()}updateBindGroups(){this.depthMapBindGroup=this.device.createBindGroup({layout:this.depthMapPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.posVelBuffer}},{binding:1,resource:{buffer:this.renderUniformBuffer}}]}),this.thicknessMapBindGroup=this.device.createBindGroup({layout:this.thicknessMapPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.posVelBuffer}},{binding:1,resource:{buffer:this.renderUniformBuffer}}]}),this.depthFilterBindGroups=[this.device.createBindGroup({layout:this.depthFilterPipeline.getBindGroupLayout(0),entries:[{binding:1,resource:this.depthMapTextureView},{binding:2,resource:{buffer:this.filterXUniformBuffer}}]}),this.device.createBindGroup({layout:this.depthFilterPipeline.getBindGroupLayout(0),entries:[{binding:1,resource:this.tmpDepthMapTextureView},{binding:2,resource:{buffer:this.filterYUniformBuffer}}]})],this.thicknessFilterBindGroups=[this.device.createBindGroup({layout:this.thicknessFilterPipeline.getBindGroupLayout(0),entries:[{binding:1,resource:this.thicknessTextureView},{binding:2,resource:{buffer:this.filterXUniformBuffer}}]}),this.device.createBindGroup({layout:this.thicknessFilterPipeline.getBindGroupLayout(0),entries:[{binding:1,resource:this.tmpThicknessTextureView},{binding:2,resource:{buffer:this.filterYUniformBuffer}}]})],this.fluidBindGroup=this.device.createBindGroup({layout:this.fluidPipeline.getBindGroupLayout(0),entries:[{binding:1,resource:this.depthMapTextureView},{binding:2,resource:{buffer:this.renderUniformBuffer}},{binding:3,resource:this.thicknessTextureView}]})}updateUniforms(e,i){this.radius=i;const r=new Float32Array(272/4);r.set([1/this.width,1/this.height],0),r[2]=i*2,r.set(e.invProj,4),r.set(e.proj,20),r.set(e.view,36),r.set(e.invView,52),this.device.queue.writeBuffer(this.renderUniformBuffer,0,r)}record(e,i){console.error("FluidRenderer.record expects CommandEncoder, not RenderPassEncoder")}render(e,i,r,c){if(!this.depthMapBindGroup)return;const a=e.beginRenderPass({colorAttachments:[{view:this.depthMapTextureView,clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}],depthStencilAttachment:{view:this.depthTestTextureView,depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"store"}});a.setPipeline(this.depthMapPipeline),a.setBindGroup(0,this.depthMapBindGroup),a.draw(6,c),a.end();const h=e.beginRenderPass({colorAttachments:[{view:this.tmpDepthMapTextureView,clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}]});h.setPipeline(this.depthFilterPipeline),h.setBindGroup(0,this.depthFilterBindGroups[0]),h.draw(6),h.end();const y=e.beginRenderPass({colorAttachments:[{view:this.depthMapTextureView,clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}]});y.setPipeline(this.depthFilterPipeline),y.setBindGroup(0,this.depthFilterBindGroups[1]),y.draw(6),y.end();const v=e.beginRenderPass({colorAttachments:[{view:this.thicknessTextureView,clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}]});v.setPipeline(this.thicknessMapPipeline),v.setBindGroup(0,this.thicknessMapBindGroup),v.draw(6,c),v.end();const p=e.beginRenderPass({colorAttachments:[{view:this.tmpThicknessTextureView,clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}]});p.setPipeline(this.thicknessFilterPipeline),p.setBindGroup(0,this.thicknessFilterBindGroups[0]),p.draw(6),p.end();const g=e.beginRenderPass({colorAttachments:[{view:this.thicknessTextureView,clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}]});g.setPipeline(this.thicknessFilterPipeline),g.setBindGroup(0,this.thicknessFilterBindGroups[1]),g.draw(6),g.end();const b=e.beginRenderPass({colorAttachments:[{view:i,loadOp:"load",storeOp:"store"}],depthStencilAttachment:{view:r,depthLoadOp:"load",depthStoreOp:"store"}});b.setPipeline(this.fluidPipeline),b.setBindGroup(0,this.fluidBindGroup),b.draw(6),b.end()}}const _=document.getElementById("canvas"),q=document.getElementById("toggleBtn"),R=document.getElementById("status"),we=document.getElementById("particleCount"),ye=document.getElementById("fps"),Be=document.getElementById("mass"),be=document.getElementById("momentum"),Fe=document.getElementById("error");let m,D,H,O,k,C,G,V,Z,B,U=!0,re=performance.now(),N=0,ne=0,oe,E=0,L=!1,z=null;function Se(u,e){const r=new URLSearchParams(window.location.search).get(u);if(r!==null){const c=parseFloat(r);return isNaN(c)?e:c}return e}const t={renderMode:"Fluid",particleCount:Se("particles",2e4),gridSizeX:64,gridSizeY:64,gridSizeZ:64,spacing:1,jitter:.1,materialType:2,temperature:300,sceneType:"single",dt:.1,stiffness:50,restDensity:1,dynamicViscosity:.1,iterations:1,fixedPointScale:1e5,tensileStrength:5,damageRate:5,mu:50,lambda:50,visualRadius:.2,fluidRadius:1,interactionRadius:9,interactionX:32,interactionY:0,interactionZ:32,interactionActive:!0,heatSourceTemp:0};q.addEventListener("click",()=>{U=!U,q.textContent=U?"Pause":"Resume",R.textContent=U?"running":"paused"});function ue(u){console.error(u),Fe.textContent=(u==null?void 0:u.stack)||String(u),R.textContent="error",U=!1,q.disabled=!0}function ae(){const u=window.devicePixelRatio||1,e=Math.max(1,Math.floor(_.clientWidth*u)),i=Math.max(1,Math.floor(_.clientHeight*u));_.width=e,_.height=i,D.configure({device:m,format:H,alphaMode:"opaque"}),O&&O.destroy(),O=m.createTexture({size:[e,i],format:"depth24plus",usage:GPUTextureUsage.RENDER_ATTACHMENT}),k=O.createView(),C&&C.resize(e,i),G&&G.resize(e,i)}class Te{constructor(e){this.device=e,this.canvasAspect=1;const{vertices:i,indices:r}=se(1,8,6);this.vertexBuffer=e.createBuffer({size:i.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(this.vertexBuffer,0,i),this.indexBuffer=e.createBuffer({size:r.byteLength,usage:GPUBufferUsage.INDEX|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(this.indexBuffer,0,r),this.indexCount=r.length,this.uniformBuffer=e.createBuffer({size:256,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});const a=e.createShaderModule({code:`
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
    `}),h=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.VERTEX,buffer:{type:"uniform"}},{binding:1,visibility:GPUShaderStage.VERTEX,buffer:{type:"read-only-storage"}}]});this.pipeline=e.createRenderPipeline({layout:e.createPipelineLayout({bindGroupLayouts:[h]}),vertex:{module:a,entryPoint:"vs_main",buffers:[{arrayStride:24,attributes:[{shaderLocation:0,offset:0,format:"float32x3"},{shaderLocation:1,offset:12,format:"float32x3"}]}]},fragment:{module:a,entryPoint:"fs_main",targets:[{format:navigator.gpu.getPreferredCanvasFormat()}]},primitive:{topology:"triangle-list",cullMode:"back"},depthStencil:{format:"depth24plus",depthWriteEnabled:!0,depthCompare:"less"}}),this.bindGroup=null}resize(e,i){this.canvasAspect=e/i}updateBindGroup(e){this.bindGroup=this.device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.uniformBuffer}},{binding:1,resource:{buffer:e}}]})}updateUniforms(e,i){const r=new Float32Array(20);r.set(e,0),r[16]=i,this.device.queue.writeBuffer(this.uniformBuffer,0,r)}record(e,i){this.bindGroup&&(e.setPipeline(this.pipeline),e.setBindGroup(0,this.bindGroup),e.setVertexBuffer(0,this.vertexBuffer),e.setIndexBuffer(this.indexBuffer,"uint16"),e.drawIndexed(this.indexCount,i))}}class Ue{constructor(e){this.device=e;const{vertices:i,indices:r}=se(1,16,12);this.vertexBuffer=e.createBuffer({size:i.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(this.vertexBuffer,0,i),this.indexBuffer=e.createBuffer({size:r.byteLength,usage:GPUBufferUsage.INDEX|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(this.indexBuffer,0,r),this.indexCount=r.length,this.uniformBuffer=e.createBuffer({size:80,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});const a=e.createShaderModule({code:`
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
        `});this.pipeline=e.createRenderPipeline({layout:"auto",vertex:{module:a,entryPoint:"vs",buffers:[{arrayStride:24,attributes:[{shaderLocation:0,offset:0,format:"float32x3"},{shaderLocation:1,offset:12,format:"float32x3"}]}]},fragment:{module:a,entryPoint:"fs",targets:[{format:navigator.gpu.getPreferredCanvasFormat()}]},primitive:{topology:"triangle-list",cullMode:"back"},depthStencil:{format:"depth24plus",depthWriteEnabled:!0,depthCompare:"less"}}),this.bindGroup=e.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.uniformBuffer}}]})}update(e,i,r){const c=new Float32Array(20);c.set(e,0),c.set(i,16),c[19]=r,this.device.queue.writeBuffer(this.uniformBuffer,0,c)}record(e){e.setPipeline(this.pipeline),e.setBindGroup(0,this.bindGroup),e.setVertexBuffer(0,this.vertexBuffer),e.setIndexBuffer(this.indexBuffer,"uint16"),e.drawIndexed(this.indexCount)}}async function d(){if(!L){L=!0,R.textContent="initializing...";try{const u=t.particleCount,e={x:t.gridSizeX,y:t.gridSizeY,z:t.gridSizeZ},i={start:[2,2,2],gridSize:e,jitter:t.jitter,spacing:t.spacing,temperature:t.temperature,restDensity:t.restDensity,materialType:t.materialType,mu:t.mu,lambda:t.lambda},r=.005,c=Math.ceil(t.dt/r),a={stiffness:t.stiffness,restDensity:t.restDensity,dynamicViscosity:t.dynamicViscosity,dt:r,fixedPointScale:t.fixedPointScale,tensileStrength:t.tensileStrength,damageRate:t.damageRate},h=m.createBuffer({size:u*32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC}),y=m.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),v=he(m,{particleCount:u,gridSize:e,iterations:c,posVelBuffer:h,interactionBuffer:y,constants:a});let p;t.sceneType==="mixed"?p=ve({totalCount:u,gridSize:e,spacing:t.spacing,jitter:t.jitter,restDensity:t.restDensity}):p=ge({count:u,gridSize:e,...i}),xe(m,v.buffers.particleBuffer,p),Z=v.domain,B=v.buffers,E=u,C&&C.updateBindGroup(h),G&&G.updateBindGroup(h),we.textContent=u.toString(),R.textContent="running"}catch(u){console.warn("Init error:",u),ue(u)}finally{L=!1}}}async function Ce(){try{m=(await pe()).device,D=_.getContext("webgpu"),H=navigator.gpu.getPreferredCanvasFormat(),C=new Te(m),G=new Pe(m,H),V=new Ue(m),ae(),window.addEventListener("resize",ae);const e=new _e(_,{target:[32,32,32],radius:120});await d();const i=new window.lil.GUI({title:"MLS-MPM Controls"});(/Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent)||window.innerWidth<768)&&i.close(),i.add(t,"renderMode",["Particles","Fluid"]).name("Render Mode");const c=i.addFolder("Visuals");c.add(t,"visualRadius",.05,1,.05).name("Particle Radius"),c.add(t,"fluidRadius",.05,1,.05).name("Fluid Radius");const a=i.addFolder("Simulation");a.add(t,"particleCount",100,8e5,1e3).name("Particle Count").onFinishChange(d);const h={"Single Material":"single","Ice + Water":"mixed"};a.add(t,"sceneType",h).name("Scene Type").onChange(d);const y={"Ice (Brittle)":0,"Rubber (Elastic)":1,"Water (Liquid)":2,"Steam (Gas)":3};a.add(t,"materialType",y).name("Material Type").onChange(l=>{switch(parseInt(l)){case 0:t.temperature=260,t.mu=50,t.lambda=50,t.stiffness=50,t.tensileStrength=5,t.damageRate=5,t.spacing=1.2,t.jitter=0,t.restDensity=.92;break;case 1:t.temperature=300,t.mu=5,t.lambda=20,t.stiffness=50,t.spacing=1.2,t.jitter=0,t.restDensity=1;break;case 2:t.temperature=300,t.stiffness=50,t.spacing=.8,t.jitter=.3,t.restDensity=1;break;case 3:t.temperature=400,t.stiffness=50,t.spacing=2,t.jitter=.5,t.restDensity=.1;break}i.controllersRecursive().forEach(f=>f.updateDisplay()),d()}),a.add(t,"gridSizeX",16,256,16).name("Box X").onFinishChange(d),a.add(t,"gridSizeY",16,256,16).name("Box Y").onFinishChange(d),a.add(t,"gridSizeZ",16,256,16).name("Box Z").onFinishChange(d),a.add(t,"spacing",.1,2,.05).name("Spacing").onFinishChange(d),a.add(t,"jitter",0,1,.1).name("Jitter").onFinishChange(d),a.add(t,"temperature",0,500,1).name("Temperature (K)").onFinishChange(d);const v=i.addFolder("Physics Constants");v.add(t,"dt",.001,.2,.001).name("Time Step (dt)").onChange(()=>d()),v.add(t,"stiffness",.1,100,.1).name("Stiffness").onFinishChange(d),v.add(t,"restDensity",.1,10,.1).onFinishChange(d),v.add(t,"dynamicViscosity",0,5,.01).onFinishChange(d);const p=i.addFolder("Solid Properties");p.add(t,"mu",1,5e3,10).name("Shear Modulus (μ)").onFinishChange(d),p.add(t,"lambda",1,5e3,10).name("Bulk Modulus (λ)").onFinishChange(d),p.add(t,"tensileStrength",.1,100,.5).name("Tensile Strength").onFinishChange(d),p.add(t,"damageRate",.1,10,.1).name("Damage Rate").onFinishChange(d);const g=i.addFolder("Interaction Sphere");g.add(t,"interactionActive").name("Active"),g.add(t,"interactionRadius",.1,20).name("Radius"),g.add(t,"interactionX",0,100).name("X"),g.add(t,"interactionY",0,100).name("Y"),g.add(t,"interactionZ",0,100).name("Z"),g.add(t,"heatSourceTemp",0,500,10).name("Heat Source (K)"),i.add({reset:d},"reset").name("Reset Simulation"),i.add({calibrate:()=>z=null},"calibrate").name("Calibrate Sensors"),R.textContent="running";let b=!1,K=0;_.addEventListener("pointerdown",l=>{l.button===0&&l.shiftKey&&(b=!0,K=t.interactionY,l.stopImmediatePropagation())}),window.addEventListener("pointerup",()=>b=!1),window.addEventListener("deviceorientation",l=>{if(!B||!B.simUniformBuffer)return;z||(z={beta:l.beta,gamma:l.gamma});const f=.3,P=(l.beta-z.beta)*(Math.PI/180),x=(l.gamma-z.gamma)*(Math.PI/180);let o=Math.sin(x)*f,n=Math.sin(P)*f,s=-Math.sqrt(Math.max(0,f*f-o*o-n*n));const A=new Float32Array(4);A.set([o,s,n],0),m.queue.writeBuffer(B.simUniformBuffer,0,A)},!0),_.addEventListener("pointermove",l=>{if(b&&t.interactionActive){const f=_.getBoundingClientRect(),P=(l.clientX-f.left)/f.width*2-1,x=-(l.clientY-f.top)/f.height*2+1,o=e.getMatrices(f.width/f.height),n=o.invView,s=o.invProj,A=[P,x,.5,1];let F=s[0]*P+s[4]*x+s[8]*.5+s[12],S=s[1]*P+s[5]*x+s[9]*.5+s[13],T=s[2]*P+s[6]*x+s[10]*.5+s[14],j=s[3]*P+s[7]*x+s[11]*.5+s[15];F/=j,S/=j,T/=j;let ce=n[0]*F+n[4]*S+n[8]*T,de=n[1]*F+n[5]*S+n[9]*T,le=n[2]*F+n[6]*S+n[10]*T;const M=o.eye,Ge=[ce,de,le],Y=[n[0]*F+n[4]*S+n[8]*T+n[12],n[1]*F+n[5]*S+n[9]*T+n[13],n[2]*F+n[6]*S+n[10]*T+n[14]],w=[Y[0]-M[0],Y[1]-M[1],Y[2]-M[2]],X=Math.hypot(w[0],w[1],w[2]);if(w[0]/=X,w[1]/=X,w[2]/=X,Math.abs(w[1])>.001){const W=(K-M[1])/w[1];W>0&&(t.interactionX=M[0]+W*w[0],t.interactionZ=M[2]+W*w[2],i.controllers.forEach(fe=>fe.updateDisplay()))}}});async function I(l){const f=(l-re)/1e3;re=l;const P=.9;if(f>0){const o=1/f;N=N*P+o*(1-P),ye.textContent=N.toFixed(1)}if(L){oe=requestAnimationFrame(I);return}if(B&&B.interactionBuffer){const o=new Float32Array(8);t.interactionActive?(o[0]=t.interactionX,o[1]=t.interactionY,o[2]=t.interactionZ,o[3]=t.interactionRadius,o[4]=0,o[5]=0,o[6]=0,o[7]=t.heatSourceTemp):o[3]=-1,m.queue.writeBuffer(B.interactionBuffer,0,o)}const x=m.createCommandEncoder({label:"mpm-visual-frame"});if(U&&Z&&Z.step(x,t.dt),t.renderMode==="Fluid"){const o=e.getMatrices(_.width/_.height);G.updateUniforms(o,t.fluidRadius);const n=D.getCurrentTexture().createView(),s=x.beginRenderPass({colorAttachments:[{view:n,clearValue:{r:.05,g:.08,b:.14,a:1},loadOp:"clear",storeOp:"store"}],depthStencilAttachment:{view:k,depthLoadOp:"clear",depthStoreOp:"store",depthClearValue:1}});t.interactionActive&&(V.update(o.viewProj,[t.interactionX,t.interactionY,t.interactionZ],t.interactionRadius),V.record(s)),s.end(),G.render(x,n,k,E)}else{const o=e.getViewProj(_.width/_.height);C.updateUniforms(o,t.visualRadius);const n=D.getCurrentTexture().createView(),s=x.beginRenderPass({colorAttachments:[{view:n,loadOp:"clear",storeOp:"store",clearValue:{r:.05,g:.08,b:.14,a:1}}],depthStencilAttachment:{view:k,depthLoadOp:"clear",depthStoreOp:"store",depthClearValue:1}});t.interactionActive&&(V.update(o,[t.interactionX,t.interactionY,t.interactionZ],t.interactionRadius),V.record(s)),C.record(s,E),s.end()}m.queue.submit([x.finish()]),l-ne>500&&U&&B&&(ne=l,me(m,B.particleBuffer,E).then(({mass:o,momentum:n})=>{Be.textContent=o.toFixed(3),be.textContent=n.map(s=>s.toFixed(3)).join(", ")}).catch(o=>console.warn(o))),oe=requestAnimationFrame(I)}requestAnimationFrame(I)}catch(u){ue(u)}}Ce();
