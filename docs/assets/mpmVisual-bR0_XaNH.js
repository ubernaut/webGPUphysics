import{i as ge}from"./device-CAsdAK37.js";import{a as xe,s as ye,M as _e,c as J,u as we}from"./diagnostics-DzI7hwxu.js";import{O as be,c as de}from"./orbitControls-BUIKUCvn.js";const $=`
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
`,ee=`
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
`,te=`
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
`,re=`
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
struct PosVel { position: vec3f, materialType: f32, velocity: vec3f, temperature: f32, }
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
`,ie=`
struct RenderUniforms {
    texel_size: vec2f, sphere_size: f32,
    inv_projection_matrix: mat4x4f, projection_matrix: mat4x4f,
    view_matrix: mat4x4f, inv_view_matrix: mat4x4f,
}
struct VertexOutput {
    @builtin(position) position: vec4f, 
    @location(0) uv: vec2f, 
    @location(1) view_position: vec3f,
    @location(2) @interpolate(flat) materialType: f32,
}
struct FragmentInput { 
    @location(0) uv: vec2f, 
    @location(1) view_position: vec3f,
    @location(2) @interpolate(flat) materialType: f32,
}
struct FragmentOutput {
    @location(0) frag_color: vec4f,
    @builtin(frag_depth) frag_depth: f32,
}
struct PosVel { position: vec3f, materialType: f32, velocity: vec3f, temperature: f32, }
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
    let p = particles[instance_index];
    let view_position = (uniforms.view_matrix * vec4f(p.position, 1.0)).xyz;
    let out_position = uniforms.projection_matrix * vec4f(view_position + corner, 1.0);
    return VertexOutput(out_position, uv, view_position, p.materialType);
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
    // Output material type (stored in R channel)
    out.frag_color = vec4f(input.materialType, 0.0, 0.0, 1.0);
    return out;
}
`,ne=`
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
`,ae=`
@group(0) @binding(1) var texture: texture_2d<f32>;
@group(0) @binding(2) var<uniform> uniforms: RenderUniforms;
@group(0) @binding(3) var thickness_texture: texture_2d<f32>;
@group(0) @binding(4) var material_texture: texture_2d<f32>;

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

// Material type constants
const MATERIAL_BRITTLE_SOLID: f32 = 0.0;  // Ice
const MATERIAL_ELASTIC_SOLID: f32 = 1.0;  // Rubber
const MATERIAL_LIQUID: f32 = 2.0;          // Water
const MATERIAL_GAS: f32 = 3.0;             // Steam
const MATERIAL_GRANULAR: f32 = 4.0;        // Sand
const MATERIAL_IRON: f32 = 5.0;            // Iron

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

fn getMaterialColor(materialType: f32) -> vec3f {
    // Return base diffuse color based on material type
    if (materialType < 0.5) {
        // Ice (Brittle Solid) - Light cyan/blue
        return vec3f(0.7, 0.9, 1.0);
    } else if (materialType < 1.5) {
        // Rubber (Elastic Solid) - Red
        return vec3f(0.9, 0.2, 0.2);
    } else if (materialType < 2.5) {
        // Water (Liquid) - Blue
        return vec3f(0.1, 0.6, 0.9);
    } else if (materialType < 3.5) {
        // Steam (Gas) - White/light gray
        return vec3f(0.9, 0.9, 0.95);
    } else if (materialType < 4.5) {
        // Granular (Sand) - Tan/brown
        return vec3f(0.76, 0.70, 0.50);
    } else {
        // Iron - Dark gray
        return vec3f(0.3, 0.3, 0.35);
    }
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
    
    // Get material type from texture
    var materialType = textureLoad(material_texture, vec2u(input.iuv), 0).r;
    
    var density = 0.5; // Reduced density for more transparency
    var thickness = textureLoad(thickness_texture, vec2u(input.iuv), 0).r;
    
    // Get color based on material type
    var diffuseColor = getMaterialColor(materialType);
    
    // Adjust density/opacity based on material
    if (materialType > 4.5) {
        // Iron is more opaque
        density = 2.0;
    } else if (materialType < 1.5 && materialType > 0.5) {
        // Rubber is more opaque
        density = 1.5;
    }
    
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
    
    // Iron and rubber have less reflection (more matte)
    var reflectivity = fresnel;
    if (materialType > 4.5 || (materialType > 0.5 && materialType < 1.5)) {
        reflectivity = fresnel * 0.3;  // Reduce reflection for metals and rubber
    }
    
    var finalColor = 1.0 * specular + mix(refractionColor, reflectionColor, reflectivity);
    
    // Gamma correction
    var color = vec4f(pow(finalColor, vec3f(1.0/2.2)), 1.0);
    
    // Compute clip space depth for depth test
    var clipPos = uniforms.projection_matrix * vec4f(viewPos, 1.0);
    var fragDepth = clipPos.z / clipPos.w;

    return FragmentOutput(color, fragDepth);
}
`;class Pe{constructor(e,r){this.device=e,this.canvasFormat=r,this.width=1,this.height=1,this.radius=.25;const i=e.createSampler({magFilter:"linear",minFilter:"linear"});this.sampler=i,this.renderUniformBuffer=e.createBuffer({size:272,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});const u=16;this.filterXUniformBuffer=e.createBuffer({size:u,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),this.filterYUniformBuffer=e.createBuffer({size:u,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(this.filterXUniformBuffer,0,new Float32Array([1,0])),e.queue.writeBuffer(this.filterYUniformBuffer,0,new Float32Array([0,1]));const l=e.createShaderModule({code:$});this.depthMapPipeline=e.createRenderPipeline({layout:"auto",vertex:{module:e.createShaderModule({code:ee}),entryPoint:"vs"},fragment:{module:e.createShaderModule({code:ee}),entryPoint:"fs",targets:[{format:"r32float"}]},primitive:{topology:"triangle-list"},depthStencil:{depthWriteEnabled:!0,depthCompare:"less",format:"depth32float"}}),this.thicknessMapPipeline=e.createRenderPipeline({layout:"auto",vertex:{module:e.createShaderModule({code:re}),entryPoint:"vs"},fragment:{module:e.createShaderModule({code:re}),entryPoint:"fs",targets:[{format:"r16float",blend:{color:{operation:"add",srcFactor:"one",dstFactor:"one"},alpha:{operation:"add",srcFactor:"one",dstFactor:"one"}}}]},primitive:{topology:"triangle-list"}}),this.depthFilterPipeline=e.createRenderPipeline({layout:"auto",vertex:{module:l,entryPoint:"vs",constants:{screenWidth:1,screenHeight:1}},fragment:{module:e.createShaderModule({code:te}),entryPoint:"fs",targets:[{format:"r32float"}],constants:{depth_threshold:.1,projected_particle_constant:100,max_filter_size:100}},primitive:{topology:"triangle-list"}}),this.thicknessFilterPipeline=e.createRenderPipeline({layout:"auto",vertex:{module:l,entryPoint:"vs",constants:{screenWidth:1,screenHeight:1}},fragment:{module:e.createShaderModule({code:ne}),entryPoint:"fs",targets:[{format:"r16float"}]},primitive:{topology:"triangle-list"}}),this.materialMapPipeline=e.createRenderPipeline({layout:"auto",vertex:{module:e.createShaderModule({code:ie}),entryPoint:"vs"},fragment:{module:e.createShaderModule({code:ie}),entryPoint:"fs",targets:[{format:"r32float"}]},primitive:{topology:"triangle-list"},depthStencil:{depthWriteEnabled:!0,depthCompare:"less",format:"depth32float"}}),this.fluidPipeline=e.createRenderPipeline({layout:"auto",vertex:{module:l,entryPoint:"vs",constants:{screenWidth:1,screenHeight:1}},fragment:{module:e.createShaderModule({code:ae}),entryPoint:"fs",targets:[{format:r,blend:{color:{srcFactor:"src-alpha",dstFactor:"one-minus-src-alpha"},alpha:{srcFactor:"one",dstFactor:"one"}}}]},primitive:{topology:"triangle-list"},depthStencil:{format:"depth24plus",depthWriteEnabled:!1,depthCompare:"always"}})}resize(e,r){this.width=e,this.height=r;const i=GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING;this.depthMapTexture=this.device.createTexture({size:[e,r],format:"r32float",usage:i}),this.tmpDepthMapTexture=this.device.createTexture({size:[e,r],format:"r32float",usage:i}),this.thicknessTexture=this.device.createTexture({size:[e,r],format:"r16float",usage:i}),this.tmpThicknessTexture=this.device.createTexture({size:[e,r],format:"r16float",usage:i}),this.depthTestTexture=this.device.createTexture({size:[e,r],format:"depth32float",usage:GPUTextureUsage.RENDER_ATTACHMENT}),this.materialTexture=this.device.createTexture({size:[e,r],format:"r32float",usage:i}),this.materialTextureView=this.materialTexture.createView(),this.materialDepthTexture=this.device.createTexture({size:[e,r],format:"depth32float",usage:GPUTextureUsage.RENDER_ATTACHMENT}),this.materialDepthTextureView=this.materialDepthTexture.createView(),this.depthMapTextureView=this.depthMapTexture.createView(),this.tmpDepthMapTextureView=this.tmpDepthMapTexture.createView(),this.thicknessTextureView=this.thicknessTexture.createView(),this.tmpThicknessTextureView=this.tmpThicknessTexture.createView(),this.depthTestTextureView=this.depthTestTexture.createView(),this.recreatePipelines(e,r),this.posVelBuffer&&this.updateBindGroups()}recreatePipelines(e,r){const i=this.device.createShaderModule({code:$}),u={screenWidth:e,screenHeight:r},l=10,x=Math.PI/4,_=12*(2*this.radius)*.05*(r/2)/Math.tan(x/2),B={depth_threshold:this.radius*l,max_filter_size:100,projected_particle_constant:_};this.depthFilterPipeline=this.device.createRenderPipeline({layout:"auto",vertex:{module:i,entryPoint:"vs",constants:u},fragment:{module:this.device.createShaderModule({code:te}),entryPoint:"fs",targets:[{format:"r32float"}],constants:B},primitive:{topology:"triangle-list"}}),this.thicknessFilterPipeline=this.device.createRenderPipeline({layout:"auto",vertex:{module:i,entryPoint:"vs",constants:u},fragment:{module:this.device.createShaderModule({code:ne}),entryPoint:"fs",targets:[{format:"r16float"}]},primitive:{topology:"triangle-list"}}),this.fluidPipeline=this.device.createRenderPipeline({layout:"auto",vertex:{module:i,entryPoint:"vs",constants:u},fragment:{module:this.device.createShaderModule({code:ae}),entryPoint:"fs",targets:[{format:this.canvasFormat,blend:{color:{srcFactor:"src-alpha",dstFactor:"one-minus-src-alpha"},alpha:{srcFactor:"one",dstFactor:"one"}}}]},primitive:{topology:"triangle-list"},depthStencil:{format:"depth24plus",depthWriteEnabled:!0,depthCompare:"less"}})}updateBindGroup(e){this.posVelBuffer=e,this.updateBindGroups()}updateBindGroups(){this.depthMapBindGroup=this.device.createBindGroup({layout:this.depthMapPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.posVelBuffer}},{binding:1,resource:{buffer:this.renderUniformBuffer}}]}),this.thicknessMapBindGroup=this.device.createBindGroup({layout:this.thicknessMapPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.posVelBuffer}},{binding:1,resource:{buffer:this.renderUniformBuffer}}]}),this.depthFilterBindGroups=[this.device.createBindGroup({layout:this.depthFilterPipeline.getBindGroupLayout(0),entries:[{binding:1,resource:this.depthMapTextureView},{binding:2,resource:{buffer:this.filterXUniformBuffer}}]}),this.device.createBindGroup({layout:this.depthFilterPipeline.getBindGroupLayout(0),entries:[{binding:1,resource:this.tmpDepthMapTextureView},{binding:2,resource:{buffer:this.filterYUniformBuffer}}]})],this.thicknessFilterBindGroups=[this.device.createBindGroup({layout:this.thicknessFilterPipeline.getBindGroupLayout(0),entries:[{binding:1,resource:this.thicknessTextureView},{binding:2,resource:{buffer:this.filterXUniformBuffer}}]}),this.device.createBindGroup({layout:this.thicknessFilterPipeline.getBindGroupLayout(0),entries:[{binding:1,resource:this.tmpThicknessTextureView},{binding:2,resource:{buffer:this.filterYUniformBuffer}}]})],this.materialMapBindGroup=this.device.createBindGroup({layout:this.materialMapPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.posVelBuffer}},{binding:1,resource:{buffer:this.renderUniformBuffer}}]}),this.fluidBindGroup=this.device.createBindGroup({layout:this.fluidPipeline.getBindGroupLayout(0),entries:[{binding:1,resource:this.depthMapTextureView},{binding:2,resource:{buffer:this.renderUniformBuffer}},{binding:3,resource:this.thicknessTextureView},{binding:4,resource:this.materialTextureView}]})}updateUniforms(e,r){this.radius=r;const i=new Float32Array(272/4);i.set([1/this.width,1/this.height],0),i[2]=r*2,i.set(e.invProj,4),i.set(e.proj,20),i.set(e.view,36),i.set(e.invView,52),this.device.queue.writeBuffer(this.renderUniformBuffer,0,i)}record(e,r){console.error("FluidRenderer.record expects CommandEncoder, not RenderPassEncoder")}render(e,r,i,u){if(!this.depthMapBindGroup)return;const l=e.beginRenderPass({colorAttachments:[{view:this.depthMapTextureView,clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}],depthStencilAttachment:{view:this.depthTestTextureView,depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"store"}});l.setPipeline(this.depthMapPipeline),l.setBindGroup(0,this.depthMapBindGroup),l.draw(6,u),l.end();const x=e.beginRenderPass({colorAttachments:[{view:this.tmpDepthMapTextureView,clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}]});x.setPipeline(this.depthFilterPipeline),x.setBindGroup(0,this.depthFilterBindGroups[0]),x.draw(6),x.end();const h=e.beginRenderPass({colorAttachments:[{view:this.depthMapTextureView,clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}]});h.setPipeline(this.depthFilterPipeline),h.setBindGroup(0,this.depthFilterBindGroups[1]),h.draw(6),h.end();const v=e.beginRenderPass({colorAttachments:[{view:this.materialTextureView,clearValue:{r:2,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}],depthStencilAttachment:{view:this.materialDepthTextureView,depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"store"}});v.setPipeline(this.materialMapPipeline),v.setBindGroup(0,this.materialMapBindGroup),v.draw(6,u),v.end();const _=e.beginRenderPass({colorAttachments:[{view:this.thicknessTextureView,clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}]});_.setPipeline(this.thicknessMapPipeline),_.setBindGroup(0,this.thicknessMapBindGroup),_.draw(6,u),_.end();const B=e.beginRenderPass({colorAttachments:[{view:this.tmpThicknessTextureView,clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}]});B.setPipeline(this.thicknessFilterPipeline),B.setBindGroup(0,this.thicknessFilterBindGroups[0]),B.draw(6),B.end();const T=e.beginRenderPass({colorAttachments:[{view:this.thicknessTextureView,clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}]});T.setPipeline(this.thicknessFilterPipeline),T.setBindGroup(0,this.thicknessFilterBindGroups[1]),T.draw(6),T.end();const a=e.beginRenderPass({colorAttachments:[{view:r,loadOp:"load",storeOp:"store"}],depthStencilAttachment:{view:i,depthLoadOp:"load",depthStoreOp:"store"}});a.setPipeline(this.fluidPipeline),a.setBindGroup(0,this.fluidBindGroup),a.draw(6),a.end()}}const y=document.getElementById("canvas"),Z=document.getElementById("toggleBtn"),E=document.getElementById("status"),Be=document.getElementById("particleCount"),Te=document.getElementById("fps"),Se=document.getElementById("mass"),Fe=document.getElementById("momentum"),Ue=document.getElementById("error");let g,I,K,D,k,C,R,O,Q,P,G=!0,oe=performance.now(),H=0,se=0,ue,Y=0,j=!1,A=null,ce=!1,b,Me;const z=[{key:"Hydrogen",id:0},{key:"Oxygen",id:1},{key:"Sodium",id:2},{key:"Potassium",id:3},{key:"Magnesium",id:4},{key:"Aluminum",id:5},{key:"Silicon",id:6},{key:"Calcium",id:7},{key:"Titanium",id:8},{key:"Iron",id:9},{key:"Lead",id:10}];function Ge(s,e){const i=new URLSearchParams(window.location.search).get(s);if(i!==null){const u=parseFloat(i);return isNaN(u)?e:u}return e}const t={renderMode:"Fluid",particleCount:Ge("particles",2e4),gridSizeX:64,gridSizeY:64,gridSizeZ:64,spacing:1,jitter:.1,materialA:"Oxygen",materialB:"Iron",temperature:300,dt:.1,gravity:-.3,ambientPressure:1,stiffness:50,restDensity:1,dynamicViscosity:.1,iterations:1,fixedPointScale:1e5,visualRadius:.2,fluidRadius:1,interactionRadius:9,interactionX:32,interactionY:10,interactionZ:32,interactionActive:!0,heatSourceTemp:400};Z.addEventListener("click",()=>{G=!G,Z.textContent=G?"Pause":"Resume",E.textContent=G?"running":"paused"});function fe(s){console.error(s),Ue.textContent=(s==null?void 0:s.stack)||String(s),E.textContent="error",G=!1,Z.disabled=!0}function d(s,e){return s!=null&&s.domElement&&(s.domElement.title=e),s}function le(){const s=window.devicePixelRatio||1,e=Math.max(1,Math.floor(y.clientWidth*s)),r=Math.max(1,Math.floor(y.clientHeight*s));y.width=e,y.height=r,I.configure({device:g,format:K,alphaMode:"opaque"}),D&&D.destroy(),D=g.createTexture({size:[e,r],format:"depth24plus",usage:GPUTextureUsage.RENDER_ATTACHMENT}),k=D.createView(),C&&C.resize(e,r),R&&R.resize(e,r)}class Ce{constructor(e){this.device=e,this.canvasAspect=1;const{vertices:r,indices:i}=de(1,8,6);this.vertexBuffer=e.createBuffer({size:r.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(this.vertexBuffer,0,r),this.indexBuffer=e.createBuffer({size:i.byteLength,usage:GPUBufferUsage.INDEX|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(this.indexBuffer,0,i),this.indexCount=i.length,this.uniformBuffer=e.createBuffer({size:256,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});const l=e.createShaderModule({code:`
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
    `}),x=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.VERTEX,buffer:{type:"uniform"}},{binding:1,visibility:GPUShaderStage.VERTEX,buffer:{type:"read-only-storage"}}]});this.pipeline=e.createRenderPipeline({layout:e.createPipelineLayout({bindGroupLayouts:[x]}),vertex:{module:l,entryPoint:"vs_main",buffers:[{arrayStride:24,attributes:[{shaderLocation:0,offset:0,format:"float32x3"},{shaderLocation:1,offset:12,format:"float32x3"}]}]},fragment:{module:l,entryPoint:"fs_main",targets:[{format:navigator.gpu.getPreferredCanvasFormat()}]},primitive:{topology:"triangle-list",cullMode:"back"},depthStencil:{format:"depth24plus",depthWriteEnabled:!0,depthCompare:"less"}}),this.bindGroup=null}resize(e,r){this.canvasAspect=e/r}updateBindGroup(e){this.bindGroup=this.device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.uniformBuffer}},{binding:1,resource:{buffer:e}}]})}updateUniforms(e,r){const i=new Float32Array(20);i.set(e,0),i[16]=r,this.device.queue.writeBuffer(this.uniformBuffer,0,i)}record(e,r){this.bindGroup&&(e.setPipeline(this.pipeline),e.setBindGroup(0,this.bindGroup),e.setVertexBuffer(0,this.vertexBuffer),e.setIndexBuffer(this.indexBuffer,"uint16"),e.drawIndexed(this.indexCount,r))}}class Re{constructor(e){this.device=e;const{vertices:r,indices:i}=de(1,16,12);this.vertexBuffer=e.createBuffer({size:r.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(this.vertexBuffer,0,r),this.indexBuffer=e.createBuffer({size:i.byteLength,usage:GPUBufferUsage.INDEX|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(this.indexBuffer,0,i),this.indexCount=i.length,this.uniformBuffer=e.createBuffer({size:80,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});const l=e.createShaderModule({code:`
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
        `});this.pipeline=e.createRenderPipeline({layout:"auto",vertex:{module:l,entryPoint:"vs",buffers:[{arrayStride:24,attributes:[{shaderLocation:0,offset:0,format:"float32x3"},{shaderLocation:1,offset:12,format:"float32x3"}]}]},fragment:{module:l,entryPoint:"fs",targets:[{format:navigator.gpu.getPreferredCanvasFormat()}]},primitive:{topology:"triangle-list",cullMode:"back"},depthStencil:{format:"depth24plus",depthWriteEnabled:!0,depthCompare:"less"}}),this.bindGroup=e.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.uniformBuffer}}]})}update(e,r,i){const u=new Float32Array(20);u.set(e,0),u.set(r,16),u[19]=i,this.device.queue.writeBuffer(this.uniformBuffer,0,u)}record(e){e.setPipeline(this.pipeline),e.setBindGroup(0,this.bindGroup),e.setVertexBuffer(0,this.vertexBuffer),e.setIndexBuffer(this.indexBuffer,"uint16"),e.drawIndexed(this.indexCount)}}async function w(){if(!j){j=!0,E.textContent="initializing...";try{const s=t.particleCount,e={x:t.gridSizeX,y:t.gridSizeY,z:t.gridSizeZ},r=.005,i=Math.ceil(t.dt/r),u={stiffness:t.stiffness,restDensity:t.restDensity,dynamicViscosity:t.dynamicViscosity,dt:r,fixedPointScale:t.fixedPointScale,tensileStrength:0,damageRate:0,ambientPressure:t.ambientPressure},l=g.createBuffer({size:s*32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC}),x=g.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),h=ye(g,{particleCount:s,gridSize:e,iterations:i,posVelBuffer:l,interactionBuffer:x,constants:u}),v=Math.floor(s/2),_=s-v,B=Math.ceil(Math.cbrt(v)),T=Math.ceil(Math.cbrt(_)),a=z.find(c=>c.key===t.materialA)??z[0],f=z.find(c=>c.key===t.materialB)??z[1]??z[0],p=new ArrayBuffer(s*_e),m=J({count:v,gridSize:e,start:[2,2,2],spacing:t.spacing,jitter:t.jitter,temperature:t.temperature,restDensity:t.restDensity,materialType:a.id,cubeSideCount:B}),n=J({count:_,gridSize:e,start:[e.x*.4,e.y*.4,e.z*.4],spacing:t.spacing,jitter:t.jitter,temperature:t.temperature,restDensity:t.restDensity,materialType:f.id,cubeSideCount:T});new Uint8Array(p,0,m.byteLength).set(new Uint8Array(m)),new Uint8Array(p,m.byteLength,n.byteLength).set(new Uint8Array(n));const o=p;we(g,h.buffers.particleBuffer,o),Q=h.domain,P=h.buffers,Y=s,C&&C.updateBindGroup(l),R&&R.updateBindGroup(l),Be.textContent=s.toString(),E.textContent="running"}catch(s){console.warn("Init error:",s),fe(s)}finally{j=!1}}}async function Ve(){try{let x=function(){const a=t.gridSizeX/2,f=t.gridSizeY/2,p=t.gridSizeZ/2;e.setTarget([a,f,p]);const m=Math.max(t.gridSizeX,t.gridSizeY,t.gridSizeZ);e.setRadius(m*1.8),t.interactionX=a,t.interactionY=f*.3,t.interactionZ=p,b.controllersRecursive().forEach(n=>n.updateDisplay()),w()};g=(await ge()).device,I=y.getContext("webgpu"),K=navigator.gpu.getPreferredCanvasFormat(),C=new Ce(g),R=new Pe(g,K),O=new Re(g),le(),window.addEventListener("resize",le);const e=new be(y,{target:[32,32,32],radius:120});await w(),b=new window.lil.GUI({title:"MLS-MPM Controls"}),(/Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent)||window.innerWidth<768)&&b.close(),d(b.add(t,"renderMode",["Particles","Fluid"]).name("Render Mode"),"Choose between particle instancing or screen-space fluid rendering");const i=b.addFolder("Visuals");d(i.add(t,"visualRadius",.05,1,.05).name("Particle Radius"),"Sphere radius for particle rendering (visual only)"),d(i.add(t,"fluidRadius",.05,3,.05).name("Fluid Radius"),"Splat radius for screen-space fluid rendering");const u=b.addFolder("Simulation");d(u.add(t,"particleCount",100,1e5,1e3).name("Particle Count").onFinishChange(w),"Total number of particles spawned (higher = heavier GPU load)");const l={};z.forEach(a=>{l[a.key]=a.key}),d(u.add(t,"materialA",l).name("Element A").onChange(w),"Element for first spawn block (50% of particles)"),d(u.add(t,"materialB",l).name("Element B").onChange(w),"Element for second spawn block (50% of particles)"),u.add(t,"gridSizeX",16,256,16).name("Box X").onFinishChange(x),u.add(t,"gridSizeY",16,256,16).name("Box Y").onFinishChange(x),u.add(t,"gridSizeZ",16,256,16).name("Box Z").onFinishChange(x),d(u.add(t,"spacing",.1,2.5,.05).name("Spacing").onFinishChange(w),"Initial particle pitch (grid units); smaller = denser packing"),d(u.add(t,"jitter",0,1,.05).name("Jitter").onFinishChange(w),"Random perturbation of initial positions to break symmetry"),Me=d(u.add(t,"temperature",0,5e3,1).name("Temperature (K)").onFinishChange(w),"Initial particle temperature for the selected material");const h=b.addFolder("Physics Constants");d(h.add(t,"dt",.001,.2,.001).name("Time Step (dt)").onChange(()=>w()),"Simulation timestep (seconds) per sub-step; smaller = more stable"),d(h.add(t,"gravity",-20,20,.1).name("Gravity (Y)"),"Gravity acceleration along Y (negative = downward)"),d(h.add(t,"ambientPressure",0,5,.05).name("Ambient Pressure"),"Ambient pressure baseline applied to fluids/gases (dimensionless)"),d(h.add(t,"stiffness",.1,100,.1).name("Stiffness").onFinishChange(w),"Fluid bulk modulus for Tait EOS (affects compressibility)"),d(h.add(t,"restDensity",.1,10,.1).onFinishChange(w),"Target rest density for mass/volume and pressure calculations"),d(h.add(t,"dynamicViscosity",0,5,.01).onFinishChange(w),"Viscosity term applied to fluids (damps shear/velocity gradients)");const v=b.addFolder("Interaction Sphere");d(v.add(t,"interactionActive").name("Active"),"Enable or disable the interaction sphere"),d(v.add(t,"interactionRadius",.1,20).name("Radius"),"Sphere radius for collision/thermal interaction"),d(v.add(t,"interactionX",0,100).name("X"),"Sphere center X (grid units)"),d(v.add(t,"interactionY",0,100).name("Y"),"Sphere center Y (grid units)"),d(v.add(t,"interactionZ",0,100).name("Z"),"Sphere center Z (grid units)"),d(v.add(t,"heatSourceTemp",0,5e3,1).name("Heat Source (K)"),"Temperature of the heat/cool sphere (0 disables thermal effect)"),d(b.add({reset:w},"reset").name("Reset Simulation"),"Rebuild buffers and restart the current scene"),d(b.add({calibrate:()=>A=null},"calibrate").name("Calibrate Sensors"),"Reset device-orientation baseline for gravity control"),E.textContent="running";let _=!1,B=0;y.addEventListener("pointerdown",a=>{a.button===1&&t.interactionActive&&(_=!0,B=t.interactionY,a.preventDefault(),a.stopImmediatePropagation())}),window.addEventListener("pointerup",()=>_=!1),y.addEventListener("wheel",a=>{if(a.ctrlKey&&t.interactionActive){a.preventDefault();const f=a.deltaY>0?-1:1;t.interactionY=Math.max(0,Math.min(t.gridSizeY,t.interactionY+f)),b.controllersRecursive().forEach(p=>p.updateDisplay())}},{passive:!1}),window.addEventListener("deviceorientation",a=>{if(!P||!P.simUniformBuffer||a.beta===null||a.gamma===null)return;A||(A={beta:a.beta,gamma:a.gamma}),ce=!0;const f=.3,p=(a.beta-A.beta)*(Math.PI/180),m=(a.gamma-A.gamma)*(Math.PI/180);let n=Math.sin(m)*f,o=Math.sin(p)*f,c=-Math.sqrt(Math.max(0,f*f-n*n-o*o));const L=new Float32Array(4);L.set([n,c,o],0),L[3]=t.ambientPressure,g.queue.writeBuffer(P.simUniformBuffer,0,L)},!0),y.addEventListener("pointermove",a=>{if(_&&t.interactionActive){const f=y.getBoundingClientRect(),p=(a.clientX-f.left)/f.width*2-1,m=-(a.clientY-f.top)/f.height*2+1,n=e.getMatrices(f.width/f.height),o=n.invView,c=n.invProj,L=[p,m,.5,1];let F=c[0]*p+c[4]*m+c[8]*.5+c[12],U=c[1]*p+c[5]*m+c[9]*.5+c[13],M=c[2]*p+c[6]*m+c[10]*.5+c[14],X=c[3]*p+c[7]*m+c[11]*.5+c[15];F/=X,U/=X,M/=X;let pe=o[0]*F+o[4]*U+o[8]*M,me=o[1]*F+o[5]*U+o[9]*M,he=o[2]*F+o[6]*U+o[10]*M;const V=n.eye,ze=[pe,me,he],W=[o[0]*F+o[4]*U+o[8]*M+o[12],o[1]*F+o[5]*U+o[9]*M+o[13],o[2]*F+o[6]*U+o[10]*M+o[14]],S=[W[0]-V[0],W[1]-V[1],W[2]-V[2]],N=Math.hypot(S[0],S[1],S[2]);if(S[0]/=N,S[1]/=N,S[2]/=N,Math.abs(S[1])>.001){const q=(B-V[1])/S[1];q>0&&(t.interactionX=V[0]+q*S[0],t.interactionZ=V[2]+q*S[2],b.controllers.forEach(ve=>ve.updateDisplay()))}}});async function T(a){const f=(a-oe)/1e3;oe=a;const p=.9;if(f>0){const n=1/f;H=H*p+n*(1-p),Te.textContent=H.toFixed(1)}if(j){ue=requestAnimationFrame(T);return}if(P&&P.simUniformBuffer&&!ce){const n=new Float32Array(4);n[0]=0,n[1]=t.gravity,n[2]=0,n[3]=t.ambientPressure,g.queue.writeBuffer(P.simUniformBuffer,0,n)}if(P&&P.interactionBuffer){const n=new Float32Array(8);t.interactionActive?(n[0]=t.interactionX,n[1]=t.interactionY,n[2]=t.interactionZ,n[3]=t.interactionRadius,n[4]=0,n[5]=0,n[6]=0,n[7]=t.heatSourceTemp):n[3]=-1,g.queue.writeBuffer(P.interactionBuffer,0,n)}const m=g.createCommandEncoder({label:"mpm-visual-frame"});if(G&&Q&&Q.step(m,t.dt),t.renderMode==="Fluid"){const n=e.getMatrices(y.width/y.height);R.updateUniforms(n,t.fluidRadius);const o=I.getCurrentTexture().createView(),c=m.beginRenderPass({colorAttachments:[{view:o,clearValue:{r:.05,g:.08,b:.14,a:1},loadOp:"clear",storeOp:"store"}],depthStencilAttachment:{view:k,depthLoadOp:"clear",depthStoreOp:"store",depthClearValue:1}});t.interactionActive&&(O.update(n.viewProj,[t.interactionX,t.interactionY,t.interactionZ],t.interactionRadius),O.record(c)),c.end(),R.render(m,o,k,Y)}else{const n=e.getViewProj(y.width/y.height);C.updateUniforms(n,t.visualRadius);const o=I.getCurrentTexture().createView(),c=m.beginRenderPass({colorAttachments:[{view:o,loadOp:"clear",storeOp:"store",clearValue:{r:.05,g:.08,b:.14,a:1}}],depthStencilAttachment:{view:k,depthLoadOp:"clear",depthStoreOp:"store",depthClearValue:1}});t.interactionActive&&(O.update(n,[t.interactionX,t.interactionY,t.interactionZ],t.interactionRadius),O.record(c)),C.record(c,Y),c.end()}g.queue.submit([m.finish()]),a-se>500&&G&&P&&(se=a,xe(g,P.particleBuffer,Y).then(({mass:n,momentum:o})=>{Se.textContent=n.toFixed(3),Fe.textContent=o.map(c=>c.toFixed(3)).join(", ")}).catch(n=>console.warn(n))),ue=requestAnimationFrame(T)}requestAnimationFrame(T)}catch(s){fe(s)}}Ve();
