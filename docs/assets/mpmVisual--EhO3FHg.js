import{i as D}from"./device-CAsdAK37.js";import{s as L,c as _,u as A,a as F}from"./diagnostics-DDWoSuSw.js";import{O as q,c as j}from"./orbitControls-DGRQJiHf.js";const o=document.getElementById("canvas"),g=document.getElementById("toggleBtn"),B=document.getElementById("status"),k=document.getElementById("particleCount"),Y=document.getElementById("fps"),N=document.getElementById("mass"),X=document.getElementById("momentum"),W=document.getElementById("error");let r,h,S,y,V,s,a=!0,C=performance.now(),p=0,E=0;const U={stiffness:2.5,restDensity:4,dynamicViscosity:.08,dt:.05,fixedPointScale:1e7};g.addEventListener("click",()=>{a=!a,g.textContent=a?"Pause":"Resume",B.textContent=a?"running":"paused"});function G(n){console.error(n),W.textContent=(n==null?void 0:n.stack)||String(n),B.textContent="error",a=!1,g.disabled=!0}function w(){const n=window.devicePixelRatio||1,e=Math.max(1,Math.floor(o.clientWidth*n)),t=Math.max(1,Math.floor(o.clientHeight*n));o.width=e,o.height=t,h.configure({device:r,format:S,alphaMode:"opaque"}),y=r.createTexture({size:[e,t],format:"depth24plus",usage:GPUTextureUsage.RENDER_ATTACHMENT}),V=y.createView(),s.resize(e,t)}class H{constructor(e){this.device=e,this.canvasAspect=1;const{vertices:t,indices:i}=j(1,8,6);this.vertexBuffer=e.createBuffer({size:t.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(this.vertexBuffer,0,t),this.indexBuffer=e.createBuffer({size:i.byteLength,usage:GPUBufferUsage.INDEX|GPUBufferUsage.COPY_DST}),e.queue.writeBuffer(this.indexBuffer,0,i),this.indexCount=i.length,this.uniformBuffer=e.createBuffer({size:256,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});const u=e.createShaderModule({code:`
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
    `}),c=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.VERTEX,buffer:{type:"uniform"}},{binding:1,visibility:GPUShaderStage.VERTEX,buffer:{type:"read-only-storage"}}]});this.pipeline=e.createRenderPipeline({layout:e.createPipelineLayout({bindGroupLayouts:[c]}),vertex:{module:u,entryPoint:"vs_main",buffers:[{arrayStride:24,attributes:[{shaderLocation:0,offset:0,format:"float32x3"},{shaderLocation:1,offset:12,format:"float32x3"}]}]},fragment:{module:u,entryPoint:"fs_main",targets:[{format:navigator.gpu.getPreferredCanvasFormat()}]},primitive:{topology:"triangle-list",cullMode:"back"},depthStencil:{format:"depth24plus",depthWriteEnabled:!0,depthCompare:"less"}}),this.bindGroup=null}resize(e,t){this.canvasAspect=e/t}updateBindGroup(e){this.bindGroup=this.device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.uniformBuffer}},{binding:1,resource:{buffer:e}}]})}updateUniforms(e,t){const i=new Float32Array(20);i.set(e,0),i[16]=t,this.device.queue.writeBuffer(this.uniformBuffer,0,i)}record(e,t){this.bindGroup&&(e.setPipeline(this.pipeline),e.setBindGroup(0,this.bindGroup),e.setVertexBuffer(0,this.vertexBuffer),e.setIndexBuffer(this.indexBuffer,"uint16"),e.drawIndexed(this.indexCount,t))}}async function J(){try{r=(await D()).device,h=o.getContext("webgpu"),S=navigator.gpu.getPreferredCanvasFormat(),s=new H(r),w(),window.addEventListener("resize",w);const e=4e3,t={x:32,y:32,z:32},i={start:[2,2,2],gridSize:t,jitter:.5,spacing:.65},d=r.createBuffer({size:e*24,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC}),{domain:u,buffers:c}=L(r,{particleCount:e,gridSize:t,iterations:1,posVelBuffer:d,constants:U}),O=_({count:e,gridSize:t,...i});A(r,c.particleBuffer,O);const T=new q(o,{target:[t.x*.5,t.y*.5,t.z*.5],radius:40});s.updateBindGroup(d),k.textContent=e.toString(),B.textContent="running";async function x(f){const v=(f-C)/1e3;C=f;const P=.9;if(v>0){const m=1/v;p=p*P+m*(1-P),Y.textContent=p.toFixed(1)}const l=r.createCommandEncoder({label:"mpm-visual-frame"});a&&u.step(l,U.dt);const I=T.getViewProj(o.width/o.height);s.updateUniforms(I,.25);const M=h.getCurrentTexture().createView(),b=l.beginRenderPass({colorAttachments:[{view:M,loadOp:"clear",storeOp:"store",clearValue:{r:.05,g:.08,b:.14,a:1}}],depthStencilAttachment:{view:V,depthLoadOp:"clear",depthStoreOp:"store",depthClearValue:1}});s.record(b,e),b.end(),r.queue.submit([l.finish()]),f-E>500&&a&&(E=f,F(r,c.particleBuffer,e).then(({mass:m,momentum:R})=>{N.textContent=m.toFixed(3),X.textContent=R.map(z=>z.toFixed(3)).join(", ")}).catch(G)),requestAnimationFrame(x)}requestAnimationFrame(x)}catch(n){G(n)}}J();
