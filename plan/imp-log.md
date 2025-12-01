llm instructions for this file:
this file contains a concise narrative of the development. including major decisions, things we've tried, bugs encountered etc. reading this file should enable an llm to pick up quickly where we left off.  

# WebGPU Physics Implementation Log

## Project Background (condensed)
- Ported the original gpu-physics.js (WebGL) to WebGPU rigid-body pipeline (buffers + WGSL). Core modules: `src/device.js`, `src/world.js`, `src/math.js`, 14 WGSL compute passes for rigid bodies. Demo exists but had WebGPU validation errors (binding mismatches) to debug later. Current branch shifts toward MLS-MPM materials while keeping the rigid-body work as context.

## Session 1 - 2025-11-27 (summary)
- Analyzed legacy codebase, shader pipeline, data layout, and built the initial WebGPU port plan.
- Set up Vite/TypeScript/WebGPU tooling and ported the rigid-body compute passes. Builds succeed.

## Session 2 - Demo Creation (summary)
- Added `demos/` with an instanced boxes demo; physics runs but shows validation errors likely due to bind group mismatches. Needs a binding audit.

## Session 3 - MLS-MPM Branch Kickoff (current)
- Read the MLS-MPM guide and WebGPU-Ocean as primary references.
- Updated `imp-plan.md` to lead with the overarching project goal (multi-material, thermodynamics, peercompute-compatible) and clarified branch goals (water-first MLS-MPM).
- Set short-term branch tasks: capture WebGPU-Ocean buffer/kernels, choose integration path (reuse `src/device.js` vs. parallel module), define particle/grid/material schemas, and outline test harness for water MVP (mass/momentum checks).

## Session 4 - WebGPU-Ocean MLS-MPM pass (2025-11-30)
- Cloned WebGPU-Ocean and reviewed MLS-MPM: pipelines = clearGrid → p2g_1 (mass/momentum scatter) → p2g_2 (stress/pressure scatter) → updateGrid → g2p → copyPosition, looped twice per frame; workgroup size 64.
- Buffer layouts observed: Particle AoS 80B (pos vec3+pad, vel vec3+pad, C mat3x3); Grid 16B (vx, vy, vz, mass) using fixed-point `i32` atomics with multiplier 1e7. Constants: dt=0.2, stiffness=3, rest_density=4, dynamic_viscosity=0.1; grid up to 64^3.
- Boundary conditions: grid velocities zeroed near edges; particle boundary softening in g2p with predictive check.
- Decisions: keep fixed-point atomic scatter for P2G prototype; wrap existing rigid `World` as `RigidDomain` and add a simple `Engine` to sequence domains before full scheduler; lock schema/offsets in plan for peercompute; target per-step volume recompute for fluids (density gather) with optional det(F) path for solids.
- Added `src/domains/mpm/schema.js` with particle/grid strides, offsets, fixed-point constants, and buffer/view helpers to standardize layouts.
- Added `src/engine.js` (minimal domain sequencer) to ease future multi-domain stepping without refactoring existing `World` immediately.
- Added `src/domains/mpm/domain.js` scaffold and `src/domains/mpm/index.js` export to formalize MLS-MPM domain wiring (pipelines injected externally, ordered dispatch helper).
- Added `src/domains/mpm/shaders.js` and `src/domains/mpm/pipelines.js` with baseline WGSL kernels (clearGrid, p2g1, p2g2, updateGrid, g2p, copyPosition) adapted to the new 128B particle schema and fixed-point atomics; pipeline/bind group factory for quick wiring.
- Added `src/domains/mpm/factory.js` (`setupMpmDomain`, `uploadParticleData`) to wire buffers/pipelines/bind groups; added `src/domains/mpm/init.js` to generate block particle data (dam-break style) aligned with the 128B schema.
- Added `src/domains/mpm/headless.js` (`createHeadlessMpm`) to spin up a headless MLS-MPM sim with optional generated particle data; exported Engine/mpm namespace via `src/index.js`. Vite build passes.
- Added `src/domains/mpm/diagnostics.js` with `computeMassMomentum` to CPU-check conserved quantities via staging buffer readback.
- Added `demos/mpm-headless.html` and `demos/mpm-headless.js` to run MLS-MPM without rendering and display mass/momentum diagnostics using `createHeadlessMpm` and `computeMassMomentum`.
- Headless demo now tracks baseline deltas (mass/momentum Δ) to monitor drift over time.
- Vite build inputs updated to include the headless demo; demos index now lists both toychest and mpm-headless. Headless UI flags drift beyond tolerance.
- Tuned water defaults in demos (stiffness 2.5, restDensity 4.0, viscosity 0.08, dt 0.05) and added baseline reset + drift alerts in headless UI.
- Added visual MLS-MPM demo (`demos/mpm-visual.html`/`.js`) rendering particles with orbit controls; uses copyPosition buffer for instancing. Demos index and Vite inputs updated to include it.

## Session 5 - Debugging "Settle and Explode" & UI Polish (2025-11-30)
- **Problem:** Functional demo exhibited strange behavior where particles would settle briefly and then explode randomly.
- **Investigation:** Compared current implementation with [WebGPU-Ocean](https://github.com/matsuoka-601/WebGPU-Ocean) reference and [NiallTL's MPM Guide](https://nialltl.neocities.org/articles/mpm_guide).
- **Finding:** A critical buffer struct alignment bug in `src/domains/mpm/schema.js`.
    - WGSL `std430` layout mandates that `mat3x3f` occupies **48 bytes** (3 columns * 16-byte alignment), not 36 bytes (9 floats) as assumed.
    - The schema had `F` (deformation gradient) at offset 48 and `C` (velocity gradient) at offset 84.
    - This caused `C` to overlap with the last column of `F` (48-96). The shader correctly read `C` at offset 96.
    - **Result:** The JS initialization wrote zeros to `C` at offset 84 (corrupting `F`), while the shader read `C` from offset 96 (uninitialized/shifted data). This corrupted the affine velocity transfer `Q = C * dist`, injecting massive energy and exploding the simulation.
- **Fix:**
    - Updated `src/domains/mpm/schema.js` to correctly place `C` at offset 96 and increase `MPM_PARTICLE_STRIDE` to 144 bytes.
    - Updated `src/domains/mpm/shaders.js` to remove unnecessary padding from the `Particle` struct (144 bytes is 16-aligned).
- **UI Enhancement:**
    - Added `lil-gui` to `demos/mpm-visual.html` and refactored `demos/mpm-visual.js` to include interactive controls.
    - Exposed controls for: Particle Count, Grid Size, Spacing, Jitter, Time Step (dt), Stiffness, Rest Density, and Viscosity.
    - Added a "Reset Simulation" button to re-initialize the domain with new parameters.

## Open Items
- Verify the fix with `demos/mpm-visual.html` (now interactive).
- Audit rigid-body demo bindings if reused; otherwise isolate MLS-MPM module.
- Sketch `Engine` + `RigidDomain` adapter and `MpmDomain` scaffolding; keep headless test harness in scope (mass/momentum checks).
