llm instructions for this file:
This file contains a high level plan for the project. with development phases and feature / architecture implementation goals for each phase. critical path etc. 

## Project Goal
- Build a WebGPU materials simulation engine that eventually covers solids/fluids/gases/plasmas with thermodynamics, basic chemistry, and EM fields, starting from MLS-MPM water and integrating with or replacing the current toychest-style sim.  This simulation engine should also be able to support astrophysics like n-body gravity simulation black hole and star formation and interactions between all of these items. these different simulation domains should be implemented in modular webgpu compute pipelines with varying timesteps depending on practicality. these modular domains should all be able to operate on the same underlying datastructure. 

this library should be compatible with the peercompute library found in /home/cos/projects/peercompute/peercompute/


## Branch Goal
- Build a modular WebGPU materials simulation engine using MLS-MPM (inspired by WebGPU-Ocean) that can run in/alongside `demos/toychest.html`, starting with water and extending to other materials and water’s three phases.

## High-Level Plan

1) **Architecture & Refactoring (Engine Core)**
   - **Refactor `src/world.js`:** Extract current rigid body logic into a `RigidDomain` class implementing a minimal `DomainInterface`.
   - **Create `Engine` class:** Implement the main loop and `SimulationContext` that manages the WebGPU device and sequences domains.
   - **Outcome:** The existing `toychest` demo runs exactly as before, but driven by `Engine -> RigidDomain`.

2) **Core Infrastructure & Prototyping**
   - **Fixed-Point Atomics Prototype (Critical Path):**
     - Write a standalone WebGPU test to validate `atomicAdd` on `i32` buffers for P2G scattering.
     - Test scaling factors ($2^{16}$ vs $2^{20}$) and check for overflow/precision loss against a CPU reference.
   - **Data Structures:** Implement the `Particle` (128-byte) and `Grid` (16-byte) buffer structures defined in `imp-arch.md`.
   - **Material Registry:** Create the initial struct layouts for material properties.

3) **Water (Single Material) MVP - Headless**
   - **Kernels:** Implement P2G (using fixed-point), Grid Update (Tait EOS for pressure, gravity), and G2P (APIC transfer).
   - **Headless Validation Harness:**
     - Create a Node.js/headless browser script that steps the engine without rendering.
     - **Tests:** Verify Mass Conservation (sum of masses constant), Momentum Conservation (closed box), and Energy Bounds.
   - **Outcome:** A running simulation in the console that numerically behaves like water.

4) **Water Phase Change Support**
   - **Thermodynamics:** Add temperature advection/conduction to the pipelines.
   - **Phase Logic:** Implement simple phase transitions (Liquid <-> Solid <-> Vapor) based on temperature thresholds and latent heat.
   - **Constitutive Switching:** Update Grid kernel to switch stress calculation based on particle phase (Elastic for solid, EOS for fluid).

5) **Additional Interacting Materials**
   - **Multi-Material Support:** Add definition for a solid material (e.g., a "Wood" block) with different density/stiffness.
   - **Interaction:** Verify that the "Wood" block floats on the "Water" particles (buoyancy emerges naturally in MPM).

6) **Demo Integration**
   - **Visualization:** Create a renderer (or adapt `demos/shared/particleRenderer.js`) to read the MPM buffers.
   - **Integration:** Wire the MPM solver into the `toychest` page alongside the rigid body solver.
   - **UI:** Add controls for timestep, reset, and material spawning.

7) **Testing, Profiling, and Stabilization**
   - **Unit Tests:** coverage for math helpers and buffer layouts.
   - **Profiling:** Use WebGPU timestamp queries to optimize P2G/G2P workgroup sizes.
   - **Documentation:** Finalize `imp-log.md` with validation results.
