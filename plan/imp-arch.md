llm instructions for this file:
This file contains the architecture plan for the project. update as necessary

# Architecture Design

## High-level structure
- **Core Engine (`Engine` / `SimulationContext`):**
  - Manages WebGPU device/adapter (reusing `src/device.js`).
  - Manages the main timeline/scheduler.
  - Sequences "Domains" (solvers) through a unified step interface.
  - Decoupled from specific physics logic.

- **Domains (Solvers):**
  - Implement a common `DomainInterface`:
    - `createBuffers(device)`
    - `createPipelines(device, sharedCode)`
    - `step(encoder, params)`
    - `getExposedBuffers()`
  - **`domains/rigid`:** Wraps the existing rigid-body logic (currently `src/world.js`).
  - **`domains/mpm`:** New MLS-MPM solver for fluids/deformables.
  - **`domains/astro`:** Future N-body/gravity solvers.
  - **`domains/em`:** Future Electromagnetism solvers.

- **Data Model:**
  - Structure-of-Arrays (SoA) for maximum throughput.
  - 16-byte alignment for all per-particle/per-cell data.
  - Explicit byte offsets defined in schema to ensure interoperability with peercompute.

## Buffer Schema (v1)

### MPM Particle Buffer (128 bytes/particle)
Standardized layout for MLS-MPM particles. 16-byte alignment.

| Field | Type | Offset (floats) | Byte Offset | Notes |
|-------|------|-----------------|-------------|-------|
| `position` | `vec3<f32>` | 0-2 | 0 | + 1 float padding (used for material_id) |
| `material_id`| `f32` | 3 | 12 | (uint cast to float) |
| `velocity` | `vec3<f32>` | 4-6 | 16 | + 1 float padding (used for phase) |
| `phase` | `f32` | 7 | 28 | 0=solid, 1=liquid, 2=gas |
| `mass` | `f32` | 8 | 32 | |
| `volume` | `f32` | 9 | 36 | Computed per-step for fluids |
| `temperature`| `f32` | 10 | 40 | Kelvin |
| `padding` | `f32` | 11 | 44 | Explicit padding for alignment |
| `F` (DefGrad)| `mat3x3<f32>`| 12-20 | 48 | Deformation Gradient (9 floats, row-major) |
| `C` (Affine) | `mat3x3<f32>`| 21-29 | 84 | APIC Affine Matrix (9 floats, row-major) |
| `padding` | `vec2<f32>` | 30-31 | 120 | Align to 128 bytes |

### MPM Grid Buffer (16 bytes/cell)
Dense grid for MLS-MPM.

| Field | Type | Offset (floats) | Byte Offset | Notes |
|-------|------|-----------------|-------------|-------|
| `velocity` | `vec3<f32>` | 0-2 | 0 | Accumulated momentum (during P2G) / Velocity (after update) |
| `mass` | `f32` | 3 | 12 | Accumulated mass |

**Note on P2G Scattering:** WebGPU lacks floating-point atomics.
- **Strategy:** Use `atomicAdd` on `i32` buffers.
- **Scaling:** Float values are scaled by a fixed factor (e.g., $2^{16}$ or $2^{20}$) and cast to integer.
- **Reconstruction:** During grid update, read `i32`, cast to float, and multiply by inverse scale factor.
- **Baseline (WebGPU-Ocean):** Uses 80-byte AoS particle structs (pos, vel, C), 16-byte cells (vx, vy, vz, mass), fixed-point multiplier 1e7, workgroup size 64, dt=0.2, stiffness=3, rest_density=4, dynamic_viscosity=0.1, grid up to 64^3.

### Material Registry
- Structs defining physical properties per material ID:
  - Density, stiffness (E/Bulk), Poisson's ratio.
  - Viscosity, drag.
  - Thermal conductivity, heat capacity.
  - Phase transition temperatures and latent heat.

## Compute Pipelines (MPM Sequence)

1.  **Clear Grid:** Zero out mass/momentum grid buffers.
2.  **P2G (Particles to Grid):**
    - Scatter mass and momentum ($v + C \cdot dist$) to 3x3x3 neighboring nodes.
    - Use quadratic B-spline weights.
    - **Critical:** Use fixed-point atomics for scattering.
3.  **Grid Update (Forces & BCs):**
    - Convert fixed-point grid data back to float.
    - Compute grid velocities ($v = p / m$).
    - Apply external forces (gravity).
    - Apply constitutive model (Stress $\to$ Force).
        - Fluids: Tait EOS (Pressure only).
        - Solids: Neo-Hookean or similar.
    - Apply boundary conditions (collide with box/shapes).
4.  **G2P (Grid to Particles):**
    - Gather velocity from grid (FLIP/PIC/APIC blend).
    - Update deformation gradient ($F$) and affine matrix ($C$).
    - Advect particles ($x += v \cdot dt$).
5.  **Phase Update (Thermodynamics):**
    - Update temperature (conduction/advection).
    - Check phase transition thresholds -> update phase ID and material params.

## Integration strategy (near term)
- Wrap existing rigid-body `World` into a `RigidDomain` (thin adapter) to avoid churn.
- Introduce a simple `Engine` that sequences domains (Rigid + MPM) in order; upgrade to a timeline scheduler later.
- Keep MLS-MPM core headless-testable; demos consume exposed buffers. Start with side-by-side rendering in toychest (rigid + MPM) via shared adapter.

## Scheduling and Timesteps
- **Master Timeline:** Controls global time.
- **Substepping:** MPM requires small $dt$ (CFL condition). Rigid bodies can run larger $dt$.
- **Engine Loop:**
  ```javascript
  while (lag >= dt_min) {
     for (domain of domains) {
       domain.step(encoder, dt_min);
     }
     lag -= dt_min;
  }
  ```

## Peercompute Compatibility
- **Contract:**
  - Buffers are raw `ArrayBuffer`s.
  - Layouts are strictly defined (as above).
  - No dependence on runtime object wrappers; data is "Just Bytes".
- **Interop:**
  - Allow passing `GPUBuffer` handles or mapped `ArrayBuffer`s to peercompute nodes.
  - Metadata (JSON) describes the schema version and active particle count.

## Rendering/Demo Integration
- **Headless First:** Core logic depends only on `src/device.js` and WGSL.
- **Visualization:** `demos/toychest` or similar reads GPU buffers (copy to CPU or direct vertex buffer binding) to render sprites/meshes.
