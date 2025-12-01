# Implementation Log

## [2025-11-30] Fixed WebGPU Validation Errors and Phase Instability
- **Bug Fix:** Located and removed duplicated code in `src/domains/mpm/shaders.js` that caused WGSL parsing errors (`expected ';'`).
- **Headless Testing:** Established a protocol for headless testing using `demos/mpm-headless.html` with Puppeteer automation (checking DOM elements for mass/momentum delta).
- **Anti-Gravity Fix:** Reduced `fixedPointScale` from `1e7` to `1e5` to prevent integer overflow in atomic additions, which was causing massive velocity spikes ("anti-gravity" effect).
- **Sub-Stepping:** Implemented physics sub-stepping in `demos/mpm-visual.js` and `demos/mpm-headless.js`. The simulation now runs fixed `0.005s` physics steps, iterating multiple times per frame to match the user's requested `dt`. This decouples stability from playback speed.
- **Solid Phase Tuning:**
    - Initially reduced Solid stiffness multiplier to `1.0` to stabilize explicit integration at high `dt`.
    - However, this proved too soft (`E=2.5`), causing the block to collapse ("shrink infinitely") under gravity ($\rho g H \approx 76 > E$).
    - **Correction:** Restored/Increased Solid stiffness multiplier to `100.0` (effective $E=250$). With sub-stepping (`dt=0.005`), this is stable ($\Delta t_{crit} \approx 0.12 > 0.005$) and strong enough to support the block's weight.
    - Added a safety clamp `max(J, 0.2)` in `P2G2_WGSL` to prevent singularity/infinite force if particles are forced into extreme compression (e.g., initial overlap).
- **Gas Phase Tuning:** Increased Gas pressure multiplier to `5.0` to ensure visual expansion against gravity.
- **Reference Material:** Verified architecture against "Incremental MPM" guide. Confirmed validity of Hybrid Lagrangian-Eulerian approach and density-based volume evolution for fluids.

## [2025-12-01] Multi-Material Constitutive Framework Implementation

### Problem Statement
Ice/solid behavior was fundamentally broken when using Neo-Hookean elasticity within MLS-MPM. The council discussion identified that Neo-Hookean is designed for rubber-like materials with large elastic deformation, not brittle crystalline solids like ice that deform minimally before fracturing.

### Solution: Unified Multi-Material Architecture

Implemented a constitutive model dispatch system where all materials use the same MPM simulation loop, but stress computation is dispatched based on `materialType`.

### Key Changes

**1. Extended Particle Buffer (schema.js)** - 160 bytes per particle
- Added `materialType` (u32): BRITTLE_SOLID, ELASTIC_SOLID, LIQUID, GAS, GRANULAR
- Added `damage` (f32): Fracture state [0,1] for brittle materials
- Added per-particle `mu`, `lambda`: Material properties that can vary per particle
- Renamed `volume` to `volume0` for clarity (reference configuration volume)
- Added `MATERIAL_PRESETS` for ice, water, steam, rubber with sensible defaults
- Renamed `DEFAULT_WATER_CONSTANTS` to `DEFAULT_SIMULATION_CONSTANTS` with tensileStrength and damageRate

**2. Constitutive Model Dispatch (shaders.js)**
- `BRITTLE_SOLID`: Linear elastic stress (σ = λ·tr(ε)·I + 2μ·ε) with principal stress fracture criterion
  - Added `eigenvalues_symmetric()` function for computing principal stresses
  - Damage accumulates when max principal stress > tensile strength
  - Effective stiffness reduced: `effective_mu = mu * (1 - damage)`
- `ELASTIC_SOLID`: Neo-Hookean (unchanged from before)
- `LIQUID`: Tait EOS + viscosity (unchanged)
- `GAS`: Ideal gas law (P ∝ ρT) (unchanged)
- `GRANULAR`: Placeholder for future Drucker-Prager implementation

**3. Phase Transitions (G2P shader)**
- Fully damaged ice (damage >= 1.0) converts to LIQUID with F reset to Identity
- Temperature-based phase transitions with hysteresis to prevent oscillation

**4. Initialization (init.js)**
- Updated `createBlockParticleData()` to accept materialType, mu, lambda
- Added helper functions: `createIceBlockData()`, `createWaterBlockData()`, etc.
- Fixed matrix padding for GPU alignment (12 floats per mat3x3f)

**5. Demo (mpm-visual.js)**
- Added material type dropdown selector (Ice, Rubber, Water, Steam)
- Added "Solid Properties" folder with sliders for μ, λ, tensile strength, damage rate
- Auto-sets temperature and material properties when material type changes

### Design Decisions

1. **Linear Elastic for Ice (not Neo-Hookean)**: Ice deforms minimally before fracture. Linear elasticity is valid for small strains and computationally simpler than Neo-Hookean.

2. **Per-Particle Material Properties**: Enables material mixing (ice shattering into water), damage evolution per particle, and heterogeneous materials.

3. **Single MPM Loop**: All materials share P2G/G2P transfers and grid operations. Only stress computation differs. Multi-material interaction emerges naturally from shared grid.

### Files Modified
- `src/domains/mpm/schema.js` - Extended buffer, new constants
- `src/domains/mpm/shaders.js` - Constitutive dispatch, fracture criterion
- `src/domains/mpm/init.js` - Material-aware initialization
- `src/domains/mpm/pipelines.js` - New shader constants
- `src/domains/mpm/domain.js` - Fixed import name
- `src/domains/mpm/index.js` - Updated exports
- `demos/mpm-visual.js` - Material selection UI
- `plan/imp-arch.md` - Added constitutive model architecture
- `plan/imp-plan.md` - Revised 10-phase implementation plan
- `plan/llm-council-discussion.md` - Added analysis and resolution

## [2025-12-01] Thermal Diffusion System (Phase 5)

### Problem Statement
Temperature was static after initialization - particles didn't exchange heat with neighbors, and there was no way to add/remove thermal energy (heat sources, cooling boundaries).

### Solution: Grid-Based Thermal Transfer

Implemented heat transfer via the MPM grid, following the same P2G/G2P pattern as momentum transfer.

### Key Changes

**1. Extended Grid Structure (schema.js)** - 32 bytes per cell
- `temperature` (i32 fixed-point): Mass-weighted temperature accumulator
- `thermalMass` (i32 fixed-point): Mass accumulator for temperature averaging
- `heatSource` (i32 fixed-point): External heat flux from interactions
- `THERMAL_CONSTANTS`: Diffusivity, latent heats, specific heat

**2. Temperature Scatter (P2G1 shader)**
- Each particle scatters `mass * temperature` to nearby grid cells
- Uses same quadratic B-spline weights as momentum transfer

**3. Temperature Gather (G2P shader)**
- Particles gather temperature from grid (mass-weighted average)
- Blended with current particle temperature for smoothing

**4. Phase Transitions with Latent Heat (G2P shader)**
- Melting (Ice→Water): Absorbs latent heat, delays temperature rise
- Freezing (Water→Ice): Releases latent heat, delays temperature drop
- Boiling (Water→Steam): High latent heat barrier
- Condensing (Steam→Water): Same in reverse
- Material type and properties update automatically on phase change

**5. Heat Source Interaction (G2P shader)**
- Interaction sphere can now have a temperature
- Particles within sphere are heated/cooled toward sphere temperature
- `mouse.temperature` field in MouseInteraction struct

**6. Demo Updates (mpm-visual.js)**
- "Heat Source (K)" slider in Interaction Sphere folder
- Set to 0 for no thermal effect, or 400K to heat, 200K to cool

### Constants Added
- `thermalDiffusivity: 0.05` in DEFAULT_SIMULATION_CONSTANTS
- Latent heat constants (scaled for real-time simulation)
- Phase transition temperatures with hysteresis

### Files Modified
- `src/domains/mpm/schema.js` - Extended grid to 32 bytes, thermal constants
- `src/domains/mpm/shaders.js` - All shaders updated for new grid, temperature transfer, latent heat
- `src/domains/mpm/pipelines.js` - thermal_diffusivity constant
- `src/domains/mpm/index.js` - Export THERMAL_CONSTANTS and DIFFUSE_TEMPERATURE_WGSL
- `demos/mpm-visual.js` - Heat source temperature control

### Usage
1. Start simulation with Ice (260K)
2. Enable interaction sphere with Heat Source = 400K
3. Move sphere through particles
4. Observe particles warming (blue→red) and melting (solid→liquid)
5. Reduce Heat Source to 200K to freeze them back
