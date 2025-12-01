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
