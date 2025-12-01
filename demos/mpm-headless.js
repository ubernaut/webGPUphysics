import { initWebGPU, mpm } from "../src/index.js";

const statusEl = document.getElementById("status");
const particleCountEl = document.getElementById("particleCount");
const gridSizeEl = document.getElementById("gridSize");
const massEl = document.getElementById("mass");
const massDeltaEl = document.getElementById("massDelta");
const momentumEl = document.getElementById("momentum");
const momentumDeltaEl = document.getElementById("momentumDelta");
const checksEl = document.getElementById("checks");
const toggleBtn = document.getElementById("toggleBtn");
const resetBtn = document.getElementById("resetBtn");
const errorEl = document.getElementById("error");

let sim = null;
let running = true;
let lastCheck = 0;
let checksPerSecond = 0;
let pendingDiag = false;
let baseline = null;
const MASS_TOL = 1e-2;
const MOM_TOL = 1e-2;
const constants = {
  stiffness: 2.5,
  restDensity: 4.0,
  dynamicViscosity: 0.08,
  dt: 0.05,
  fixedPointScale: 1e7
};

function setStatus(text) {
  statusEl.textContent = text;
}

function setError(err) {
  console.error(err);
  errorEl.textContent = err?.stack || String(err);
  setStatus("error");
  running = false;
  toggleBtn.disabled = true;
}

toggleBtn.addEventListener("click", () => {
  running = !running;
  toggleBtn.textContent = running ? "Pause" : "Resume";
  setStatus(running ? "running" : "paused");
});

resetBtn.addEventListener("click", () => {
  baseline = null;
  statusEl.textContent = "baseline reset";
});

async function setup() {
  try {
    setStatus("initializing WebGPU…");
    const { device } = await initWebGPU();

    const particleCount = 4000;
    const gridSize = { x: 32, y: 32, z: 32 };

    setStatus("creating MLS-MPM…");
    const blockOptions = { start: [2, 2, 2], gridSize, jitter: 0.5, spacing: 0.65 };
    sim = mpm.createHeadlessMpm({
      device,
      particleCount,
      gridSize,
      iterations: 1,
      blockOptions,
      constants
    });

    particleCountEl.textContent = particleCount.toString();
    gridSizeEl.textContent = `${gridSize.x}×${gridSize.y}×${gridSize.z}`;
    setStatus("running");

    function loop(ts) {
      if (running && sim) {
        sim.step();
      }
      // Diagnostics every 500 ms (avoid overlap)
      if (sim && !pendingDiag && (ts - lastCheck) > 500) {
        pendingDiag = true;
        lastCheck = ts;
        mpm.computeMassMomentum(device, sim.buffers.particleBuffer, particleCount)
          .then(({ mass, momentum }) => {
            checksPerSecond = checksPerSecond * 0.7 + 0.3 * 2; // ~2 per second target
            massEl.textContent = mass.toFixed(4);
            momentumEl.textContent = momentum.map((v) => v.toFixed(4)).join(", ");
            if (!baseline) {
              baseline = { mass, momentum };
            }
            const massDelta = baseline ? mass - baseline.mass : 0;
            const momDelta = baseline
              ? momentum.map((v, i) => v - baseline.momentum[i])
              : [0, 0, 0];
            massDeltaEl.textContent = massDelta.toExponential(3);
            momentumDeltaEl.textContent = momDelta.map((v) => v.toExponential(3)).join(", ");
            // Flag drift visually
            const momMag = Math.hypot(momDelta[0], momDelta[1], momDelta[2]);
            if (Math.abs(massDelta) > MASS_TOL || momMag > MOM_TOL) {
              statusEl.textContent = `drift detected (Δm=${massDelta.toExponential(2)}, |Δp|=${momMag.toExponential(2)})`;
            } else if (running) {
              statusEl.textContent = "running";
            }
            checksEl.textContent = checksPerSecond.toFixed(1);
          })
          .catch(setError)
          .finally(() => { pendingDiag = false; });
      }
      requestAnimationFrame(loop);
    }
    requestAnimationFrame(loop);
  } catch (err) {
    setError(err);
  }
}

setup();
