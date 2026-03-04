# DaMaSCUS--Diffusion

**Replacing Monte Carlo with Diffusion Models for Dark Matter Simulation in the Sun**

A conditional diffusion model that learns the scattering kernel of dark matter (DM) particles propagating through the Sun, trained on [DaMaSCUS-SUN](https://github.com/temken/DaMaSCUS-SUN) Monte Carlo trajectory data. Once trained, the model can generate post-scattering states orders of magnitude faster than traditional MC sampling, while preserving the exact gravitational dynamics and scattering rates from the Standard Solar Model (AGSS09).

---

## Physics Background

When dark matter particles from the galactic halo enter the Sun, they undergo repeated scatterings with solar nuclei (H, He, O, Fe, …). Between scatterings, particles follow deterministic orbits in the solar gravitational potential $\Phi(r)$. Each scattering event changes the particle's velocity and energy:

$$(\mathbf{r}, \mathbf{v}, E)_\text{before} \xrightarrow{\text{scatter}} (\mathbf{r}, \mathbf{v}', E')_\text{after}$$

**DaMaSCUS-SUN** computes these transitions via Monte Carlo sampling of the scattering kinematics. **This project** replaces that MC step with a trained conditional diffusion model:

| Component | DaMaSCUS-SUN (MC) | This Project (Diffusion) |
|-----------|-------------------|--------------------------|
| Gravitational propagation | Analytic orbit integration | Same (RK45 in solar potential) |
| Scattering rate $\Gamma(r)$ | $\sum_i n_i(r)\,\sigma_i\,v$ | Same (from AGSS09 model) |
| **Scattering outcome** | **MC sampling of kinematics** | **Conditional diffusion model** |
| Speed per trajectory | ~seconds | ~milliseconds (after training) |

The **scattering rate** correctly accounts for coherent spin-independent (SI) enhancement over all solar nuclei:

$$\Gamma(r) = \sum_i n_i(r)\,\sigma_i\,v, \quad \sigma_i = \sigma_p A_i^2 \left(\frac{\mu_i}{\mu_p}\right)^2$$

where $A_i$ is the mass number, $\mu_i$ and $\mu_p$ are the reduced masses for DM-nucleus and DM-proton scattering respectively.

---

## Architecture

The core model is a **FiLM-conditioned Score Network** that predicts noise $\epsilon$ given:
- $\mathbf{x}_t$: noised post-scattering state
- $t$: diffusion timestep
- $\mathbf{c}$: pre-scattering condition $[r, v_\text{rad}, v_\text{tan}, E]$

Each residual layer is modulated by the condition via **Feature-wise Linear Modulation (FiLM)**:

$$\mathbf{h}' = \gamma(\mathbf{c}, t) \odot \text{LayerNorm}(\mathbf{h}) + \beta(\mathbf{c}, t)$$

Key specs:

| Parameter | Value |
|-----------|-------|
| Noise schedule | VP-SDE with cosine schedule |
| Hidden dim | 256 |
| FiLM residual blocks | 6 |
| Parameters | ~1.8 M |
| EMA decay | 0.9999 |
| Sampling | DDIM (50 steps, default) / DDPM (200 steps) |

---

## Project Structure

```
DaMaSCUS--Diffusion/
│
├── data/                           # Raw input data (gitignored)
│   ├── model_agss09.dat            # AGSS09 Standard Solar Model (Serenelli+2009)
│   └── results_<m>_<sigma>/        # DaMaSCUS-SUN trajectory files (*.txt)
│
├── data_pipeline/                  # Data preprocessing
│   ├── parser.py                   # Multi-process parser: txt → parsed_transitions.npz
│   └── transform.py                # Cartesian→spherical + Z-score normalisation (PyTorch Dataset)
│
├── sde_physics/                    # Physics core (no ML)
│   ├── solar_model.py              # AGSS09 loader: T(r), ρ(r), Φ(r), n_i(r), v_esc(r)
│   ├── free_streaming.py           # RK45 orbit integrator in solar gravitational potential
│   └── scattering_rate.py          # SI cross-section with A² coherence enhancement
│
├── training/                       # Model training
│   ├── mlp_score.py                # FiLM Conditional Score Network definition
│   └── train.py                    # VP-SDE trainer with EMA + cosine LR scheduling
│
├── inference/                      # Inference, evaluation & validation
│   ├── sampler.py                  # DDIM / DDPM reverse-diffusion sampler
│   ├── evaluate.py                 # Single-step quality: W₁ distance, constraint checks
│   ├── trajectory_simulator.py     # Hybrid physics/ML end-to-end simulator
│   └── trajectory_validator.py     # End-to-end trajectory validation vs. MC ground truth
│
├── checkpoints/                    # Model weight files (gitignored, *.pth)
│   └── damascus_diffusion_ep{N}.pth
│
├── outputs/                        # All generated outputs (plots & results)
│   ├── evaluation_results.png      # Single-step W₁ & distribution comparison
│   ├── trajectory_validation.png   # End-to-end trajectory comparison
│   └── solar_model_profiles.png    # AGSS09 radial profile plots
│
├── parsed_transitions.npz          # Parsed scattering events (gitignored, ~400 MB)
├── .gitignore
└── README.md
```

---

## Installation

```bash
# Python 3.10+ required
pip install torch numpy pandas scipy tqdm matplotlib
```

Supported devices: **CUDA** → **MPS (Apple Silicon)** → **CPU** (auto-detected).

---

## Step-by-Step Workflow

### Step 1 — Parse raw MC trajectory data

```bash
python data_pipeline/parser.py
```

Recursively scans all `data/results_*/trajectory_*.txt` files, identifies scattering events via energy discontinuities ($|\Delta E| > 10^{-3}$ eV), and extracts (before, after) state pairs.

**Output:** `parsed_transitions.npz` (~400 MB, ≈5.2 M scattering events)

Each event stores $[r, v_x, v_y, v_z, E]$ in Cartesian and spherical representations:

| Feature | Unit | Description |
|---------|------|-------------|
| $r$ | km | Radial distance from solar centre |
| $v_\text{rad}$ | km/s | Radial velocity (negative = inward) |
| $v_\text{tan}$ | km/s | Tangential speed $\geq 0$ |
| $E$ | eV | Total specific energy |

---

### Step 2 — Train the diffusion model

```bash
python training/train.py
```

Trains a FiLM-conditioned VP-SDE score network for 300 epochs using AdamW + cosine LR annealing.

| Option | Default | Description |
|--------|---------|-------------|
| epochs | 300 | Training epochs (FiLM needs ≥300 to converge) |
| batch_size | 8192 | Samples per batch |
| lr | 1e-3 | Initial learning rate |
| EMA decay | 0.9999 | Exponential moving average for inference weights |

**Output:** `checkpoints/damascus_diffusion_ep{10,20,…,300}.pth`

Each checkpoint contains:
```python
{
  "epoch":            int,
  "model_state_dict": dict,   # raw training weights
  "ema_state_dict":   dict,   # EMA weights (use for inference)
  "optimizer_state":  dict,
}
```

> **Tip:** Resume from a checkpoint by modifying the `__main__` block in `train.py` to load `ema_state_dict` before calling `trainer.train()`.

---

### Step 3 — Evaluate single-step scattering quality

```bash
python inference/evaluate.py [--n_samples 10000] [--method ddim] [--num_steps 50]
```

Draws `n_samples` real MC conditions, generates matching outputs with the diffusion model, and computes:

- **Wasserstein-1 distance** $W_1/\sigma$ for each of $[r, v_\text{rad}, v_\text{tan}, E]$
- Per-sample relative error (mean & median)
- Physical constraint violation rates ($v_\text{tan} < 0$, $\Delta r/r > 10\%$, …)
- Marginal distribution histograms (MC vs. diffusion)
- Conditional response scatter plots ($r_\text{in} \to r_\text{out}$, $E_\text{in} \to E_\text{out}$)
- Energy-transfer $\Delta E$ distribution

**Output:** `outputs/evaluation_results.png`

Example result at epoch 100 (DDIM, 50 steps, 10 k samples):

| Feature | $W_1/\sigma$ |
|---------|-------------|
| $r$ | 0.0151 |
| $v_\text{rad}$ | 0.0085 |
| $v_\text{tan}$ | 0.0294 |
| $E$ | 0.0128 |

---

### Step 4 — End-to-end trajectory validation

```bash
python inference/trajectory_validator.py [--n_traj 30] [--max_scatter 500] [--max_mc_scatter 1000] [--dt 10]
```

**This is the decisive test**: does the model produce physically consistent full trajectories when used as a drop-in MC replacement?

The validator:
1. Parses `N` real MC trajectories from `data/results_*/`
2. Extracts their initial conditions $[r_0, v_\text{rad,0}, v_\text{tan,0}, E_0]$
3. Runs the exact same conditions through the diffusion-model simulator
4. Compares trajectory-level statistics side-by-side

| Argument | Default | Description |
|----------|---------|-------------|
| `--n_traj` | 30 | Number of trajectories to compare |
| `--max_scatter` | 500 | Max scatterings per diffusion trajectory |
| `--max_mc_scatter` | 1000 | Filter out MC trajectories with more scatterings (avoid very long ones) |
| `--max_time` | 1e6 | Max simulation time per trajectory [s] |
| `--dt` | 10.0 | Free-streaming time step [s] |
| `--checkpoint` | auto (latest) | Path to a specific `.pth` file |
| `--output` | `outputs/trajectory_validation.png` | Output figure path |

**Compared statistics:**

| Metric | Description |
|--------|-------------|
| Outcome distribution | Escaped / Captured / Max reached |
| Scattering count | Distribution of number of scatterings per trajectory |
| Penetration depth | Minimum $r / R_\odot$ reached |
| Duration | Total simulation time |
| Energy change | $\Delta E = E_\text{final} - E_\text{initial}$ |
| Scatter location | Radial distribution of all scattering events |

**Output:** `outputs/trajectory_validation.png`, `outputs/trajectory_validation_results.npz`

---

### Step 5 — Standalone trajectory simulation

```bash
python inference/trajectory_simulator.py
```

Runs 5 test trajectories from the solar surface using the hybrid physics/ML pipeline.
Useful for rapid sanity checks or generating custom trajectory ensembles:

```python
from inference.trajectory_simulator import TrajectorySimulator
from sde_physics.solar_model import R_SUN_KM

sim = TrajectorySimulator(
    model_checkpoint="checkpoints/damascus_diffusion_ep100.pth",
    npz_path="parsed_transitions.npz",
    solar_model_path="data/model_agss09.dat",
    m_chi_GeV=1.0,
    sigma_p_cm2=1e-35,
    dt_step=10.0,
)

# Single trajectory from solar surface
result = sim.simulate_single(
    r_init=R_SUN_KM,
    v_rad_init=-600.0,   # inward [km/s]
    v_tan_init=400.0,    # tangential [km/s]
    E_init=-12000.0,     # [eV]
    max_scatterings=500,
)
print(result['outcome'], result['n_scatter'])

# Batch from Maxwell-Boltzmann initial conditions
results = sim.simulate_batch(n_trajectories=100, seed=42)
```

---

### Step 6 — Inspect the solar model

```bash
python sde_physics/solar_model.py
```

Prints a radial summary table of $T(r)$, $\rho(r)$, $\Phi(r)$, $v_\text{esc}(r)$, $n_H(r)$ and saves profile plots.

**Output:** `outputs/solar_model_profiles.png`

---

## File Conventions

| Path | Tracked in git | Size | Notes |
|------|---------------|------|-------|
| `data/model_agss09.dat` | ✅ yes | ~500 KB | AGSS09 solar model table |
| `data/results_*/` | ❌ no | ~GB | Raw DaMaSCUS-SUN trajectories |
| `parsed_transitions.npz` | ❌ no | ~400 MB | Generated by `data_pipeline/parser.py` |
| `checkpoints/*.pth` | ❌ no | ~30 MB each | Saved by `training/train.py` |
| `outputs/*.png` | ✅ yes | <1 MB each | Evaluation & validation plots |
| `outputs/*.npz` | ❌ no | varies | Numerical results (re-generated on demand) |

---

## References

- **DaMaSCUS-SUN**: T. Emken, [arXiv:1706.02862](https://arxiv.org/abs/1706.02862), [GitHub](https://github.com/temken/DaMaSCUS-SUN)
- **Standard Solar Model AGSS09**: Serenelli, Basu & Ferguson, [arXiv:0909.2668](https://arxiv.org/abs/0909.2668)
- **Score-based Diffusion (VP-SDE)**: Song et al., [arXiv:2011.13456](https://arxiv.org/abs/2011.13456)
- **FiLM Conditioning**: Perez et al., [arXiv:1709.07871](https://arxiv.org/abs/1709.07871)
- **DDIM Sampler**: Song et al., [arXiv:2010.02502](https://arxiv.org/abs/2010.02502)


