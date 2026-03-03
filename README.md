# DaMaSCUS--Diffusion
This is a project that employs diffusion models to replace Monte Carlo methods for simulating the propagation of dark matter particles within the Sun.

This our project tree

DaMaSCUS-Diffusion/
├── data/                    # Stores the large number of trajectory_*.txt files you generate
├── 1_data_pipeline/         # 【Data Preprocessing Module】
│   ├── parser.py            # Reads txt files and extracts key collision states
│   └── transform.py         # Coordinate system transformation and normalization (critical)
├── 2_sde_physics/           # 【Physics Prior Module】
│   └── langevin.py          # Defines drift and diffusion terms for forward SDE
├── 3_models/                # 【Neural Network Module】
│   ├── mlp_score.py         # Residual MLP network fitting the Score function
│   └── time_embedding.py    # Sine position encoding for time step t
├── 4_training/              # 【Training Module】
│   ├── loss.py              # Denoising Score Matching loss function
│   └── train.py             # Main script for distributed/single-GPU training
└── 5_inference/             # [Generation and Evaluation Module]
    ├── sampler.py           # Inverse SDE solver (Euler-Maruyama, etc.)
    └── evaluate.py          # Compare generated escape spectra with your MC data