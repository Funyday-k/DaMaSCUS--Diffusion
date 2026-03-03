# DaMaSCUS--Diffusion
This is a project that employs diffusion models to replace Monte Carlo methods for simulating the propagation of dark matter particles within the Sun.

This our project tree

```
DaMaSCUS-Diffusion/
├── data/                        # Raw trajectory data from DaMaSCUS-SUN simulations
│   └── results_*/*.txt          # Trajectory files (trajectory_*_task*.txt)
├── data_pipeline/               # 【Data Preprocessing Module】
│   ├── parser.py                # Parses txt files, extracts scattering transitions → .npz
│   └── transform.py             # Spherical coordinate transform + Z-score normalization
├── sde_physics/                 # 【Physics Prior Module】(TBD)
├── models/                      # 【Neural Network Module】(TBD)
├── training/                    # 【Training Module】
│   ├── mlp_score.py             # Conditional Score Network (residual MLP + time embedding)
│   └── train.py                 # Diffusion model training engine (VP-SDE, cosine schedule)
├── inference/                   # 【Generation and Evaluation Module】
│   └── sampler.py               # Reverse SDE sampler (TBD)
├── .gitignore
└── README.md
```