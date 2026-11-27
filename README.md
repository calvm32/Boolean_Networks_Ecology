# Stochastic Boolean Network Model of White-Nose Syndrome in Bat Hibernacula

Computational simulation, parameter estimation, sensitivity analysis, and phase-diagram codebase accompanying our manuscript on White-Nose Syndrome (WNS) dynamics in Midwestern bat hibernacula (submitted to *Ecological Modelling*, November 2026; preprint available soon on arXiv).

## Overview

Since 2006, white-nose syndrome (WNS) has rapidly
spread across the continental United States and
devastated bat populations. Northern long-eared bats,
tricolored bats, and little brown bats have suffered
population declines exceeding 90\% in affected areas.
This repository implements an individual-based stochastic 
Boolean network framework coupling individual bat states 
(hibernating, aroused/active, infected, immune/recovering, 
and cumulative mortality) with seasonal temperature cycles 
and environmental fungal pathogen accumulation. 

Key capabilities in this repository include:
- **Stochastic Boolean Network Simulation:** Daily state transitions incorporating torpor-arousal bouts, seasonal hibernation timing, transmission (both from bat-to-bat and environmental sources), and immunity pathways (via acquired, inherited, and temporary immune responses).
- **MPI-Parallel Particle Swarm Optimization (PSO):** Calibration against multi-year hibernacula survey counts from North and South Midwest sites using a Negative Binomial log-likelihood objective.
- **Global Sensitivity & Scenario Analysis:** Sobol' variance-based sensitivity analysis ($S_1$ and $S_T$), multi-decadal invasion scenarios across environmental fungal decay regimes, and parameter phase diagrams.

## Repository Structure

```
Boolean_networks_ecology/
├── simulate/
│   ├── rules.py             # Boolean state transition rules (environment & individuals)
│   ├── simulate.py          # Core daily time-stepping simulation engine
│   ├── helper_funcs.py      # State initialization, aggregation, and plotting utilities
│   ├── working_params.py    # Baseline biological and regional latitude parameter sets
│   ├── data.py              # Empirical hibernacula survey time-series data
│   ├── compare_regimes/     # Sobol' sensitivity, invasion, and phase diagram scripts
│   └── running_solvers/     # MPI PSO calibration and scripts to run one simulation, average over many, or compare with site data 
├── requirements.txt         # Python dependencies
└── setup.py / pyproject.toml
```

## Installation
Create and activate a virtual environment, then install the package and dependencies:
(Bash)
```
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

## Running Experiments
### Single / Ensemble Simulations: 
Run simulation scripts inside simulate/ using baseline regional parameter configurations (latitude1_NorthMidwest or latitude2_SouthMidwest).

### MPI-Parallel Parameter Calibration (PSO):
(Bash)
```
SITE_NAME=<site_key> SIM_OUTPUT_DIR=./results mpirun -np <num_ranks> python3 <pso_script.py>
```

### Sensitivity & Phase Diagrams: 
Scripts in `simulate/compare_regimes/` generate Sobol' indices, invasion trajectories, and 2D/3D parameter phase diagrams.

## Citation
If you use this code in your research, please cite our manuscript:

```
@unpublished{heldt2026wns,
  author = {Heldt, Calum and ...},
  title  = {A Stochastic Boolean Network Model of White-Nose Syndrome Dynamics in Midwestern Bat Hibernacula},
  note   = {Submitted to Ecological Modelling; preprint on arXiv},
  year   = {2026}
}
```

## License
This project is licensed under the MIT License — see the LICENSE file for details.
