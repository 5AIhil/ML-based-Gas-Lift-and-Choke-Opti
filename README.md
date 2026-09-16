# Gas Lift and Choke Optimization using Machine Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Domain](https://img.shields.io/badge/Domain-Petroleum%20Engineering-green.svg)]()

A Machine Learning framework designed to model, simulate, and optimize **Gas Lift Systems** and **Choke Valves** in petroleum production engineering. By utilizing physics-informed synthetic data generation, Random Forest Regression, and Gradient Boosting, this project replaces traditional trial-and-error approaches with real-time, data-driven operational decision-making.

---

## 📌 Executive Summary & Industry Context

In offshore and onshore oil & gas field operations:
* **Gas Lift Systems** inject high-pressure gas into the well tubing to reduce fluid mixture density and lower hydrostatic pressure, allowing reservoir pressure to lift fluids to the surface.
  * *The Optimization Challenge*: Under-injecting gas yields insufficient lift, while over-injecting gas causes fluid turbulence, excessive back-pressure, and wasted gas capacity. An optimal injection point must be continuously maintained.
* **Choke Valves** control wellhead pressure and flow rates measured in choke bean sizes ($1/64$ inches).
  * *The Optimization Challenge*: Field operators frequently adjust chokes based on static nomographs or dynamic field intuition. ML models can accurately capture non-linear multiphase flow under critical and sub-critical conditions.

### 📈 Industry Benchmarks
* **North Sea / Forties Field**: ML-driven production optimization achieved a **15% production increase** and **\$2M annual savings**.
* **Chevron**: Reported **15–20% production gains** on optimized gas lift wells.
* **Shell**: Saved over **\$1M annually** on a single platform using automated choke flow optimization.

---

## 🚀 Key Features

* **Physics-Informed Data Generation**: Generates synthetic datasets incorporating hydrostatic pressure differentials, fluid properties (GLR, viscosity, density), water cut penalties, and choke critical flow equations.
* **Gas Lift Regressor (`RandomForestRegressor`)**: Predicts oil production (`bbl/day`) based on gas injection rate, wellhead pressure, reservoir pressure, GOR, water cut, tubing size, and well depth. Uses 5-fold cross-validated `GridSearchCV`.
* **Choke Flow Regressor (`GradientBoostingRegressor`)**: Models multiphase fluid flow (`bbl/day`) through choke beans ($8/64"$ to $64/64"$) considering pressure drop, temperature, fluid density, and viscosity.
* **Automated Optimization Engine**: Evaluates well operating envelopes to pinpoint the exact gas injection rate or choke size that maximizes production.
* **Artifact Persistence**: Automatically exports datasets (CSV/Pickle), trained model binaries (`.pkl`), hyperparameter JSON metadata, and comprehensive project summaries (`project_summary.json`).
* **Visual Diagnostics**: Generates optimization curves with peak annotations and feature importance charts.

---

## 🛠 Project Architecture & Data Schema

### 1. Gas Lift Optimization Model
* **Algorithm**: `RandomForestRegressor` (Hyperparameter tuned via `GridSearchCV`)
* **Target Output**: `oil_production` ($bbl/day$)

| Feature | Description | Units | Range |
| :--- | :--- | :--- | :--- |
| `gas_injection_rate` | Lift gas injection rate | MMscf/day | $0.5 - 5.0$ |
| `wellhead_pressure` | Wellhead tubing pressure | psi | $100 - 800$ |
| `reservoir_pressure` | Static reservoir pressure | psi | $1000 - 4000$ |
| `water_cut` | Water volume fraction | % | $0 - 95$ |
| `gor` | Gas-Oil Ratio | scf/bbl | $200 - 2000$ |
| `tubing_diameter` | Internal tubing diameter | inches | $2.375, 2.875, 3.5, 4.5$ |
| `well_depth` | Total vertical depth | ft | $5000 - 15000$ |

### 2. Choke Valve Optimization Model
* **Algorithm**: `GradientBoostingRegressor` (Hyperparameter tuned via `GridSearchCV`)
* **Target Output**: `flow_rate` ($bbl/day$)

| Feature | Description | Units | Range |
| :--- | :--- | :--- | :--- |
| `choke_size` | Choke orifice size | 1/64 inches | $8 - 64$ |
| `upstream_pressure` | Pressure before choke | psi | $500 - 3000$ |
| `downstream_pressure` | Pressure after choke (line pressure) | psi | $500 - 500$ |
| `glr` | Gas-Liquid Ratio | scf/bbl | $100 - 2000$ |
| `fluid_density` | Fluid mixture density | lb/ft³ | $30 - 60$ |
| `temperature` | Flowline temperature | °F | $100 - 250$ |
| `viscosity` | Oil viscosity | cp | $0.5 - 10$ |

---

## 📂 Repository Structure

```
├── Gas_Lift_and_Choke_Optimization_.ipynb   # Main Jupyter Notebook containing complete pipeline code
├── choke_data_20250705_234920.csv             # Sample generated dataset for Choke optimization
├── gas_lift_data_20250705_234920.csv          # Sample generated dataset for Gas Lift optimization
├── LICENSE                                    # MIT License
└── README.md                                  # Documentation
```

---

## 💻 Quickstart & Usage

### 1. Prerequisites
Ensure Python 3.8+ is installed. Install required packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 2. Python Usage Example

```python
from Gas_Lift_and_Choke_Optimization_ import GasLiftChokeOptimizer

# Initialize optimizer instance
optimizer = GasLiftChokeOptimizer(save_dir='petroleum_optimization_data')

# 1. Generate or Load Synthetic Training Data
gas_lift_df = optimizer.generate_gas_lift_data(n_samples=1500)
choke_df = optimizer.generate_choke_data(n_samples=1500)

# 2. Train and Evaluate ML Models
gas_results = optimizer.train_gas_lift_model()
choke_results = optimizer.train_choke_model()

# 3. Optimize Gas Lift Injection Rate for Specific Well Conditions
well_conditions = {
    'wellhead_pressure': 300,
    'reservoir_pressure': 2500,
    'water_cut': 30,
    'gor': 800,
    'tubing_diameter': 3.5,
    'well_depth': 10000
}
opt_gas = optimizer.optimize_gas_lift(well_conditions)
print(f"Optimal Injection Rate: {opt_gas['optimal_gas_rate']:.2f} MMscf/day")
print(f"Max Predicted Production: {opt_gas['optimal_production']:.2f} bbl/day")

# 4. Optimize Choke Size for Specific Flow Conditions
flow_conditions = {
    'upstream_pressure': 1500,
    'downstream_pressure': 200,
    'glr': 600,
    'fluid_density': 45,
    'temperature': 180,
    'viscosity': 2.5
}
opt_choke = optimizer.optimize_choke(flow_conditions)
print(f"Optimal Choke Size: {opt_choke['optimal_choke_size']:.2f}/64 inches")

# 5. Plot Diagnostics & Feature Importance
optimizer.plot_optimization_results(opt_gas, opt_choke)
optimizer.plot_feature_importance()
```

---

## 📄 License

This project is open-source software licensed under the **[MIT License](file:///Users/sahilchoudhary/Desktop/Projects/Gas%20lift%20and%20Choke%20Optimization%20%28ML%29/LICENSE)**.

```
MIT License

Copyright (c) 2026 Sahil Choudhary

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
...
```

See the full [LICENSE](file:///Users/sahilchoudhary/Desktop/Projects/Gas%20lift%20and%20Choke%20Optimization%20%28ML%29/LICENSE) file for details.
