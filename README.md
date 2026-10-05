# Batch Cooling Crystallization of Plate-Like Crystals: Method of Moments Reproduction

**Master Training Project — Chemical Engineering Department, Indian Institute of Technology Kharagpur**  
**Author:** Devan Singh Faujdar  

---

## 1. Executive Summary & Objective

This repository contains the complete first-principles reproduction of the research benchmark:

> **Botond Szilágyi and Béla G. Lakatos (2015)**  
> *"Batch Cooling Crystallization of Plate-like Crystals: A Simulation Study"*  
> *Periodica Polytechnica Chemical Engineering*, 59(2), pp. 151–158.  
> [DOI: 10.3311/PPch.7581](https://doi.org/10.3311/PPch.7581)

The objective is to simulate and reproduce the dynamic two-dimensional (2D) size and morphology evolution of plate-like crystals during seeded batch cooling crystallization **exclusively using the Method of Moments (Quadrature Method of Moments — QMOM)** without spatial grid discretization.

All published figures (**Figures 1, 3, 6, 8, 9, 10, and 11**) are successfully reproduced with high physical fidelity.

---

## 2. Mathematical Modeling & Governing Equations

### 2.1 Crystal Geometry & 2D Population Balance Equation (PBE)
Plate-like crystals have two dominant growing dimensions: **Length ($L_1$)** and **Width ($L_2$)**, while crystal thickness remains constant ($L_3 = k_V = 5 \times 10^{-6}\text{ m}$). The individual crystal volume is:
$$v_p = k_V L_1 L_2$$

The population density function $n(L_1, L_2, t)$ satisfies the 2D hyperbolic population balance equation:
$$\frac{\partial n}{\partial t} + \frac{\partial (G_1 n)}{\partial L_1} + \frac{\partial (G_2 n)}{\partial L_2} = 0, \quad \forall L_1, L_2 > 0$$

Boundary influx condition at nucleus birth size ($L_1, L_2 \to 0$):
$$\lim_{L_1, L_2 \to 0} \left[ G_1 G_2 n(L_1, L_2, t) \right] = B(t) \cdot \delta(L_1 - L_{10}) \cdot \delta(L_2 - L_{20})$$

### 2.2 Why the Method of Moments (QMOM)?
Direct discretization of the 2D PDE over a spatial grid $(L_1, L_2)$ requires thousands of coupled ODEs and introduces artificial numerical diffusion. Instead, taking bivariate cross-moments:
$$\mu_{k,m}(t) \equiv \int_0^\infty \int_0^\infty L_1^k L_2^m n(L_1, L_2, t) \, dL_1 \, dL_2$$

Using the **Quadrature Method of Moments (QMOM)** with $N_q = 3$ Dirac delta nodes:
$$n(L_1, L_2, t) \approx \sum_{i=1}^3 w_i(t) \cdot \delta(L_1 - L_{1i}(t)) \cdot \delta(L_2 - L_{2i}(t))$$
$$\mu_{k,m}(t) \cong \sum_{i=1}^3 w_i(t) L_{1i}^k(t) L_{2i}^m(t)$$

Applying the moment transformation and integration by parts reduces the system to **9 coupled stiff ODEs**:
$$\frac{d\mu_{00}}{dt} = B(\sigma, \epsilon)$$
$$\frac{d\mu_{10}}{dt} = \sum_{i=1}^3 w_i G_1(\sigma, L_{1i})$$
$$\frac{d\mu_{01}}{dt} = \sum_{i=1}^3 w_i G_2(\sigma, L_{2i})$$
$$\frac{d\mu_{11}}{dt} = \sum_{i=1}^3 w_i \left[ L_{2i} G_1(\sigma, L_{1i}) + L_{1i} G_2(\sigma, L_{2i}) \right]$$

### 2.3 Thermodynamics & Kinetics
* **Modified Apelblat Solubility Model** ($T$ in $^\circ\text{C}$):
  $$c_s(T) = 1000 \cdot \exp\left( a_1 + \frac{a_2}{T} + a_3 \ln(T) \right)$$
  where $a_1 = -69.51, a_2 = 368.6, a_3 = 16.18$.
* **Relative Supersaturation:**
  $$\sigma(t) = \max\left(0, \frac{c(t) - c_s(T(t))}{c_s(T(t))}\right)$$
* **Face-Specific Size-Dependent Growth Rates:**
  $$G_1(\sigma, L_1) = k_1 \sigma^{g_1} \left( 1 + \gamma_1 L_{1,\mu\text{m}}^{\alpha_1} \right), \quad g_1 = 1.5, \, \gamma_1 = 0.05, \, \alpha_1 = 0.8$$
  $$G_2(\sigma, L_2) = k_2 \sigma^{g_2} \left( 1 + \gamma_2 L_{2,\mu\text{m}}^{\alpha_2} \right), \quad g_2 = 1.7, \, \gamma_2 = 0.03, \, \alpha_2 = 0.9$$
* **Secondary Contact Nucleation Rate:**
  $$B(\sigma, \epsilon) = k_S \cdot \epsilon \cdot \mu_{11} \cdot \sigma^{b_1}, \quad k_S = 3.6 \times 10^5\text{ \#/(m}^3\cdot\text{s (W/kg))}, \, b_1 = 2.0$$
* **Solute Mass Balance:**
  $$\frac{dc}{dt} = -\rho_c R_V = -\rho_c k_V \frac{d\mu_{11}}{dt}, \quad \rho_c = 1665\text{ kg/m}^3$$
* **Output Morphological Properties:**
  - Mean Crystal Length: $\langle L_1 \rangle = \mu_{10} / \mu_{00}$
  - Mean Crystal Width: $\langle L_2 \rangle = \mu_{01} / \mu_{00}$
  - Crystal Aspect Ratio: $AR = \langle L_1 \rangle / \langle L_2 \rangle$

---

## 3. Initial Seed Nodes & Weights (Table 2 in Paper)

From Table 2 of Szilágyi & Lakatos (2015), the initial seed population is represented by $3$ quadrature nodes:

| Node | Weight $w_i$ [#/m³] | Length $L_{1i}$ [m] | Width $L_{2i}$ [m] |
|:---:|:---:|:---:|:---:|
| **1** | $1.86 \times 10^{10}$ | $9.74 \times 10^{-5}$ | $6.85 \times 10^{-5}$ |
| **2** | $6.46 \times 10^{10}$ | $9.80 \times 10^{-5}$ | $5.74 \times 10^{-5}$ |
| **3** | $1.281 \times 10^{11}$ | $1.138 \times 10^{-4}$ | $6.05 \times 10^{-5}$ |

Initial population moments:
$$\mu_{00}(0) = 2.113 \times 10^{11}\text{ \#/m}^3 \quad (\text{Total Seed Crystals})$$
$$\langle L_1 \rangle_0 = 107.5\,\mu\text{m}, \quad \langle L_2 \rangle_0 = 60.3\,\mu\text{m}, \quad AR_0 = 1.78$$

---

## 4. The 11 Differential-Algebraic Equations (QMOM)

The exact 11 ODE system formulation matching Szilágyi & Lakatos (2015) is documented in [`THE_11_ODE_EQUATIONS_QMOM.md`](file:///Users/devansinghfaujdar/Documents/Master_Training_Project_iitkgp/THE_11_ODE_EQUATIONS_QMOM.md):
- **ODEs 1–3**: Length growth for 3 nodes: $dL_{1,i}/dt = G_1(\sigma, L_{1,i})$
- **ODEs 4–6**: Width growth for 3 nodes: $dL_{2,i}/dt = G_2(\sigma, L_{2,i})$
- **ODEs 7–9**: Dynamic weights linear system: $\mathbf{M} \mathbf{\dot{w}} = [B, 0, 0]^T$
- **ODE 10**: Solute mass balance: $dc/dt = -\rho_c R_V$
- **ODE 11**: Temperature cooling schedule: $dT/dt = -cr$

---

## 5. Clean Repository Layout

```
├── THE_11_ODE_EQUATIONS_QMOM.md             # Complete mathematical documentation of the 11 ODEs
├── simulate_all_cases_matlab.m              # Master MATLAB script solving 11 ODEs via ode15s (BDF)
├── generate_simulation_data.py              # Python runner solving identical 11 ODEs via SciPy BDF
├── reproduce_paper_results.py               # Visualizer loading CSV data to generate Figs 1, 3-11
├── config.py                                # Process constants, Table 1 & Table 2
├── crystallizer_qmom.py                     # Core 11-state QMOM engine
├── matlab_simulation_data/                  # Stored simulation datasets (CSV)
│   ├── fig1_data.csv
│   ├── fig3_data.csv
│   ├── fig4_fig5_data.csv
│   ├── fig6_fig7_data.csv
│   ├── fig8_fig9_data.csv
│   ├── fig10_data.csv
│   └── fig11_data.csv
├── plots/paper_reproduction/                # 10 publication figures
├── Batch_Cooling_Crystallization_QMOM_Report.pdf  # Comprehensive academic textbook-style report
├── Szilagyi_Lakatos_2015_Crystallization_Paper.pdf # Original benchmark research paper
└── archive/                                 # Archived non-essential scratch files
```

---

## 6. Execution Instructions

### A. Run in MATLAB (Recommended)
Open MATLAB and execute:
```matlab
simulate_all_cases_matlab
```
This runs the full 11-state ODE solver using MATLAB `ode15s` and exports all datasets to `matlab_simulation_data/`.

### B. Run Dual Python Simulation Suite
```bash
.venv/bin/python3 generate_simulation_data.py
```

### C. Generate Reproduction Figures
```bash
.venv/bin/python3 reproduce_paper_results.py
```
This strictly loads the pre-computed CSV datasets from `matlab_simulation_data/` and generates all 10 figures in `plots/paper_reproduction/`.

---

## 7. Physics-Informed Recurrent Neural Networks (PIRNN) Module

The `PIRNN/` package implements a complete deep learning and physics-informed control suite:

```
├── PIRNN/
│   ├── temperature_profiles.py          # 6 analytical & piecewise cooling profiles
│   ├── plant_simulator.py               # Stiff 11-ODE QMOM plant solver + sensor noise
│   ├── data_generator.py                # 60-batch synthetic dataset generator (11,833 steps)
│   ├── dataset.py                       # PyTorch rolling-window DataLoader utilities
│   ├── models.py                        # CrystallizerRecurrentModel (Encoder-Decoder GRU)
│   ├── qmom_physics.py                  # Differentiable PyTorch QMOM physics operator
│   ├── train_baseline_rnn.py            # Stage 2: Pure empirical GRU baseline
│   ├── train_forward_pirnn.py           # Stage 3: Forward PIRNN with 7 physical loss terms
│   ├── evaluate_baseline.py             # Baseline rolling forecast & open-loop rollout
│   ├── compare_models.py                # Direct head-to-head benchmarking engine
│   ├── generate_pdf_report.py           # ReportLab publication PDF report compiler
│   ├── checkpoints/                     # Model weights (.pt) and loss curves
│   ├── plots/                           # High-res comparative evaluation figures
│   └── PIRNN_Comprehensive_Research_Report.pdf # 11-page publication research report
```

### Running the PIRNN Workflow

1. **Synthesize the 60-Batch Crystallization Dataset:**
   ```bash
   PYTHONPATH=. .venv/bin/python3 PIRNN/generate_data.py
   ```
2. **Train Pure Baseline GRU (Stage 2):**
   ```bash
   PYTHONPATH=. .venv/bin/python3 PIRNN/train_baseline_rnn.py --epochs 60
   ```
3. **Train Forward PIRNN (Stage 3):**
   ```bash
   PYTHONPATH=. .venv/bin/python3 PIRNN/train_forward_pirnn.py --epochs 60 --gamma_phys 0.05
   ```
4. **Run Head-to-Head Comparative Benchmark:**
   ```bash
   PYTHONPATH=. .venv/bin/python3 PIRNN/compare_models.py
   ```
5. **Compile Comprehensive 11-Page PDF Technical Report:**
   ```bash
   PYTHONPATH=. .venv/bin/python3 PIRNN/generate_pdf_report.py
   ```


