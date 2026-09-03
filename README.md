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

## 4. Repository Structure

```
├── Szilagyi_Lakatos_2015_Crystallization_Paper.pdf  # Original research paper
├── config.py                                        # Process constants, Table 1 & Table 2
├── crystallizer_mom.py                              # First-principles 2D QMOM ODE simulator
├── reproduce_paper_results.py                       # Runs simulation to reproduce Figures 1-11
├── plots/
│   └── paper_reproduction/                          # High-resolution benchmark figures
│       ├── fig1_concentration_solubility.png
│       ├── fig3_mean_crystal_sizes.png
│       ├── fig6_cooling_stirring_effects.png
│       ├── fig8_product_sizes_vs_Ts.png
│       ├── fig9_aspect_ratio_vs_Ts.png
│       ├── fig10_nucleation_rate_evolution.png
│       └── fig11_3d_phase_trajectory.png
└── README.md
```

---

## 5. How to Run the Reproduction

### Step 1: Install Dependencies
```bash
pip install numpy scipy matplotlib
```

### Step 2: Test the First-Principles MOM Simulator
```bash
python3 crystallizer_mom.py
```
*Expected Output:*
```text
Testing First-Principles Method of Moments Batch Crystallizer...
Simulation completed successfully!
Time span: 0 to 12005 s
Initial Concentration: 240.00 kg/m^3 -> Final: 68.26 kg/m^3
Initial <L1>: 107.5 um -> Final: 479.3 um
Initial <L2>: 60.3 um -> Final: 213.1 um
Initial Aspect Ratio: 1.78 -> Final: 2.25
Final mu11: 22000.26 m^2/m^3
```

### Step 3: Generate All Published Figures
```bash
python3 reproduce_paper_results.py
```
This runs the full parametric matrix across cooling rates, stirring powers, and seeding temperatures, generating the exact benchmark plots matching the paper in `plots/paper_reproduction/`.
