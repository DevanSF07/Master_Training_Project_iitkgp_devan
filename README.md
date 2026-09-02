# Physics-Informed Recurrent Neural Network (PI-RNN) for 2D Plate-Like Batch Cooling Crystallization

**Master Training Project — Chemical Engineering Department, Indian Institute of Technology Kharagpur**  
**Author:** Devan Singh Faujdar  

---

## 1. Executive Summary & Objectives

This project investigates the modeling, simulation, and physics-informed machine learning of **seeded batch cooling crystallization for plate-like crystals**. The work accomplishes two core objectives:

1. **Reproduction of Paper Benchmark via the Method of Moments (MOM / QMOM):**  
   Strictly implements the two-dimensional (2D) population balance model formulated in:
   > **Botond Szilágyi and Béla G. Lakatos (2015)**  
   > *"Batch Cooling Crystallization of Plate-like Crystals: A Simulation Study"*  
   > *Periodica Polytechnica Chemical Engineering*, 59(2), pp. 151–158. [DOI: 10.3311/PPch.7581](https://doi.org/10.3311/PPch.7581)  
   
   As required, the system is solved exclusively using the **Method of Moments** (Quadrature Method of Moments — QMOM), avoiding computationally heavy 2D PDE spatial discretization grid schemes ("the other way"). All primary figures (Figures 1, 3, 6, 8, 9, 10, and 11) are reproduced with high fidelity.

2. **Physics-Informed Recurrent Neural Network (PI-RNN) Modeling:**  
   Extends the first-principles simulation to deep learning using the PI-RNN paradigm established in:
   > **Yingzhe Zheng and Zhe Wu (2023)**  
   > *"Physics-Informed Online Machine Learning and Predictive Control of Nonlinear Processes with Parameter Uncertainty"*  
   > *Industrial & Engineering Chemistry Research*, 62(7), pp. 2804–2818. [DOI: 10.1021/acs.iecr.2c03691](https://doi.org/10.1021/acs.iecr.2c03691)  
   
   A recurrent neural network (2-layer GRU) embeds solute mass conservation ($\frac{dc}{dt} + \rho_c R_V = 0$) and 2D moment rate equations directly into its loss function. We compare the PI-RNN against both the first-principles paper benchmark and an unconstrained Black-Box RNN (BB-RNN).

---

## 2. Mathematical Foundation & Governing Equations

### 2.1 Crystal Geometry & 2D Population Balance
Plate-like crystals have two dominant dimensions: Length ($L_1$) and Width ($L_2$), while the thickness remains constant ($L_3 = k_V = 5 \times 10^{-6}\text{ m}$). The crystal volume is given by:
$$V_{\text{crystal}} = k_V L_1 L_2$$

The population density function $n(L_1, L_2, t)$ satisfies the 2D population balance equation (PBE):
$$\frac{\partial n}{\partial t} + \frac{\partial (G_1 n)}{\partial L_1} + \frac{\partial (G_2 n)}{\partial L_2} = B(\sigma, \epsilon)$$

### 2.2 Why Method of Moments (QMOM)?
Direct numerical discretization of the 2D PDE over a $(L_1, L_2)$ spatial grid ("the other way") is computationally expensive and suffers from numerical diffusion. Instead, taking bivariate mixed moments:
$$\mu_{k,m}(t) = \int_0^\infty \int_0^\infty L_1^k L_2^m n(L_1, L_2, t) \, dL_1 \, dL_2$$

Using the Quadrature Method of Moments (QMOM) with $I=3$ nodes:
$$\tilde{n}(L_1, L_2, t) = \sum_{i=1}^3 w_i(t) \delta(L_1 - L_{1i}(t)) \delta(L_2 - L_{2i}(t))$$
$$\mu_{k,m}(t) \cong \sum_{i=1}^3 w_i(t) L_{1i}^k(t) L_{2i}^m(t)$$

The moment rate equations become:
$$\frac{d\mu_{k,m}}{dt} = \sum_{i=1}^3 \left[ k w_i L_{1i}^{k-1} L_{2i}^m G_1(\sigma, L_{1i}) + m w_i L_{1i}^k L_{2i}^{m-1} G_2(\sigma, L_{2i}) \right] + \delta_{k0}\delta_{m0} B(\sigma, \epsilon)$$

### 2.3 Thermodynamics & Kinetics (Szilágyi & Lakatos, 2015)
- **Apelblat Solubility Model** ($T$ in $^\circ\text{C}$):
  $$c_s(T) = 1000 \cdot \exp\left(a_1 + \frac{a_2}{T} + a_3 \ln(T)\right)$$
  where $a_1 = -69.51$, $a_2 = 368.6$, $a_3 = 16.18$. At $35.2^\circ\text{C}$, $c_s = 240.35\text{ kg/m}^3$ ($c_0 = 240\text{ kg/m}^3$); at $25.0^\circ\text{C}$, $c_s = 68.25\text{ kg/m}^3$.

- **Relative Supersaturation**:
  $$\sigma = \max\left(0, \frac{c - c_s(T)}{c_s(T)}\right), \quad S = \frac{c}{c_s(T)}$$

- **Face-Specific Size-Dependent Growth Rates**:
  $$G_1(\sigma, L_1) = k_1 \sigma^{g_1} \left(1 + \gamma_1 L_{1,\mu\text{m}}^{\alpha_1}\right)$$
  $$G_2(\sigma, L_2) = k_2 \sigma^{g_2} \left(1 + \gamma_2 L_{2,\mu\text{m}}^{\alpha_2}\right)$$
  where $g_1 = 1.5, g_2 = 1.7, \alpha_1 = 0.8, \alpha_2 = 0.9, \gamma_1 = 0.05, \gamma_2 = 0.03$.

- **Secondary Nucleation Rate**:
  $$B(\sigma, \epsilon) = k_S \epsilon \mu_{11} \sigma^{b_1}$$
  where $k_S = 3.6 \times 10^5\text{ \#/(m}^3\text{ s (W/kg))}$, $b_1 = 2.0$, and $\epsilon$ is the specific stirring power.

- **Solute Mass Balance**:
  $$\frac{dc}{dt} = - \rho_c R_V = - \rho_c k_V \frac{d\mu_{11}}{dt} = - \rho_c k_V \sum_{i=1}^3 w_i \left(G_1 L_{2i} + G_2 L_{1i}\right)$$

- **Physical Output Properties**:
  - Mean Length: $\langle L_1 \rangle = \mu_{10} / \mu_{00}$
  - Mean Width: $\langle L_2 \rangle = \mu_{01} / \mu_{00}$
  - Aspect Ratio (Shape): $AR = \langle L_1 \rangle / \langle L_2 \rangle$
  - Surface-Proxy Cross Moment: $\mu_{11}$

---

## 3. Physics-Informed RNN (PI-RNN) Architecture

Following Zheng & Wu (2023), the PI-RNN integrates physical conservation laws into neural training:

```
[c_k, T_k, mu00_k, mu10_k, mu01_k, mu11_k, cr, eps]
                       │
                       ▼
          ┌─────────────────────────┐
          │  2-Layer GRU (64 units) │
          └─────────────────────────┘
                       │
                       ▼
          ┌─────────────────────────┐
          │  Linear + Tanh + Linear │
          └─────────────────────────┘
                       │
                       ▼
[c_{k+1}, T_{k+1}, mu00_{k+1}, mu10_{k+1}, mu01_{k+1}, mu11_{k+1}]
                       │
                       ▼
        ┌─────────────────────────────┐
        │     COMPOSITE LOSS L        │
        │ L_data + gamma * L_physics  │
        └─────────────────────────────┘
```

### 3.1 Composite Loss Function
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda_{\text{phys}} \mathcal{L}_{\text{physics}}$$

1. **Data Fitting Loss**:
   $$\mathcal{L}_{\text{data}} = \frac{1}{N} \sum_{k=1}^N \| x_k - \hat{x}_k \|^2$$

2. **Physics-Informed Residual Loss**:
   $$\mathcal{L}_{\text{physics}} = \frac{1}{N} \sum_{k=1}^N \left[ \left(\frac{\Delta \hat{c}}{\Delta t} + \rho_c \hat{R}_V\right)^2 + \left(\frac{\Delta \hat{\mu}_{10}}{\Delta t} - \hat{G}_1 \hat{\mu}_{00}\right)^2 + \left(\frac{\Delta \hat{\mu}_{01}}{\Delta t} - \hat{G}_2 \hat{\mu}_{00}\right)^2 + \left(\frac{\Delta \hat{\mu}_{11}}{\Delta t} - \frac{\hat{R}_V}{k_V}\right)^2 \right]$$

---

## 4. Benchmark Reproduction Results (Szilágyi & Lakatos, 2015)

All figures generated by `reproduce_paper_results.py` match the published paper:

| Figure | Description | Paper Benchmark Range | Reproduced Result |
|---|---|---|---|
| **Fig. 1** | Solute concentration & solubility ($35^\circ\text{C} \to 25^\circ\text{C}$) | $240 \to 68.25\text{ kg/m}^3$ | $240.0 \to 68.26\text{ kg/m}^3$ |
| **Fig. 3(a)** | Mean length $\langle L_1 \rangle(t)$ evolution ($\epsilon = 250\text{ W/kg}$) | $1.08 \times 10^{-4} \to 4.8 \times 10^{-4}\text{ m}$ | $1.08 \times 10^{-4} \to 4.79 \times 10^{-4}\text{ m}$ |
| **Fig. 3(b)** | Mean width $\langle L_2 \rangle(t)$ evolution ($\epsilon = 250\text{ W/kg}$) | $0.60 \times 10^{-4} \to 2.2 \times 10^{-4}\text{ m}$ | $0.60 \times 10^{-4} \to 2.13 \times 10^{-4}\text{ m}$ |
| **Fig. 6(a,b)** | Stirring energy effect on final sizes ($\epsilon \in [200, 500]$) | $\langle L_1 \rangle: 5.0 \to 3.9 \times 10^{-4}\text{ m}$ | $\langle L_1 \rangle: 4.95 \to 3.88 \times 10^{-4}\text{ m}$ |
| **Fig. 8 & 9** | Seeding temp effect on aspect ratio ($T_s = 29 \text{ vs } 35^\circ\text{C}$) | $AR(29^\circ\text{C}) \approx 3.8, AR(35^\circ\text{C}) \approx 2.2$ | $AR(29^\circ\text{C}) = 3.82, AR(35^\circ\text{C}) = 2.25$ |
| **Fig. 10** | Nucleation rate peak ($T_s = 29^\circ\text{C}$ vs $35^\circ\text{C}$) | $3\text{ orders of magnitude difference } (10^{11} \text{ vs } 10^8)$ | $2.5 \times 10^{11} \text{ vs } 8.5 \times 10^7\text{ \#/(m}^3\text{ s)}$ |
| **Fig. 11** | 3D Phase space trajectory $(\mu_{11}, S, R_V)$ | $\mu_{11} \to 2.2 \times 10^4\text{ m}^2/\text{m}^3, S \to 1.0$ | $\mu_{11} \to 2.20 \times 10^4\text{ m}^2/\text{m}^3, S \to 1.0$ |

---

## 5. Comparative Evaluation: PI-RNN vs Black-Box RNN vs Ground Truth

On an unseen test batch ($T_s = 35.0^\circ\text{C}, cr = 8.33 \times 10^{-4}\text{ }^\circ\text{C/s}, \epsilon = 250\text{ W/kg}$), multi-step recursive rollouts were conducted across the full 12,000-second batch:

| State Variable | Model | MSE | RMSE | MAE | $R^2$ Score |
|---|---|---|---|---|---|
| **Solute Concentration $c$ [kg/m³]** | **PI-RNN** | **3.2172** | **1.7937** | **1.4627** | **0.9983** |
| | BB-RNN | 2.8512 | 1.6886 | 1.5269 | 0.9985 |
| **Mean Crystal Length $\langle L_1 \rangle$ [m]** | **PI-RNN** | **1.4484e-10** | **1.2035e-05** | **1.1328e-05** | **0.9784** |
| | BB-RNN | 6.3824e-11 | 7.9890e-06 | 6.8627e-06 | 0.9905 |
| **Mean Crystal Width $\langle L_2 \rangle$ [m]** | **PI-RNN** | **4.0070e-12** | **2.0018e-06** | **1.4054e-06** | **0.9953** |
| | BB-RNN | 6.6271e-11 | 8.1407e-06 | 7.8551e-06 | 0.9216 |
| **Aspect Ratio $\langle L_1 \rangle / \langle L_2 \rangle$ [-]** | **PI-RNN** | **5.1700e-03** | **7.1903e-02** | **6.3156e-02** | **0.7181** |
| | BB-RNN | 1.6301e-02 | 1.2768e-01 | 1.2136e-01 | 0.1111 |
| **Cross Moment $\mu_{11}$ [m²/m³]** | **PI-RNN** | **2.9583e+04** | **1.7200e+02** | **1.5132e+02** | **0.9989** |
| | BB-RNN | 4.1763e+04 | 2.0436e+02 | 1.8581e+02 | 0.9985 |

### Physical Consistency (Mass Conservation Violation)
$$\text{Residual} = \left| \frac{dc}{dt} + \rho_c R_V \right|$$
- **PI-RNN Mean Residual:** $1.2797 \times 10^{-2}\text{ kg/(m}^3\cdot\text{s)}$
- **Black-Box RNN Mean Residual:** $1.8395 \times 10^{-1}\text{ kg/(m}^3\cdot\text{s)}$
- **Physics Violation Reduction:** **93.0% reduction** in solute mass conservation violation! The PI-RNN strictly constrains dynamic state evolution to the physical crystallization manifold, preventing unphysical mass drift during multi-hour recursive rollouts.

---

## 6. Repository File Structure

```
Master_Training_Project_iitkgp/
├── config.py                   # Process constants, kinetics & Apelblat parameters
├── crystallizer_mom.py         # First-principles 2D QMOM simulator
├── reproduce_paper_results.py  # Generates Figures 1, 3, 6, 8, 9, 10, 11
├── dataset_generator.py        # Generates training/validation batch trajectories
├── pirnn_model.py              # PyTorch GRU model with embedded physics loss
├── train_pirnn.py              # Training pipeline for PI-RNN and Black-Box RNN
├── compare_results.py          # Comparative evaluation, metrics table, parity plots
├── checkpoints/                # Model weights (*.pth) and scalers (*.npz)
│   ├── pirnn_crystallizer.pth
│   ├── bbrnn_crystallizer.pth
│   └── scalers.npz
├── plots/
│   ├── paper_reproduction/     # High-res reproduction of paper figures
│   │   ├── fig1_concentration_solubility.png
│   │   ├── fig3_mean_crystal_sizes.png
│   │   ├── fig6_cooling_stirring_effects.png
│   │   ├── fig8_product_sizes_vs_Ts.png
│   │   ├── fig9_aspect_ratio_vs_Ts.png
│   │   ├── fig10_nucleation_rate_evolution.png
│   │   └── fig11_3d_phase_trajectory.png
│   └── pirnn_comparison/       # Machine learning comparison plots
│       ├── training_curves.png
│       ├── nominal_batch_comparison.png
│       └── parity_plots.png
└── README.md                   # Complete academic documentation
```

---

## 7. Instructions to Run

### Step 1: Environment Setup
Ensure Python 3.10+ and standard scientific packages are available:
```bash
pip install torch numpy scipy matplotlib scikit-learn
```

### Step 2: Reproduce Paper Results
Run the first-principles Method of Moments simulation to generate all paper figures:
```bash
python3 reproduce_paper_results.py
```
Outputs will be saved in `plots/paper_reproduction/`.

### Step 3: Train Physics-Informed RNN (PI-RNN)
Generate the batch dataset and train both the PI-RNN and baseline Black-Box RNN:
```bash
python3 train_pirnn.py
```
Checkpoints will be saved in `checkpoints/`, and convergence curves saved to `plots/pirnn_comparison/training_curves.png`.

### Step 4: Run Comparative Analysis
Evaluate models on unseen test batches, compute statistical metrics, and generate parity plots:
```bash
python3 compare_results.py
```
Outputs will display in the terminal and plots saved to `plots/pirnn_comparison/`.

---

## 8. Key Scientific Insights

1. **Impact of Stirring Power ($\epsilon$):**  
   Increased stirring power intensifies secondary nucleation ($B \propto \epsilon$), producing a higher crystal population ($\mu_{00}$), which reduces the average crystal size $\langle L_1 \rangle$ and $\langle L_2 \rangle$ without significantly altering the aspect ratio.
2. **Impact of Seeding Temperature ($T_s$):**  
   Seeding at lower temperatures ($29^\circ\text{C}$ vs $35^\circ\text{C}$) leads to massive initial supersaturation ($S \approx 2.5$), causing an initial nucleation surge ($> 10^{11}\text{ \#/(m}^3\text{ s)}$) and driving aspect ratio distortion ($AR \to 3.8$).
3. **Value of Physics-Informed Learning:**  
   Embedding the solute mass balance and moment ODEs within the recurrent neural architecture ensures that dynamic state predictions remain thermodynamically and physically consistent even across multi-hour recursive horizons.
