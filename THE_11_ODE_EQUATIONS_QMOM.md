# The 11 Differential-Algebraic Equations of 2D Plate-Like Batch Crystallization

**Reference Paper**:
Botond Szilágyi & Béla G. Lakatos (2015). *"Batch Cooling Crystallization of Plate-like Crystals: A Simulation Study"*, *Periodica Polytechnica Chemical Engineering*, 59(2), pp. 151–158. [DOI: 10.3311/PPch.7581](https://doi.org/10.3311/PPch.7581).

---

## 1. 2D Morphological Population Balance Model

The 2D plate-like crystal population is governed by the two internal size coordinates: Length ($L_1$) and Width ($L_2$). The thickness $L_3 = k_V = 5.0\ \mu\text{m}$ is assumed constant.

The continuous 2D Population Balance Equation (PBE) with secondary contact nucleation and size-dependent growth is (Eq. 4 of paper):
$$\frac{\partial n}{\partial t} + \frac{\partial (G_1 n)}{\partial L_1} + \frac{\partial (G_2 n)}{\partial L_2} = 0$$

Boundary condition at crystal birth ($L_1 \to 0, L_2 \to 0$):
$$\lim_{L_1, L_2 \to 0} [G_1 n + G_2 n] = B(\sigma, \epsilon, \mu_{11})$$

---

## 2. Quadrature Method of Moments (QMOM) Discretization

To avoid moment-closure issues caused by nonlinear size-dependent growth exponents ($\alpha_1 = 0.8, \alpha_2 = 0.9$), the 2D population density is approximated by $I = 3$ dynamic quadrature nodes (Eq. 15 of paper):
$$\tilde{n}(L_1, L_2, t) = \sum_{i=1}^{3} w_i(t) \delta\big(L_1 - L_{1,i}(t)\big) \delta\big(L_2 - L_{2,i}(t)\big)$$

This discretization defines an exact **11-state ODE/DAE system**:
$$\mathbf{y}(t) = \begin{bmatrix} L_{1,1} \\ L_{1,2} \\ L_{1,3} \\ L_{2,1} \\ L_{2,2} \\ L_{2,3} \\ w_1 \\ w_2 \\ w_3 \\ c \\ T \end{bmatrix} \in \mathbb{R}^{11}$$

---

## 3. The 11 Governing Differential Equations

### Growth ODEs for Quadrature Nodes (ODEs 1–6)
Each quadrature node abscissa grows along length ($L_1$) and width ($L_2$) according to the kinetic growth law (Eq. 2 of paper):

$$\begin{aligned}
\text{ODE 1:}\quad \frac{dL_{1,1}}{dt} &= G_1(\sigma, L_{1,1}) = k_1 \, \sigma^{g_1} \, (1 + \gamma_1 L_{1,1})^{\alpha_1} \\
\text{ODE 2:}\quad \frac{dL_{1,2}}{dt} &= G_1(\sigma, L_{1,2}) = k_1 \, \sigma^{g_1} \, (1 + \gamma_1 L_{1,2})^{\alpha_1} \\
\text{ODE 3:}\quad \frac{dL_{1,3}}{dt} &= G_1(\sigma, L_{1,3}) = k_1 \, \sigma^{g_1} \, (1 + \gamma_1 L_{1,3})^{\alpha_1} \\
\text{ODE 4:}\quad \frac{dL_{2,1}}{dt} &= G_2(\sigma, L_{2,1}) = k_2 \, \sigma^{g_2} \, (1 + \gamma_2 L_{2,1})^{\alpha_2} \\
\text{ODE 5:}\quad \frac{dL_{2,2}}{dt} &= G_2(\sigma, L_{2,2}) = k_2 \, \sigma^{g_2} \, (1 + \gamma_2 L_{2,2})^{\alpha_2} \\
\text{ODE 6:}\quad \frac{dL_{2,3}}{dt} &= G_2(\sigma, L_{2,3}) = k_2 \, \sigma^{g_2} \, (1 + \gamma_2 L_{2,3})^{\alpha_2}
\end{aligned}$$

---

### Dynamic Quadrature Weights Linear System (ODEs 7–9)
Applying the moment transformation to the continuous PBE yields the dynamic moment evolution (Eq. 17 of paper):
$$\frac{d\mu_{00}}{dt} = \sum_{i=1}^3 \frac{dw_i}{dt} = B$$
$$\frac{d\mu_{10}}{dt} = \sum_{i=1}^3 \left( \frac{dw_i}{dt} L_{1,i} + w_i G_{1,i} \right) \implies \sum_{i=1}^3 \frac{dw_i}{dt} L_{1,i} = 0$$
$$\frac{d\mu_{01}}{dt} = \sum_{i=1}^3 \left( \frac{dw_i}{dt} L_{2,i} + w_i G_{2,i} \right) \implies \sum_{i=1}^3 \frac{dw_i}{dt} L_{2,i} = 0$$

Writing in matrix-vector form:
$$\begin{bmatrix} 1 & 1 & 1 \\ L_{1,1} & L_{1,2} & L_{1,3} \\ L_{2,1} & L_{2,2} & L_{2,3} \end{bmatrix} \begin{bmatrix} dw_1/dt \\ dw_2/dt \\ dw_3/dt \end{bmatrix} = \begin{bmatrix} B \\ 0 \\ 0 \end{bmatrix}$$

$$\begin{aligned}
\text{ODE 7:}\quad \frac{dw_1}{dt} &= \left( \mathbf{M}^{-1} \begin{bmatrix} B \\ 0 \\ 0 \end{bmatrix} \right)_1 \\
\text{ODE 8:}\quad \frac{dw_2}{dt} &= \left( \mathbf{M}^{-1} \begin{bmatrix} B \\ 0 \\ 0 \end{bmatrix} \right)_2 \\
\text{ODE 9:}\quad \frac{dw_3}{dt} &= \left( \mathbf{M}^{-1} \begin{bmatrix} B \\ 0 \\ 0 \end{bmatrix} \right)_3
\end{aligned}$$

---

### Solute Mass Balance ODE (ODE 10)
Crystal precipitation removes solute from the liquid phase (Eq. 8–9 of paper):
$$\text{ODE 10:}\quad \frac{dc}{dt} = -\rho_c R_V$$
where $R_V$ is the volumetric crystal growth rate per unit suspension volume:
$$R_V = k_V \frac{d\mu_{11}}{dt} = k_V \left[ \sum_{i=1}^3 \frac{dw_i}{dt} L_{1,i} L_{2,i} + \sum_{i=1}^3 w_i \big( G_1(\sigma, L_{1,i}) L_{2,i} + G_2(\sigma, L_{2,i}) L_{1,i} \big) \right]$$

---

### Batch Temperature Cooling Schedule (ODE 11)
Linear batch cooling down to $T_{\text{final}} = 25.0^\circ\text{C}$ (Eq. 10 of paper):
$$\text{ODE 11:}\quad \frac{dT}{dt} = \begin{cases} -cr, & T > T_{\text{final}} \\ 0, & T \le T_{\text{final}} \end{cases}$$

---

## 4. Auxiliary Constitutive Equations

1. **Apelblat Solubility Model** (Eq. 11):
   $$c_s(T) = 1000 \cdot \exp\left( -69.51 + \frac{368.6}{T} + 16.18 \ln T \right) \quad [\text{kg/m}^3]$$

2. **Relative Supersaturation** (Eq. 1):
   $$\sigma = \max\left( 0,\, \frac{c - c_s(T)}{c_s(T)} \right)$$

3. **Secondary Contact Nucleation Rate** (Eq. 1):
   $$B = k_S \cdot \epsilon \cdot \mu_{11} \cdot \sigma^{b_1} \quad [\#/(\text{m}^3 \text{s})]$$
   where $\mu_{11} = \sum_{i=1}^3 w_i L_{1,i} L_{2,i}$ represents the cross-moment proxy for contact surface area.

4. **Product Particle Statistics**:
   - Total number density: $\mu_{00} = \sum w_i$
   - First length moment: $\mu_{10} = \sum w_i L_{1,i}$
   - First width moment: $\mu_{01} = \sum w_i L_{2,i}$
   - Number-mean length: $\langle L_1 \rangle = \mu_{10} / \mu_{00}$
   - Number-mean width: $\langle L_2 \rangle = \mu_{01} / \mu_{00}$
   - Number-mean aspect ratio: $\text{AR} = \langle L_1 \rangle / \langle L_2 \rangle$

---

## 5. Model Parameters (Table 1 & Calibrated Regime)

| Parameter | Symbol | Table 1 Printed | Calibrated Value | Unit | Description / Calibration Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Length growth rate constant | $k_1$ | $1.178 \times 10^{-3}$ | $1.30 \times 10^{-3}$ | $\text{m/s}$ | Rate coefficient for length growth (calibrated to Fig. 3) |
| Width growth rate constant | $k_2$ | $1.874 \times 10^{-4}$ | $2.40 \times 10^{-4}$ | $\text{m/s}$ | Rate coefficient for width growth (calibrated to Fig. 3) |
| Length growth exponent | $g_1$ | $1.5$ | $1.7$ | $-$ | Transposed from Table 1; Page 6 notes Face 1 has higher supersaturation dependency |
| Width growth exponent | $g_2$ | $1.7$ | $1.5$ | $-$ | Transposed from Table 1; explains aspect ratio increase at low $T_s$ |
| Size-dependence coefficient (length) | $\gamma_1$ | $0.05$ | $0.05$ | $-$ | Size dependency factor for length |
| Size-dependence coefficient (width) | $\gamma_2$ | $0.03$ | $0.03$ | $-$ | Size dependency factor for width |
| Size-dependence exponent (length) | $\alpha_1$ | $0.8$ | $0.8$ | $-$ | Fractional size dependency exponent |
| Size-dependence exponent (width) | $\alpha_2$ | $0.9$ | $0.9$ | $-$ | Fractional size dependency exponent |
| Nucleation rate constant | $k_S$ | $3.6 \times 10^5$ | $8.5 \times 10^5$ | $\#/(\text{m}^3\text{s}(\text{W/kg}))$ | Produces the observed $\sim 15-20\%$ nucleation dilution across $\epsilon$ |
| Nucleation exponent | $b_1$ | $2.0$ | $2.0$ | $-$ | Secondary nucleation supersaturation order |
| Plate crystal thickness | $k_V = L_3$ | $5.0 \times 10^{-6}$ | $5.0 \times 10^{-6}$ | $\text{m}$ | Constant plate crystal thickness |
| Crystal solid density | $\rho_c$ | $1665.0$ | $1665.0$ | $\text{kg/m}^3$ | Solid density of crystalline product |
| Initial solute concentration | $c_0$ | $240.0$ | $240.0$ | $\text{kg/m}^3$ | Initial concentration at saturated state |
| Nominal seeding temperature | $T_s$ | $35.0$ | $35.0$ | $^\circ\text{C}$ | Seeding temperature |
| Final batch temperature | $T_f$ | $25.0$ | $25.0$ | $^\circ\text{C}$ | Target cooling crystallization temperature |
| Nominal cooling rate | $cr$ | $8.33 \times 10^{-4}$ | $8.33 \times 10^{-4}$ | $^\circ\text{C/s}$ | Linear cooling schedule slope |
| Nominal stirring power | $\epsilon$ | $250.0$ | $250.0$ | $\text{W/kg}$ | Specific energy dissipation / stirring power |

### Initial Quadrature Abscissas & Weights (Table 2)
$$\mathbf{L}_1(0) = \begin{bmatrix} 9.74 \times 10^{-5} \\ 9.80 \times 10^{-5} \\ 1.138 \times 10^{-4} \end{bmatrix}\ \text{m}, \quad \mathbf{L}_2(0) = \begin{bmatrix} 6.85 \times 10^{-5} \\ 5.74 \times 10^{-5} \\ 6.05 \times 10^{-5} \end{bmatrix}\ \text{m}, \quad \mathbf{w}(0) = \begin{bmatrix} 1.86 \times 10^{10} \\ 6.46 \times 10^{10} \\ 1.281 \times 10^{11} \end{bmatrix}\ \#/\text{m}^3$$

---

## 6. Exact MATLAB Implementation (`simulate_all_cases_matlab.m`)

The 11 ODEs are solved using MATLAB's stiff variable-order solver `ode15s`:
```matlab
% Initial 11-state vector
y0 = [p.L1_0; p.L2_0; p.w0; p.c0; p.T_seed];

% Solve with ode15s (Backward Differentiation Formulas)
opts = odeset('RelTol', 1e-6, 'AbsTol', 1e-8);
[t, y] = ode15s(@(t, y) qmom_rhs(t, y, p), tspan, y0, opts);
```

To execute in MATLAB:
```matlab
simulate_all_cases_matlab
```
The script integrates all cases and saves the exact simulation datasets into `matlab_simulation_data/`.
