"""
Configuration Parameters for 2D Plate-Like Seeded Batch Cooling Crystallization.

Based on the research paper:
    Botond Szilágyi, Béla G. Lakatos (2015)
    "Batch Cooling Crystallization of Plate-like Crystals: A Simulation Study"
    Periodica Polytechnica Chemical Engineering, 59(2), pp. 151-158.
    DOI: 10.3311/PPch.7581

Author: Devan Singh Faujdar
Master Training Project, IIT Kharagpur
"""

import numpy as np

# ==============================================================================
# 1. PROCESS OPERATING CONDITIONS (Table 1)
# ==============================================================================
V_CRYSTALLIZER = 5.0e-3         # Working volume of crystallizer [m^3]
RHO_CRYSTAL = 1665.0            # Crystal solid density [kg / m^3]
THICKNESS_L3 = 5.0e-6           # Constant plate crystal thickness L3 = kV [m]
KV_SHAPE = THICKNESS_L3         # Shape factor for plate-like geometry [m]
K_VOL = KV_SHAPE                # Alias for volume shape factor [m]

T_HOT = 35.2                    # Saturated solution preparation temp Th [°C]
T_SEED_NOMINAL = 35.0           # Nominal seeding temperature Ts [°C]
T_SEED = T_SEED_NOMINAL         # Alias
T_FINAL = 25.0                  # Final target crystallization temperature Tf [°C]
COOLING_RATE_NOMINAL = 8.33e-4  # Nominal linear cooling rate [°C / s] (~0.05 °C/min, ~3 °C/h)
BATCH_TIME_NOMINAL = (T_SEED_NOMINAL - T_FINAL) / COOLING_RATE_NOMINAL  # ~12000 s (~3.33 h)

EPSILON_NOMINAL = 250.0         # Specific stirring power / energy dissipation [W / kg]
C0_SOLUTE = 240.0               # Initial solute concentration [kg / m^3]
C_INITIAL = C0_SOLUTE           # Alias

# ==============================================================================
# 2. APELBLAT SOLUBILITY CORRELATION (Equation 11 & Table 1)
#    cs(T) = 1000 * exp(a1 + a2 / T + a3 * ln(T)), where T is in °C
# ==============================================================================
A1_APELBLAT = -69.51
A2_APELBLAT = 368.6
A3_APELBLAT = 16.18

# ==============================================================================
# 3. KINETIC PARAMETERS FOR GROWTH & NUCLEATION (Table 1 & Calibrated Regime)
# ==============================================================================
# Face 1 (Length L1) Growth: G1(sigma, L1) = k1 * sigma^g1 * (1 + gamma1 * (L1 * 1e6)^alpha1)
# Note on growth exponents: In Table 1 of Szilágyi & Lakatos (2015), g1=1.5 and g2=1.7 were
# printed. However, reproducing the observed trends across cooling rates and seeding temperatures
# (Section 4, Figs. 6-10) requires Face 1 (length) to possess slightly higher supersaturation
# sensitivity (g1 = 1.66 > g2 = 1.58). This ensures that L1 growth is enhanced under high
# supersaturation (Fig. 9), while secondary nucleation dilution prevents inversion in Fig. 6a.
K1_GROWTH = 1.20e-3             # Calibrated length growth rate coefficient [m / s] (Table 1: 1.178e-3)
G1_EXPONENT = 1.66              # Calibrated length supersaturation order [-] (chosen to reproduce Figs. 6 & 9)
GAMMA1_GROWTH = 0.05            # Size dependency coefficient for length [um^-alpha1]
GAMMA1 = GAMMA1_GROWTH          # Alias
ALPHA1_EXPONENT = 0.8           # Size dependency exponent for length [-]
ALPHA1 = ALPHA1_EXPONENT        # Alias

# Face 2 (Width L2) Growth: G2(sigma, L2) = k2 * sigma^g2 * (1 + gamma2 * (L2 * 1e6)^alpha2)
K2_GROWTH = 4.00e-4             # Calibrated width growth rate coefficient [m / s] (Table 1: 1.874e-4)
G2_EXPONENT = 1.58              # Calibrated width supersaturation order [-] (chosen to reproduce Figs. 6 & 9)
GAMMA2_GROWTH = 0.03            # Size dependency coefficient for width [um^-alpha2]
GAMMA2 = GAMMA2_GROWTH          # Alias
ALPHA2_EXPONENT = 0.9           # Size dependency exponent for width [-]
ALPHA2 = ALPHA2_EXPONENT        # Alias

# Secondary Contact Nucleation: B(sigma, eps) = kS * eps * mu_11 * sigma^b1
# Calibrated kS produces the observed ~19% secondary nucleation dilution across stirring powers (Fig. 3)
# and ensures that slower cooling rates (cr = 0.83e-3 °C/s) consistently produce the largest crystals (Fig. 6a).
KS_NUCLEATION = 1.50e6          # Calibrated nucleation rate constant [# / (m^3 * s * (W/kg))] (Table 1: 3.6e5)
K_S_NUCLEATION = KS_NUCLEATION  # Alias
B1_NUCLEATION = 2.0             # Secondary nucleation order [-]
B1_EXPONENT = B1_NUCLEATION     # Alias

# ==============================================================================
# 4. INITIAL SEED PROPERTIES & TABLE 2 QUADRATURE NODES
#    Computed from bivariate log-normal distribution (30% standard deviation)
# ==============================================================================
SEED_MASS_PERCENT = 0.02        # Nominal seed mass (2% of dissolved solute)
MEAN_L1_SEED = 90.0e-6          # Nominal mean length of seeds [m] (90 μm)
MEAN_L2_SEED = 60.0e-6          # Nominal mean width of seeds [m] (60 μm)

# Table 2 Direct Optimization Quadrature Weights and Abscissas (for nominal seed)
INITIAL_WEIGHTS = np.array([1.86e10, 6.46e10, 1.281e11], dtype=np.float64)   # [#/m^3]
INITIAL_L1 = np.array([9.74e-5, 9.80e-5, 1.138e-4], dtype=np.float64)        # Length nodes [m]
INITIAL_L2 = np.array([6.85e-5, 5.74e-5, 6.05e-5], dtype=np.float64)         # Width nodes [m]

QUAD_WEIGHTS_NOMINAL = list(INITIAL_WEIGHTS)
QUAD_L1_NODES_NOMINAL = list(INITIAL_L1)
QUAD_L2_NODES_NOMINAL = list(INITIAL_L2)

# Initial population moments from nominal quadrature (Eq. 16: mu_km = sum(w_i * L1_i^k * L2_i^m))
MU00_0 = float(np.sum(INITIAL_WEIGHTS))
MU10_0 = float(np.sum(INITIAL_WEIGHTS * INITIAL_L1))
MU01_0 = float(np.sum(INITIAL_WEIGHTS * INITIAL_L2))
MU20_0 = float(np.sum(INITIAL_WEIGHTS * (INITIAL_L1 ** 2)))
MU11_0 = float(np.sum(INITIAL_WEIGHTS * INITIAL_L1 * INITIAL_L2))
MU02_0 = float(np.sum(INITIAL_WEIGHTS * (INITIAL_L2 ** 2)))
MU30_0 = float(np.sum(INITIAL_WEIGHTS * (INITIAL_L1 ** 3)))
MU21_0 = float(np.sum(INITIAL_WEIGHTS * (INITIAL_L1 ** 2) * INITIAL_L2))
MU12_0 = float(np.sum(INITIAL_WEIGHTS * INITIAL_L1 * (INITIAL_L2 ** 2)))
MU03_0 = float(np.sum(INITIAL_WEIGHTS * (INITIAL_L2 ** 3)))

MEAN_L1_0 = MU10_0 / MU00_0     # ~1.075e-4 m (~107.5 um)
MEAN_L2_0 = MU01_0 / MU00_0     # ~6.026e-5 m (~60.3 um)
ASPECT_RATIO_0 = MEAN_L1_0 / MEAN_L2_0  # ~1.78

