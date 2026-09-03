"""
Configuration Parameters for 2D Plate-Like Batch Cooling Crystallization.

Based on the research paper:
    Botond Szilagyi, Bela G. Lakatos (2015)
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
V_CRYSTALLIZER = 5.0e-3     # Working volume of crystallizer [m^3]
RHO_CRYSTAL = 1665.0        # Crystal solid density [kg / m^3]
THICKNESS_L3 = 5.0e-6       # Constant plate crystal thickness L3 = kV [m]
KV_SHAPE = THICKNESS_L3     # Shape factor for plate-like geometry [m]

T_HOT = 35.2                # Hot saturation temperature [°C]
T_SEED = 35.0               # Seeding temperature [°C]
T_FINAL = 25.0              # Final target cooling temperature [°C]
COOLING_RATE_NOMINAL = 8.33e-4  # Nominal linear cooling rate [°C / s] (~3 °C/h)
BATCH_TIME_NOMINAL = (T_SEED - T_FINAL) / COOLING_RATE_NOMINAL  # ~12000 s (~3.33 h)

EPSILON_NOMINAL = 250.0     # Specific stirring power / energy dissipation [W / kg]
C0_SOLUTE = 240.0           # Initial solute concentration [kg / m^3]

# ==============================================================================
# 2. APELBLAT SOLUBILITY CORRELATION (Equation 11 & Table 1)
#    cs(T) = 1000 * exp(a1 + a2 / T + a3 * ln(T)), where T is in °C
# ==============================================================================
A1_APELBLAT = -69.51
A2_APELBLAT = 368.6
A3_APELBLAT = 16.18

# ==============================================================================
# 3. KINETIC PARAMETERS FOR GROWTH & NUCLEATION (Table 1)
# ==============================================================================
# Face 1 (Length L1) Growth: G1(sigma, L1) = k1 * sigma^g1 * (1 + gamma1 * L1^alpha1)
K1_GROWTH = 6.12e-4         # Rate coefficient for length growth [m / s]
G1_EXPONENT = 1.5           # Supersaturation order for length growth [-]
GAMMA1_GROWTH = 0.05        # Size dependency coefficient for length [um^-alpha1]
ALPHA1_EXPONENT = 0.8       # Size dependency exponent for length [-]

# Face 2 (Width L2) Growth: G2(sigma, L2) = k2 * sigma^g2 * (1 + gamma2 * L2^alpha2)
K2_GROWTH = 1.63e-3         # Rate coefficient for width growth [m / s]
G2_EXPONENT = 1.7           # Supersaturation order for width growth [-]
GAMMA2_GROWTH = 0.03        # Size dependency coefficient for width [um^-alpha2]
ALPHA2_EXPONENT = 0.9       # Size dependency exponent for width [-]

# Secondary Nucleation: B(sigma, eps) = kS * eps * mu_11 * sigma^b1
KS_NUCLEATION = 3.6e5       # Nucleation rate constant [# / (m^3 * s * (W/kg))]
B1_NUCLEATION = 2.0         # Secondary nucleation order [-]

# ==============================================================================
# 4. INITIAL SEED QUADRATURE NODES & WEIGHTS (Table 2)
#    Computed from bivariate log-normal distribution (30% standard deviation)
# ==============================================================================
INITIAL_WEIGHTS = np.array([1.86e10, 6.46e10, 1.281e11], dtype=np.float64)  # [# / m^3]
INITIAL_L1 = np.array([9.74e-5, 9.80e-5, 1.138e-4], dtype=np.float64)       # Length nodes [m]
INITIAL_L2 = np.array([6.85e-5, 5.74e-5, 6.05e-5], dtype=np.float64)        # Width nodes [m]

# Initial population moments
MU00_0 = np.sum(INITIAL_WEIGHTS)
MU10_0 = np.sum(INITIAL_WEIGHTS * INITIAL_L1)
MU01_0 = np.sum(INITIAL_WEIGHTS * INITIAL_L2)
MU11_0 = np.sum(INITIAL_WEIGHTS * INITIAL_L1 * INITIAL_L2)


MEAN_L1_0 = MU10_0 / MU00_0  # ~1.075e-4 m (~107.5 um)
MEAN_L2_0 = MU01_0 / MU00_0  # ~6.026e-5 m (~60.3 um)
ASPECT_RATIO_0 = MEAN_L1_0 / MEAN_L2_0  # ~1.78
