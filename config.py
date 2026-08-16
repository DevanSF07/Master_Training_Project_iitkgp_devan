import os
import torch

class Config:
    # Hardware & Device Setup
    DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42

    # CSTR Physical Process Parameters (Table 2 in Paper)
    F_0 = 5.0                # m^3/h (Feed volumetric flow rate)
    V = 1.0                  # m^3 (Reactor volume)
    k_0_t0 = 8.46e6          # m^3/(kmol*h) (Pre-exponential factor)
    E = 5.0e4                # kJ/kmol (Activation energy)
    R = 8.314                # kJ/(kmol*K) (Gas constant)
    T_0 = 300.0              # K (Feed temperature)
    DELTA_H = -1.15e4        # kJ/kmol (Enthalpy change of reaction)
    RHO_L = 1000.0           # kg/m^3 (Liquid density)
    C_P = 0.231              # kJ/(kg*K) (Heat capacity)

    # Steady State Values (Operating Point)
    C_A0s = 4.0              # kmol/m^3 (Steady state inlet concentration)
    Q_s = 0.0                # kJ/h (Steady state heat input)
    C_As = 1.95              # kmol/m^3 (Steady state reactor concentration)
    T_s = 402.0              # K (Steady state reactor temperature)

    # Deviation Physical Constraints
    DELTA_C_A0_MAX = 3.5     # kmol/m^3 (|ΔC_A0| <= 3.5)
    DELTA_Q_MAX = 5.0e5      # kJ/h (|ΔQ| <= 5e5)

    # Time Horizon Settings
    DELTA = 1.0e-2           # h (Sampling period Δ = 0.01 h = 36 s)
    H_C = 1.0e-3             # h (Internal numerical step h_c = 0.001 h, 10 sub-steps per Δ)
    SUB_STEPS = int(DELTA / H_C)  # 10 sub-steps

    # PIRNN Hyperparameters
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    NUM_COLLOCATION_INITIAL = 200    # Initial state pairs
    NUM_COLLOCATION_INPUTS = 400     # Manipulated input pairs
    TOTAL_COLLOCATION = NUM_COLLOCATION_INITIAL * NUM_COLLOCATION_INPUTS  # 80,000 collocation trajectories
    BATCH_SIZE = 2048
    LEARNING_RATE = 1.0e-3
    ETA_PHYSICS = 0.1        # Weight coefficient η for physics-informed loss term
    EPOCHS = 10
    PATIENCE = 5             # Early stopping patience

    # Error-Triggered Mechanism
    ERROR_THRESHOLD_E_T = 1.0e-4   # E_T = 1e-4

    # Paths
    CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
    PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
    NOMINAL_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "pirnn_nominal.pth")

os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(Config.PLOT_DIR, exist_ok=True)
