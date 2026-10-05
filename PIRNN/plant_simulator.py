"""
Dynamic Crystallizer Plant Module: 2D Bivariate Population Balance Simulator.

Solves the continuous bivariate moment differential equations and solute mass balance
for 2D plate-like batch cooling crystallization across arbitrary temperature profiles.
Supports batch-to-batch variation, kinetic drift parameters, and realistic sensor noise.
"""

import math
import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, Any, Optional, Tuple, Union

import config
from .temperature_profiles import BaseTemperatureProfile, LinearProfile


def apelblat_solubility(T_celsius: float) -> float:
    """
    Computes saturation concentration cs(T) via Apelblat correlation (Eq. 11).
    Valid and strictly monotonic (dc_s/dT > 0) for T >= 25.0 °C.
    """
    T = max(float(T_celsius), 1.0)
    exponent = config.A1_APELBLAT + (config.A2_APELBLAT / T) + (config.A3_APELBLAT * math.log(T))
    return 1000.0 * math.exp(exponent)


def calculate_supersaturation(c: float, T: float) -> Tuple[float, float]:
    """Computes supersaturation ratio S and relative supersaturation sigma."""
    cs = apelblat_solubility(T)
    if cs <= 1e-9:
        return 1.0, 0.0
    S = float(c / cs)
    sigma = max(float((c - cs) / cs), 0.0)
    return S, sigma


class DynamicCrystallizerPlant:
    """
    Continuous-time plant simulator for 2D batch cooling crystallization.
    Uses closed bivariate population balance moment differential equations
    coupled to solute mass and energy balances.
    """

    def __init__(
        self,
        kV: float = config.K_VOL,
        rho_crystal: float = config.RHO_CRYSTAL,
        k1_base: float = config.K1_GROWTH,
        k2_base: float = config.K2_GROWTH,
        kS_base: float = config.KS_NUCLEATION,
        epsilon: float = config.EPSILON_NOMINAL,
        nucleus_size: float = 1.0e-6,
    ):
        self.kV = float(kV)
        self.rho_c = float(rho_crystal)
        self.k1_base = float(k1_base)
        self.k2_base = float(k2_base)
        self.kS_base = float(kS_base)
        self.epsilon = float(epsilon)
        self.L_nuc = float(nucleus_size)

    def simulate_batch(
        self,
        profile: Union[BaseTemperatureProfile, float] = None,
        duration: Optional[float] = None,
        dt_sample: float = 60.0,
        c0: Optional[float] = None,
        seed_mass_fraction: float = config.SEED_MASS_PERCENT,
        mean_L1_seed: float = config.MEAN_L1_0,
        mean_L2_seed: float = config.MEAN_L2_0,
        epsilon: Optional[float] = None,
        lambda_k1: float = 1.0,
        lambda_k2: float = 1.0,
        lambda_kS: float = 1.0,
        solver_method: str = "BDF",
    ) -> Dict[str, Any]:
        """
        Executes a single batch simulation under the specified temperature profile.

        Parameters:
        - profile: A BaseTemperatureProfile instance or constant cooling rate [°C/s]
        - duration: Total batch time [s] (defaults to profile.duration)
        - dt_sample: Discrete measurement/control sampling period [s] (default: 60 s)
        - c0: Initial solute concentration [kg/m^3] (if None, set to cs(Ts)*(1+0.025))
        - seed_mass_fraction: Seed loading (e.g. 0.02 for 2%)
        - mean_L1_seed, mean_L2_seed: Mean dimensions of seed population [m]
        - epsilon: Stirring energy dissipation [W/kg]
        - lambda_k1, lambda_k2, lambda_kS: Kinetic multipliers (nominal = 1.0)
        - solver_method: 'BDF' or 'Radau' (stiff ODE solver)
        """
        eps = float(epsilon) if epsilon is not None else self.epsilon
        k1 = self.k1_base * float(lambda_k1)
        k2 = self.k2_base * float(lambda_k2)
        kS = self.kS_base * float(lambda_kS)

        # Handle temperature profile input
        if isinstance(profile, (int, float)):
            cr_val = float(profile)
            dur = float(duration) if duration is not None else 12000.0
            T_seed = config.T_SEED_NOMINAL
            T_final = max(T_seed - cr_val * dur, config.T_FINAL)
            temp_profile = LinearProfile(T_seed, T_final, dur)
        elif profile is None:
            dur = float(duration) if duration is not None else 12000.0
            temp_profile = LinearProfile(config.T_SEED_NOMINAL, config.T_FINAL, dur)
        else:
            temp_profile = profile
            dur = temp_profile.duration if duration is None else float(duration)

        T_start = temp_profile.get_temperature(0.0)

        # Realistic metastable seeding concentration if not provided
        if c0 is None:
            c0_val = apelblat_solubility(T_start) * 1.025
        else:
            c0_val = float(c0)

        # Discrete sampling grid
        num_steps = int(math.ceil(dur / dt_sample)) + 1
        t_eval = np.linspace(0.0, dur, num_steps)

        # Initial seed population moments
        # Seed volume = mu11 * kV, seed mass = rho_c * kV * mu11
        # Mass balance: mu11_0 = (c0 * seed_mass_fraction) / (rho_c * kV)
        mu11_0 = (c0_val * float(seed_mass_fraction)) / (self.rho_c * self.kV)
        mu00_0 = mu11_0 / (mean_L1_seed * mean_L2_seed)
        mu10_0 = mu00_0 * mean_L1_seed
        mu01_0 = mu00_0 * mean_L2_seed
        mu20_0 = mu00_0 * (mean_L1_seed ** 2) * 1.09
        mu02_0 = mu00_0 * (mean_L2_seed ** 2) * 1.09

        # State vector: [mu00, mu10, mu01, mu11, mu20, mu02, c, T]
        y0 = np.array([mu00_0, mu10_0, mu01_0, mu11_0, mu20_0, mu02_0, c0_val, T_start], dtype=np.float64)

        L_nuc = self.L_nuc

        def derivatives(t: float, y: np.ndarray) -> np.ndarray:
            mu00 = max(y[0], 1.0e-5)
            mu10 = max(y[1], 1.0e-5)
            mu01 = max(y[2], 1.0e-5)
            mu11 = max(y[3], 1.0e-5)
            mu20 = max(y[4], 1.0e-5)
            mu02 = max(y[5], 1.0e-5)
            c = y[6]
            T = y[7]

            _, sigma = calculate_supersaturation(c, T)
            sig = max(sigma, 0.0)

            # Mean crystal dimensions
            L1 = max(mu10 / mu00, 1.0e-6)
            L2 = max(mu01 / mu00, 1.0e-6)

            # Face growth rates G1 and G2 (Eq. 2)
            L1_um = L1 * 1.0e6
            L2_um = L2 * 1.0e6
            f1 = 1.0 + config.GAMMA1_GROWTH * (L1_um ** config.ALPHA1_EXPONENT)
            f2 = 1.0 + config.GAMMA2_GROWTH * (L2_um ** config.ALPHA2_EXPONENT)
            G1 = k1 * (sig ** config.G1_EXPONENT) * f1
            G2 = k2 * (sig ** config.G2_EXPONENT) * f2

            # Contact nucleation rate (Eq. 1)
            B = float(kS * eps * mu11 * (sig ** config.B1_NUCLEATION)) if sig > 1e-12 else 0.0

            # Closed bivariate moment differential equations
            dmu00_dt = B
            dmu10_dt = G1 * mu00 + B * L_nuc
            dmu01_dt = G2 * mu00 + B * L_nuc
            dmu11_dt = G1 * mu01 + G2 * mu10 + B * (L_nuc ** 2)
            dmu20_dt = 2.0 * G1 * mu10 + B * (L_nuc ** 2)
            dmu02_dt = 2.0 * G2 * mu01 + B * (L_nuc ** 2)

            # Solute mass balance (Eq. 8): dc/dt = - rho_c * kV * dmu11/dt
            dc_dt = - self.rho_c * self.kV * dmu11_dt

            # Temperature profile cooling schedule
            cr_inst = temp_profile.get_cooling_rate(t)
            dT_dt = -cr_inst

            return np.array([dmu00_dt, dmu10_dt, dmu01_dt, dmu11_dt, dmu20_dt, dmu02_dt, dc_dt, dT_dt], dtype=np.float64)

        # Stiff ODE solution via BDF
        sol = solve_ivp(
            derivatives,
            (0.0, dur),
            y0,
            method=solver_method,
            t_eval=t_eval,
            rtol=1.0e-5,
            atol=1.0e-7,
        )

        time_vec = sol.t
        mu00_traj = sol.y[0, :]
        mu10_traj = sol.y[1, :]
        mu01_traj = sol.y[2, :]
        mu11_traj = sol.y[3, :]
        mu20_traj = sol.y[4, :]
        mu02_traj = sol.y[5, :]
        c_traj = sol.y[6, :]
        T_traj = sol.y[7, :]

        # Physical mean lengths and aspect ratio
        mean_L1 = mu10_traj / np.maximum(mu00_traj, 1e-9)
        mean_L2 = mu01_traj / np.maximum(mu00_traj, 1e-9)
        aspect_ratio = mean_L1 / np.maximum(mean_L2, 1e-9)

        # Supersaturation metrics
        cs_traj = np.array([apelblat_solubility(T) for T in T_traj])
        S_traj = c_traj / np.maximum(cs_traj, 1e-9)
        sigma_traj = np.maximum((c_traj - cs_traj) / np.maximum(cs_traj, 1e-9), 0.0)

        # Dynamic rates along trajectory
        cr_traj = np.array([temp_profile.get_cooling_rate(t) for t in time_vec])

        L1_um = mean_L1 * 1.0e6
        L2_um = mean_L2 * 1.0e6
        f1_traj = 1.0 + config.GAMMA1_GROWTH * (L1_um ** config.ALPHA1_EXPONENT)
        f2_traj = 1.0 + config.GAMMA2_GROWTH * (L2_um ** config.ALPHA2_EXPONENT)
        G1_traj = k1 * (sigma_traj ** config.G1_EXPONENT) * f1_traj
        G2_traj = k2 * (sigma_traj ** config.G2_EXPONENT) * f2_traj
        B_traj = kS * eps * np.maximum(mu11_traj, 0.0) * (sigma_traj ** config.B1_NUCLEATION)

        trajectory = {
            "time": time_vec,
            "T": T_traj,
            "cooling_rate": cr_traj,
            "c": c_traj,
            "cs": cs_traj,
            "S": S_traj,
            "sigma": sigma_traj,
            "mu00": mu00_traj,
            "mu10": mu10_traj,
            "mu01": mu01_traj,
            "mu20": mu20_traj,
            "mu11": mu11_traj,
            "mu02": mu02_traj,
            "mean_L1": mean_L1,
            "mean_L2": mean_L2,
            "aspect_ratio": aspect_ratio,
            "growth_rate_L1": G1_traj,
            "growth_rate_L2": G2_traj,
            "nucleation_rate": B_traj,
            "epsilon": eps,
            "lambda_k1": lambda_k1,
            "lambda_k2": lambda_k2,
            "lambda_kS": lambda_kS,
            "profile_info": temp_profile.to_dict(),
        }
        return trajectory

    @staticmethod
    def inject_sensor_noise(
        trajectory: Dict[str, Any],
        std_T: float = 0.10,            # °C thermocouple noise
        std_c: float = 0.30,            # kg/m^3 ATR-FTIR noise
        rel_std_size: float = 0.02,     # 2% size measurement noise
        rel_std_moments: float = 0.05,  # 5% moments noise
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Applies realistic plant sensor noise to clean simulation trajectory.
        Preserves clean ground-truth values while adding noisy measurement fields.
        """
        rng = np.random.default_rng(seed)
        N = len(trajectory["time"])

        noisy = dict(trajectory)
        noisy["T_meas"] = trajectory["T"] + rng.normal(0.0, std_T, size=N)
        noisy["c_meas"] = np.maximum(trajectory["c"] + rng.normal(0.0, std_c, size=N), 0.0)

        # Sizes with 2% relative noise
        noisy["mean_L1_meas"] = np.maximum(trajectory["mean_L1"] * (1.0 + rng.normal(0.0, rel_std_size, size=N)), 1e-7)
        noisy["mean_L2_meas"] = np.maximum(trajectory["mean_L2"] * (1.0 + rng.normal(0.0, rel_std_size, size=N)), 1e-7)
        noisy["aspect_ratio_meas"] = noisy["mean_L1_meas"] / np.maximum(noisy["mean_L2_meas"], 1e-7)

        # Moments with 5% relative noise
        moment_keys = ["mu00", "mu10", "mu01", "mu20", "mu11", "mu02"]
        for k in moment_keys:
            if k in trajectory:
                noisy[f"{k}_meas"] = np.maximum(trajectory[k] * (1.0 + rng.normal(0.0, rel_std_moments, size=N)), 0.0)

        return noisy
