"""
Dynamic Crystallizer Plant Module: 2D Bivariate QMOM Simulator.

Solves the 11 stiff differential equations of the 2D Quadrature Method of Moments
(QMOM) system for arbitrary continuous or piecewise-defined temperature profiles.
Supports batch-to-batch variation, kinetic drift parameters, and realistic sensor noise.
"""

import math
import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, Any, Optional, Tuple, Union

import config
from .temperature_profiles import BaseTemperatureProfile, LinearProfile


def apelblat_solubility(T_celsius: float) -> float:
    """Computes saturation concentration cs(T) via Apelblat correlation (Eq. 11)."""
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
    """

    def __init__(
        self,
        kV: float = config.K_VOL,
        rho_crystal: float = config.RHO_CRYSTAL,
        k1_base: float = config.K1_GROWTH,
        k2_base: float = config.K2_GROWTH,
        kS_base: float = config.KS_NUCLEATION,
        epsilon: float = config.EPSILON_NOMINAL,
    ):
        self.kV = float(kV)
        self.rho_c = float(rho_crystal)
        self.k1_base = float(k1_base)
        self.k2_base = float(k2_base)
        self.kS_base = float(kS_base)
        self.epsilon = float(epsilon)

    def simulate_batch(
        self,
        profile: Union[BaseTemperatureProfile, float] = None,
        duration: Optional[float] = None,
        dt_sample: float = 60.0,
        c0: float = config.C0_SOLUTE,
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
        - dt_sample: Discrete measurement/control sampling period Delta [s] (default: 60 s)
        - c0: Initial solute concentration [kg/m^3]
        - seed_mass_fraction: Seed loading (e.g. 0.02 for 2%)
        - mean_L1_seed, mean_L2_seed: Mean dimensions of seed population [m]
        - epsilon: Stirring energy dissipation [W/kg]
        - lambda_k1, lambda_k2, lambda_kS: Kinetic multipliers (nominal = 1.0)
        - solver_method: 'BDF' (stiff ODE solver, ode15s twin)
        """
        eps = float(epsilon) if epsilon is not None else self.epsilon
        k1 = self.k1_base * float(lambda_k1)
        k2 = self.k2_base * float(lambda_k2)
        kS = self.kS_base * float(lambda_kS)

        # Handle temperature profile input
        if isinstance(profile, (int, float)):
            # Passed a constant cooling rate
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

        # Discrete sampling grid
        num_steps = int(math.ceil(dur / dt_sample)) + 1
        t_eval = np.linspace(0.0, dur, num_steps)

        # Initialize seed nodes and weights (Table 2 of Szilágyi & Lakatos 2015)
        w0 = np.array(config.INITIAL_WEIGHTS, dtype=np.float64)
        L1_0 = np.array(config.INITIAL_L1, dtype=np.float64)
        L2_0 = np.array(config.INITIAL_L2, dtype=np.float64)

        # Scale seed dimensions and weights by physical mass
        scale_L1 = mean_L1_seed / config.MEAN_L1_0
        scale_L2 = mean_L2_seed / config.MEAN_L2_0
        L1_0 = L1_0 * scale_L1
        L2_0 = L2_0 * scale_L2

        # In Table 2, seed volume = sum(w * L1 * L2 * kV) corresponds to 2% seed mass
        vol_factor = (scale_L1 * scale_L2)
        w0 = w0 * (float(seed_mass_fraction) / config.SEED_MASS_PERCENT) / max(vol_factor, 1e-6)

        mu11_0 = float(np.sum(w0 * L1_0 * L2_0))
        T_start = temp_profile.get_temperature(0.0)

        # State vector: [L1_1, L1_2, L1_3, L2_1, L2_2, L2_3, w1, w2, w3, c, T]
        y0 = np.concatenate([L1_0, L2_0, w0, [c0], [T_start]])

        def derivatives(t: float, y: np.ndarray) -> np.ndarray:
            L1 = np.maximum(y[0:3], 1.0e-7)
            L2 = np.maximum(y[3:6], 1.0e-7)
            w = np.maximum(y[6:9], 1.0e-5)
            c = y[9]
            T = y[10]

            _, sigma = calculate_supersaturation(c, T)
            sig = max(sigma, 0.0)

            # Growth rates G1 and G2 (Eq. 2)
            L1_um = L1 * 1.0e6
            L2_um = L2 * 1.0e6
            f1 = 1.0 + config.GAMMA1_GROWTH * (L1_um ** config.ALPHA1_EXPONENT)
            f2 = 1.0 + config.GAMMA2_GROWTH * (L2_um ** config.ALPHA2_EXPONENT)
            G1 = k1 * (sig ** config.G1_EXPONENT) * f1
            G2 = k2 * (sig ** config.G2_EXPONENT) * f2

            # Cross-moment mu11 via solute mass balance
            mu11 = max((c0 - c) / (self.rho_c * self.kV) + mu11_0, 0.0)

            # Contact nucleation rate (Eq. 1)
            B = float(kS * eps * mu11 * (sig ** config.B1_NUCLEATION)) if (sig > 1e-12 and mu11 > 0.0) else 0.0

            # Weight derivatives via Cramer's rule on M * dw/dt = [B, 0, 0]^T
            c1 = L1[1] * L2[2] - L1[2] * L2[1]
            c2 = L1[2] * L2[0] - L1[0] * L2[2]
            c3 = L1[0] * L2[1] - L1[1] * L2[0]
            det_M = c1 + c2 + c3
            if abs(det_M) > 1.0e-30:
                inv_det = 1.0 / det_M
                dw_dt = np.array([B * c1 * inv_det, B * c2 * inv_det, B * c3 * inv_det], dtype=np.float64)
            else:
                dw_dt = np.array([B / 3.0, B / 3.0, B / 3.0], dtype=np.float64)

            # Overall volumetric growth rate RV (Eq. 9)
            dmu11_dt = np.sum(w * (G1 * L2 + G2 * L1))
            RV = self.kV * dmu11_dt

            # Solute mass balance (Eq. 8)
            dc_dt = - self.rho_c * RV

            # Temperature schedule: guided by analytical profile derivative
            cr_inst = temp_profile.get_cooling_rate(t)
            dT_dt = -cr_inst

            return np.concatenate([G1, G2, dw_dt, [dc_dt], [dT_dt]])

        # Solve with BDF
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
        L1_traj = sol.y[0:3, :]
        L2_traj = sol.y[3:6, :]
        w_traj = sol.y[6:9, :]
        c_traj = sol.y[9, :]
        T_traj = sol.y[10, :]

        # Calculate exact bivariate moments
        mu00_traj = np.sum(w_traj, axis=0)
        mu10_traj = np.sum(w_traj * L1_traj, axis=0)
        mu01_traj = np.sum(w_traj * L2_traj, axis=0)
        mu20_traj = np.sum(w_traj * (L1_traj ** 2), axis=0)
        mu02_traj = np.sum(w_traj * (L2_traj ** 2), axis=0)
        mu11_traj = (c0 - c_traj) / (self.rho_c * self.kV) + mu11_0
        mu30_traj = np.sum(w_traj * (L1_traj ** 3), axis=0)
        mu21_traj = np.sum(w_traj * (L1_traj ** 2) * L2_traj, axis=0)
        mu12_traj = np.sum(w_traj * L1_traj * (L2_traj ** 2), axis=0)
        mu03_traj = np.sum(w_traj * (L2_traj ** 3), axis=0)

        # Physical mean lengths and aspect ratio
        w_pos = np.maximum(w_traj, 0.0)
        mu00_pos = np.maximum(np.sum(w_pos, axis=0), 1e-9)
        mean_L1 = np.sum(w_pos * L1_traj, axis=0) / mu00_pos
        mean_L2 = np.sum(w_pos * L2_traj, axis=0) / mu00_pos
        aspect_ratio = mean_L1 / np.maximum(mean_L2, 1e-9)

        # Supersaturation metrics
        cs_traj = np.array([apelblat_solubility(T) for T in T_traj])
        S_traj = c_traj / np.maximum(cs_traj, 1e-9)
        sigma_traj = np.maximum((c_traj - cs_traj) / np.maximum(cs_traj, 1e-9), 0.0)

        # Growth and nucleation rates
        cr_traj = np.array([temp_profile.get_cooling_rate(t) for t in time_vec])
        B_traj = np.array([
            kS * eps * max(mu11_traj[i], 0.0) * (sigma_traj[i] ** config.B1_NUCLEATION)
            if sigma_traj[i] > 1e-12 else 0.0
            for i in range(len(time_vec))
        ])

        # Pack clean states into dictionary
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
            "mu30": mu30_traj,
            "mu21": mu21_traj,
            "mu12": mu12_traj,
            "mu03": mu03_traj,
            "mean_L1": mean_L1,
            "mean_L2": mean_L2,
            "aspect_ratio": aspect_ratio,
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
        moment_keys = ["mu00", "mu10", "mu01", "mu20", "mu11", "mu02", "mu30", "mu21", "mu12", "mu03"]
        for k in moment_keys:
            noisy[f"{k}_meas"] = np.maximum(trajectory[k] * (1.0 + rng.normal(0.0, rel_std_moments, size=N)), 0.0)

        return noisy
