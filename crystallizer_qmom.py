"""
2D Bivariate Batch Cooling Crystallization Simulator: Pure Quadrature Method of Moments (QMOM).

This module implements the mathematical model of seeded batch cooling crystallization
of plate-like crystals described by:
    Botond Szilágyi and Béla G. Lakatos (2015)
    "Batch Cooling Crystallization of Plate-like Crystals: A Simulation Study"
    Periodica Polytechnica Chemical Engineering, 59(2), pp. 151-158.

Mathematical Architecture:
1. Apelblat solubility model and supersaturation driving force (Eq. 11).
2. 2D Size-dependent face growth rates G1(sigma, L1) and G2(sigma, L2) with fractional
   size-dependency exponents alpha1 = 0.8, alpha2 = 0.9 (Eq. 2).
3. Secondary contact nucleation rate B(sigma, epsilon, mu11) (Eq. 1).
4. Solute mass balance dc/dt = - rho_c * RV = - rho_c * kV * d(mu11)/dt (Eq. 8, 9).
5. Quadrature Method of Moments (QMOM) formulation (Eq. 15, 16, 17):
   Because alpha1, alpha2 are non-integers, standard MOM is unclosed.
   QMOM approximates the 2D population density with N_q = 3 quadrature nodes (w_i, L1_i, L2_i),
   closing the differential system into a stiff ODE/DAE solved via Backward Differentiation
   Formulas (BDF / MATLAB ode15s equivalent) or Runge-Kutta (RK45 / MATLAB ode45 equivalent).

Author: Devan Singh Faujdar
Master Training Project, IIT Kharagpur
"""

import math
import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, Any, Tuple, Optional

import config


def calculate_solubility(temperature_celsius: float) -> float:
    """
    Computes the saturation concentration cs(T) using the Apelblat correlation (Eq. 11).

    cs(T) = 1000 * exp(a1 + a2/T + a3 * ln(T))  [kg / m^3, T in °C]
    Optimized for high-speed scalar execution during stiff ODE integration.
    """
    T = float(temperature_celsius)
    if T < 1.0:
        T = 1.0
    exponent = config.A1_APELBLAT + (config.A2_APELBLAT / T) + (config.A3_APELBLAT * math.log(T))
    return 1000.0 * math.exp(exponent)


def calculate_supersaturation(concentration: float, temperature_celsius: float) -> Tuple[float, float]:
    """
    Calculates the supersaturation ratio S and relative supersaturation sigma.

    S = c / cs(T)
    sigma = max(0, (c - cs(T)) / cs(T)) = max(0, S - 1)
    """
    cs = calculate_solubility(temperature_celsius)
    if cs <= 1e-9:
        return 1.0, 0.0
    S = float(concentration / cs)
    sigma = float(max((concentration - cs) / cs, 0.0))
    return S, sigma


def calculate_growth_rates(
    sigma: float,
    L1: Any,
    L2: Any,
    k1: float = config.K1_GROWTH,
    k2: float = config.K2_GROWTH,
) -> Tuple[Any, Any]:
    """
    Computes 2D size-dependent crystal growth rates for Length (L1) and Width (L2) (Eq. 2).

    G1(sigma, L1) = k1 * sigma^g1 * (1 + gamma1 * (L1 * 1e6)^alpha1)  [m/s]
    G2(sigma, L2) = k2 * sigma^g2 * (1 + gamma2 * (L2 * 1e6)^alpha2)  [m/s]
    """
    sig = np.maximum(np.asarray(sigma, dtype=np.float64), 0.0)
    L1_um = np.maximum(np.asarray(L1, dtype=np.float64) * 1.0e6, 0.0)
    L2_um = np.maximum(np.asarray(L2, dtype=np.float64) * 1.0e6, 0.0)

    f1 = 1.0 + config.GAMMA1_GROWTH * (L1_um ** config.ALPHA1_EXPONENT)
    f2 = 1.0 + config.GAMMA2_GROWTH * (L2_um ** config.ALPHA2_EXPONENT)

    G1 = k1 * (sig ** config.G1_EXPONENT) * f1
    G2 = k2 * (sig ** config.G2_EXPONENT) * f2
    return G1, G2


def calculate_nucleation_rate(
    sigma: float,
    epsilon: float,
    mu11: float,
    kS: float = config.KS_NUCLEATION,
) -> float:
    """
    Computes the secondary contact nucleation rate B(sigma, epsilon, mu11) (Eq. 1).

    B(sigma, epsilon) = kS * epsilon * mu11 * sigma^b1  [#/m^3/s]
    """
    if sigma <= 1e-12 or mu11 <= 0.0:
        return 0.0
    B = float(kS * epsilon * mu11 * (sigma ** config.B1_NUCLEATION))
    return B


class BatchCrystallizerQMOM:
    """
    2D Seeded Batch Cooling Crystallization Simulator using the Quadrature Method of Moments (QMOM).
    
    Exclusively implements the closed QMOM system of Szilágyi & Lakatos (2015),
    tracking quadrature node trajectories and evaluating bivariate moments directly.
    """

    def __init__(
        self,
        kV: float = config.K_VOL,
        rho_crystal: float = config.RHO_CRYSTAL,
        k1: float = config.K1_GROWTH,
        k2: float = config.K2_GROWTH,
        kS: float = config.KS_NUCLEATION,
        epsilon: float = config.EPSILON_NOMINAL,
    ):
        self.kV = float(kV)
        self.rho_c = float(rho_crystal)
        self.k1 = float(k1)
        self.k2 = float(k2)
        self.kS = float(kS)
        self.epsilon = float(epsilon)

    def simulate(
        self,
        T_seed: float = config.T_SEED_NOMINAL,
        T_final: float = config.T_FINAL,
        cooling_rate: float = config.COOLING_RATE_NOMINAL,
        epsilon: Optional[float] = None,
        c_init: Optional[float] = None,
        c0: Optional[float] = None,
        weights_init: Optional[np.ndarray] = None,
        L1_init: Optional[np.ndarray] = None,
        L2_init: Optional[np.ndarray] = None,
        mean_L1_seed: Optional[float] = None,
        mean_L2_seed: Optional[float] = None,
        seed_mass_fraction: Optional[float] = None,
        t_span: Optional[Tuple[float, float]] = None,
        t_eval: Optional[np.ndarray] = None,
        num_points: int = 150,
        solver_method: str = "BDF",
    ) -> Dict[str, Any]:
        """
        Executes dynamic batch crystallization simulation using the 2D Quadrature Method of Moments (QMOM).
        
        Parameters:
        - T_seed: Seeding temperature [°C]
        - T_final: Batch final temperature [°C]
        - cooling_rate: Linear cooling rate cr [°C/s]
        - epsilon: Specific power dissipation / stirring energy [W/kg]
        - c0 / c_init: Initial solute concentration [kg/m^3]
        - mean_L1_seed, mean_L2_seed: Mean dimensions of seed crystal population [m]
        - seed_mass_fraction: Seed loading fraction (e.g. 0.02 for 2%)
        - solver_method: 'BDF' (MATLAB ode15s equivalent) or 'RK45' (MATLAB ode45 equivalent)
        """
        eps = float(epsilon) if epsilon is not None else self.epsilon
        cr = float(cooling_rate)
        c_start = float(c0) if c0 is not None else (float(c_init) if c_init is not None else config.C0_SOLUTE)

        if t_span is None:
            t_end = (T_seed - T_final) / cr
            t_span = (0.0, float(t_end))
        else:
            t_end = t_span[1]

        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], num_points)
        else:
            t_eval = np.asarray(t_eval, dtype=np.float64)
            t_span = (min(float(t_span[0]), float(t_eval[0])), max(float(t_span[1]), float(t_eval[-1])))

        # Base seed weights and nodes (Table 2)
        if weights_init is not None and L1_init is not None and L2_init is not None:
            w0 = np.array(weights_init, dtype=np.float64)
            L1_0 = np.array(L1_init, dtype=np.float64)
            L2_0 = np.array(L2_init, dtype=np.float64)
        else:
            w0 = np.array(config.INITIAL_WEIGHTS, dtype=np.float64)
            L1_0 = np.array(config.INITIAL_L1, dtype=np.float64)
            L2_0 = np.array(config.INITIAL_L2, dtype=np.float64)

            # Support seed size / mass scaling for Figs 4 & 5
            if mean_L1_seed is not None and mean_L2_seed is not None:
                s1 = mean_L1_seed / config.MEAN_L1_0
                s2 = mean_L2_seed / config.MEAN_L2_0
                L1_0 = L1_0 * s1
                L2_0 = L2_0 * s2
            if seed_mass_fraction is not None:
                # Table 2 corresponds to 2% seed mass (config.SEED_MASS_PERCENT = 0.02)
                w0 = w0 * (float(seed_mass_fraction) / config.SEED_MASS_PERCENT)

        mu00_0 = float(np.sum(w0))
        mu11_0 = float(np.sum(w0 * L1_0 * L2_0))

        # Pure QMOM kinetic rate constants (Table 1)
        k1_eff = self.k1
        k2_eff = self.k2

        # State vector y (11 states matching Szilágyi & Lakatos 2015):
        # y[0:3]   = Length abscissas (L1_1, L1_2, L1_3)
        # y[3:6]   = Width abscissas  (L2_1, L2_2, L2_3)
        # y[6:9]   = Weights          (w_1, w_2, w_3)
        # y[9]     = Solute concentration (c)
        # y[10]    = Temperature      (T)
        y0 = np.concatenate([L1_0, L2_0, w0, [c_start], [T_seed]])

        def ode_derivatives(t: float, y: np.ndarray) -> np.ndarray:
            L1 = np.maximum(y[0:3], 1.0e-7)
            L2 = np.maximum(y[3:6], 1.0e-7)
            w  = np.maximum(y[6:9], 1.0e-5)
            c  = y[9]
            T  = y[10]

            # Supersaturation from dynamic concentration c and temperature T
            _, sigma = calculate_supersaturation(c, T)

            # Quadrature node growth rates: G1(sigma, L1), G2(sigma, L2) (Eq. 2)
            G1, G2 = calculate_growth_rates(sigma, L1, L2, k1_eff, k2_eff)

            # Crystal cross-moment mu11 via solute mass balance identity to eliminate negative weight drift
            mu11 = (c_start - c) / (self.rho_c * self.kV) + mu11_0

            # Secondary contact nucleation rate: B = kS * eps * mu11 * sigma^b1 (Eq. 1)
            B = calculate_nucleation_rate(sigma, eps, mu11, self.kS)

            # Analytical solution of linear moment system M * dw/dt = [B, 0, 0]^T via Cramer's rule (Eq. 17)
            # Avoids matrix allocation and linear solver overhead inside stiff ODE integrator:
            c1 = L1[1] * L2[2] - L1[2] * L2[1]
            c2 = L1[2] * L2[0] - L1[0] * L2[2]
            c3 = L1[0] * L2[1] - L1[1] * L2[0]
            det_M = c1 + c2 + c3
            if abs(det_M) > 1.0e-30:
                inv_det = 1.0 / det_M
                dw_dt = np.array([B * c1 * inv_det, B * c2 * inv_det, B * c3 * inv_det], dtype=np.float64)
            else:
                dw_dt = np.array([B / 3.0, B / 3.0, B / 3.0], dtype=np.float64)

            # Overall crystal volume growth rate RV per unit suspension volume (Eq. 9):
            # Pure volumetric growth without artificial nucleation subtraction
            dmu11_dt = np.sum(w * (G1 * L2 + G2 * L1))
            RV = self.kV * dmu11_dt

            # Solute mass balance: dc/dt = - rho_c * RV (Eq. 8)
            dc_dt = - self.rho_c * RV

            # Temperature schedule: dT/dt = -cr (Eq. 10) until T_final
            dT_dt = -cr if T > T_final else 0.0

            return np.concatenate([G1, G2, dw_dt, [dc_dt], [dT_dt]])

        # Solve ODE with selected method (BDF = ode15s, RK45 = ode45)
        sol = solve_ivp(
            ode_derivatives,
            t_span,
            y0,
            method=solver_method,
            t_eval=t_eval,
            rtol=1.0e-5,
            atol=1.0e-7,
        )

        t_res = sol.t
        L1_traj = sol.y[0:3, :]
        L2_traj = sol.y[3:6, :]
        w_traj = sol.y[6:9, :]
        c_traj = sol.y[9, :]
        T_traj = sol.y[10, :]

        # Solubility along trajectory
        cs_traj = np.array([calculate_solubility(T) for T in T_traj])
        S_traj = c_traj / cs_traj
        sigma_traj = np.maximum(0.0, (c_traj - cs_traj) / cs_traj)

        # Pure QMOM moments: mu_km(t) = sum_i(w_i(t) * L1_i(t)^k * L2_i(t)^m)
        mu00_traj = np.sum(w_traj, axis=0)
        mu10_traj = np.sum(w_traj * L1_traj, axis=0)
        mu01_traj = np.sum(w_traj * L2_traj, axis=0)
        mu20_traj = np.sum(w_traj * (L1_traj ** 2), axis=0)
        mu02_traj = np.sum(w_traj * (L2_traj ** 2), axis=0)
        # Exact mass balance identity for mu11 to prevent negative weight numerical artifact:
        mu11_traj = (c_traj[0] - c_traj) / (self.rho_c * self.kV) + mu11_0
        mu30_traj = np.sum(w_traj * (L1_traj ** 3), axis=0)
        mu21_traj = np.sum(w_traj * (L1_traj ** 2) * L2_traj, axis=0)
        mu12_traj = np.sum(w_traj * L1_traj * (L2_traj ** 2), axis=0)

        # Pure QMOM characteristic particle dimensions (Eq. 17 & Section 4):
        # Uses physical non-negative weights to prevent catastrophic cancellation at high nucleation:
        w_phys = np.maximum(w_traj, 0.0)
        mu00_phys = np.sum(w_phys, axis=0)
        mu10_phys = np.sum(w_phys * L1_traj, axis=0)
        mu01_phys = np.sum(w_phys * L2_traj, axis=0)
        mean_L1_traj = mu10_phys / np.maximum(mu00_phys, 1.0e-9)
        mean_L2_traj = mu01_phys / np.maximum(mu00_phys, 1.0e-9)
        aspect_ratio_traj = mean_L1_traj / np.maximum(mean_L2_traj, 1.0e-9)

        # Dynamic rates across time
        B_traj = np.zeros_like(t_res)
        RV_traj = np.zeros_like(t_res)
        for idx in range(len(t_res)):
            sig = sigma_traj[idx]
            g1, g2 = calculate_growth_rates(sig, L1_traj[:, idx], L2_traj[:, idx], k1_eff, k2_eff)
            mu11_val = mu11_traj[idx]
            b_nuc = calculate_nucleation_rate(sig, eps, mu11_val, self.kS)
            dmu11_idx = np.sum(w_traj[:, idx] * (g1 * L2_traj[:, idx] + g2 * L1_traj[:, idx]))
            rv = self.kV * dmu11_idx
            RV_traj[idx] = rv
            B_traj[idx] = b_nuc

        return {
            "time": t_res,
            "temperature": T_traj,
            "concentration": c_traj,
            "solubility": cs_traj,
            "supersaturation_ratio": S_traj,
            "S_ratio": S_traj,
            "relative_supersaturation": sigma_traj,
            "sigma": sigma_traj,
            "weights": w_traj,
            "L1_nodes": L1_traj,
            "L2_nodes": L2_traj,
            "mean_L1": mean_L1_traj,
            "mean_L2": mean_L2_traj,
            "aspect_ratio": aspect_ratio_traj,
            "mu00": mu00_traj,
            "mu10": mu10_traj,
            "mu01": mu01_traj,
            "mu20": mu20_traj,
            "mu11": mu11_traj,
            "mu02": mu02_traj,
            "mu30": mu30_traj,
            "mu21": mu21_traj,
            "mu12": mu12_traj,
            "nucleation_rate": B_traj,
            "volumetric_growth_rate": RV_traj,
            "RV": RV_traj,
            "success": sol.success,
            "message": sol.message,
            "solver": solver_method,
            "nfev": sol.nfev,
            "num_steps": len(sol.t),
        }


# Backwards compatibility alias
BatchCrystallizerMOM = BatchCrystallizerQMOM


if __name__ == "__main__":
    print("Testing 2D Seeded Batch Cooling Crystallizer (Pure QMOM)...")
    sim = BatchCrystallizerQMOM()
    res = sim.simulate()
    print("Nominal QMOM Simulation Successful!")
    print(f"Time: 0 to {res['time'][-1]:.0f} s")
    print(f"Concentration: {res['concentration'][0]:.2f} -> {res['concentration'][-1]:.2f} kg/m^3")
    print(f"Final <L1>: {res['mean_L1'][-1]*1e4:.2f} x 10^-4 m (Paper: 4.75)")
    print(f"Final <L2>: {res['mean_L2'][-1]*1e4:.2f} x 10^-4 m (Paper: 2.42)")
    print(f"Final Aspect Ratio: {res['aspect_ratio'][-1]:.2f} (Paper: 1.96)")
    print(f"Final mu11: {res['mu11'][-1]:.1f} m^2/m^3 (Paper: 22000)")
