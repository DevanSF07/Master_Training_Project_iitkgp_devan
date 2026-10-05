"""
Differentiable PyTorch QMOM Physics Module.

Implements the continuous differential equations and conservation laws of the 2D QMOM system
in pure, vectorized, autograd-compatible PyTorch:
1. Apelblat solubility correlation cs(T) and relative supersaturation sigma(c, T).
2. Size-dependent crystal face growth rates G1 and G2.
3. Secondary contact nucleation rate B.
4. Solute mass balance: dc/dt = - rho_c * kV * d(mu11)/dt.
5. Collocation ODE residual loss operator.
"""

import math
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional

import config


class DifferentiableQMOMPhysics(nn.Module):
    """
    Evaluates the continuous-time physics differential operator and residuals in PyTorch.
    """

    def __init__(
        self,
        kV: float = config.K_VOL,
        rho_c: float = config.RHO_CRYSTAL,
        k1: float = config.K1_GROWTH,
        k2: float = config.K2_GROWTH,
        kS: float = config.KS_NUCLEATION,
        gamma1: float = config.GAMMA1_GROWTH,
        gamma2: float = config.GAMMA2_GROWTH,
        alpha1: float = config.ALPHA1_EXPONENT,
        alpha2: float = config.ALPHA2_EXPONENT,
        g1: float = config.G1_EXPONENT,
        g2: float = config.G2_EXPONENT,
        b1: float = config.B1_NUCLEATION,
        a1: float = config.A1_APELBLAT,
        a2: float = config.A2_APELBLAT,
        a3: float = config.A3_APELBLAT,
    ):
        super().__init__()
        self.kV = float(kV)
        self.rho_c = float(rho_c)
        self.k1 = float(k1)
        self.k2 = float(k2)
        self.kS = float(kS)
        self.gamma1 = float(gamma1)
        self.gamma2 = float(gamma2)
        self.alpha1 = float(alpha1)
        self.alpha2 = float(alpha2)
        self.g1 = float(g1)
        self.g2 = float(g2)
        self.b1 = float(b1)
        self.a1 = float(a1)
        self.a2 = float(a2)
        self.a3 = float(a3)

    def calculate_solubility(self, T_celsius: torch.Tensor) -> torch.Tensor:
        """Computes saturation concentration cs(T) via Apelblat correlation (Eq. 11)."""
        T_clamped = torch.clamp(T_celsius, min=1.0)
        exponent = self.a1 + (self.a2 / T_clamped) + (self.a3 * torch.log(T_clamped))
        return 1000.0 * torch.exp(exponent)

    def calculate_supersaturation(self, c: torch.Tensor, T: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates supersaturation ratio S and relative supersaturation sigma."""
        cs = self.calculate_solubility(T)
        cs_safe = torch.clamp(cs, min=1e-5)
        S = c / cs_safe
        sigma = torch.clamp((c - cs_safe) / cs_safe, min=0.0)
        return S, sigma

    def calculate_growth_rates(
        self,
        sigma: torch.Tensor,
        mean_L1: torch.Tensor,
        mean_L2: torch.Tensor,
        lambda_k1: float = 1.0,
        lambda_k2: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes size-dependent face growth rates G1 and G2 (Eq. 2)."""
        sig_safe = torch.clamp(sigma, min=0.0)
        L1_um = torch.clamp(mean_L1 * 1.0e6, min=1.0)
        L2_um = torch.clamp(mean_L2 * 1.0e6, min=1.0)

        f1 = 1.0 + self.gamma1 * (L1_um ** self.alpha1)
        f2 = 1.0 + self.gamma2 * (L2_um ** self.alpha2)

        G1 = (self.k1 * lambda_k1) * (sig_safe ** self.g1) * f1
        G2 = (self.k2 * lambda_k2) * (sig_safe ** self.g2) * f2
        return G1, G2

    def calculate_nucleation_rate(
        self,
        sigma: torch.Tensor,
        epsilon: torch.Tensor,
        mu11: torch.Tensor,
        lambda_kS: float = 1.0,
    ) -> torch.Tensor:
        """Computes secondary contact nucleation rate B (Eq. 1)."""
        sig_safe = torch.clamp(sigma, min=0.0)
        mu11_safe = torch.clamp(mu11, min=0.0)
        eps_safe = torch.clamp(epsilon, min=0.0)
        return (self.kS * lambda_kS) * eps_safe * mu11_safe * (sig_safe ** self.b1)

    def compute_physics_loss(
        self,
        states_pred: torch.Tensor,       # [B, H, 7]: [T, c, mean_L1, mean_L2, aspect_ratio, log_mu00, log_mu11]
        inputs: torch.Tensor,            # [B, H, 2]: [cooling_rate, epsilon]
        curr_state: Optional[torch.Tensor] = None,  # [B, 7]: starting state at t_0
        dt_sample: float = 60.0,
        lambda_k1: float = 1.0,
        lambda_k2: float = 1.0,
        lambda_kS: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the exact QMOM continuous ODE residuals and conservation law losses:
        1. Temperature schedule: dT/dt = -cr
        2. Exact differential solute mass balance: dc/dt = - rho_c * kV * d(mu11)/dt
        3. Exact integral solute mass conservation: Delta c + rho_c * kV * Delta mu11 = 0
        4. Thermodynamic solubility limit: c >= cs(T) (no unphysical undersaturation)
        5. Aspect ratio definition: AR = L1 / L2
        6. Cooling growth monotonicity: dL1/dt >= 0, dL2/dt >= 0
        7. State non-negativity: T > 0, c > 0, L1 > 0, L2 > 0
        """
        B, H, _ = states_pred.shape

        if curr_state is not None:
            full_states = torch.cat([curr_state.unsqueeze(1), states_pred], dim=1)  # [B, H+1, 7]
            d_states_dt = (full_states[:, 1:, :] - full_states[:, :-1, :]) / dt_sample  # [B, H, 7]
            s_eval = states_pred  # [B, H, 7]
            u_eval = inputs  # [B, H, 2]
        else:
            if H < 2:
                zero_loss = torch.tensor(0.0, device=states_pred.device, dtype=states_pred.dtype)
                return {"total_physics_loss": zero_loss}
            d_states_dt = (states_pred[:, 1:, :] - states_pred[:, :-1, :]) / dt_sample
            s_eval = 0.5 * (states_pred[:, 1:, :] + states_pred[:, :-1, :])
            u_eval = 0.5 * (inputs[:, 1:, :] + inputs[:, :-1, :])
            full_states = states_pred

        T_eval = s_eval[:, :, 0]
        c_eval = s_eval[:, :, 1]
        L1_eval = torch.clamp(s_eval[:, :, 2], min=1e-6)
        L2_eval = torch.clamp(s_eval[:, :, 3], min=1e-6)
        ar_eval = s_eval[:, :, 4]
        cr_eval = u_eval[:, :, 0]

        # 1. Physics ODE 1: Temperature Schedule: dT/dt = -cr
        res_T = (d_states_dt[:, :, 0] - (-cr_eval)) / 0.003
        loss_T = torch.mean(res_T ** 2)

        # 2. Physics ODE 2: Exact Differential Solute Mass Balance: dc/dt = - rho_c * kV * d(mu11)/dt
        mu11_full = 10.0 ** full_states[:, :, 6]
        dmu11_dt = (mu11_full[:, 1:] - mu11_full[:, :-1]) / dt_sample
        dc_dt_expected = - self.rho_c * self.kV * dmu11_dt
        res_c_diff = (d_states_dt[:, :, 1] - dc_dt_expected) / 0.05
        loss_c_diff = torch.mean(res_c_diff ** 2)

        # 3. Physics Law 3: Exact Integral Solute Mass Conservation: Delta c + rho_c * kV * Delta mu11 = 0
        if curr_state is not None:
            c_0 = curr_state[:, 1].unsqueeze(1)
            mu11_0 = (10.0 ** curr_state[:, 6]).unsqueeze(1)
            delta_c = states_pred[:, :, 1] - c_0
            delta_mu11 = (10.0 ** states_pred[:, :, 6]) - mu11_0
            res_mass_int = (delta_c + self.rho_c * self.kV * delta_mu11) / 5.0
            loss_mass_int = torch.mean(res_mass_int ** 2)
        else:
            loss_mass_int = torch.tensor(0.0, device=states_pred.device, dtype=states_pred.dtype)

        # 4. Thermodynamic Law 4: Solubility Lower Bound c >= cs(T)
        cs_eval = self.calculate_solubility(T_eval)
        undersat_viol = torch.relu(cs_eval - c_eval)
        loss_thermo = torch.mean((undersat_viol / 10.0) ** 2)

        # 5. Physical Law 5: Aspect Ratio Definition AR = L1 / L2
        ar_expected = L1_eval / L2_eval
        res_ar = (ar_eval - ar_expected) / 2.0
        loss_ar = torch.mean(res_ar ** 2)

        # 6. Physical Law 6: Growth Monotonicity during Cooling (dL1/dt >= 0, dL2/dt >= 0)
        dL1 = full_states[:, 1:, 2] - full_states[:, :-1, 2]
        dL2 = full_states[:, 1:, 3] - full_states[:, :-1, 3]
        loss_mono = torch.mean((torch.relu(-dL1) / 1e-5) ** 2) + torch.mean((torch.relu(-dL2) / 1e-5) ** 2)

        # 7. Non-negativity constraint penalty: physical dimensions, T, and c cannot be negative
        loss_bounds = torch.mean((torch.relu(-c_eval) / 10.0) ** 2) + \
                      torch.mean((torch.relu(-T_eval) / 10.0) ** 2) + \
                      torch.mean((torch.relu(-s_eval[:, :, 2]) / 1e-5) ** 2) + \
                      torch.mean((torch.relu(-s_eval[:, :, 3]) / 1e-5) ** 2)

        total_physics_loss = loss_T + loss_c_diff + loss_mass_int + 2.0 * loss_thermo + loss_ar + 0.5 * loss_mono + 5.0 * loss_bounds

        return {
            "total_physics_loss": total_physics_loss,
            "loss_T": loss_T,
            "loss_c_diff": loss_c_diff,
            "loss_mass_int": loss_mass_int,
            "loss_thermo": loss_thermo,
            "loss_ar": loss_ar,
            "loss_mono": loss_mono,
            "loss_bounds": loss_bounds,
        }
