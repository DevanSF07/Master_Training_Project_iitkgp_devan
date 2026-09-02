"""
Physics-Informed Recurrent Neural Network (PI-RNN) for Crystallization Modeling.

Implements the physics-informed neural network methodology developed in:
    Yingzhe Zheng and Zhe Wu (2023)
    "Physics-Informed Online Machine Learning and Predictive Control of
    Nonlinear Processes with Parameter Uncertainty"
    Industrial & Engineering Chemistry Research, 62(7), pp. 2804-2818.

Physical Laws Embedded:
1. Solute Mass Conservation:
   dc/dt + rho_c * RV = 0, where RV = kV * (G1 * mu01 + G2 * mu10)
2. 2D Population Balance Moment Growth Rates:
   d(mu00)/dt = B(sigma, eps)
   d(mu10)/dt = G1 * mu00
   d(mu01)/dt = G2 * mu00
   d(mu11)/dt = G1 * mu01 + G2 * mu10
3. Apelblat thermodynamic solubility equilibrium & supersaturation driving force.

Author: Devan Singh Faujdar
Master Training Project, IIT Kharagpur
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional

import config


class RecurrentPredictor(nn.Module):
    """Deep GRU-based recurrent predictor for dynamic crystallization states."""

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Multi-layer GRU backbone
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Dense projection head
        self.fc_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input sequence of shape (batch_size, seq_len, input_dim).
        h0 : torch.Tensor, optional
            Initial hidden state.
            
        Returns
        -------
        out : torch.Tensor
            Predicted states of shape (batch_size, seq_len, output_dim).
        hn : torch.Tensor
            Final hidden state.
        """
        gru_out, hn = self.gru(x, h0)
        out = self.fc_head(gru_out)
        return out, hn


class PhysicsLossModule(nn.Module):
    """
    Evaluates physical residuals from First-Principles Method of Moments equations.
    """

    def __init__(
        self,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        dt_nominal: float = 120.0,  # Nominal timestep between sequential points
    ):
        super().__init__()
        # Register normalization scalers as buffers (non-trainable)
        self.register_buffer("state_mean", torch.tensor(state_mean, dtype=torch.float32))
        self.register_buffer("state_std", torch.tensor(state_std, dtype=torch.float32))
        self.dt = dt_nominal

        # Physical constants
        self.rho_c = config.RHO_CRYSTAL
        self.kV = config.KV_SHAPE
        self.k1 = config.K1_GROWTH
        self.k2 = config.K2_GROWTH
        self.kS = config.KS_NUCLEATION

    def unnormalize(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Converts normalized predictions back to physical units."""
        return x_norm * self.state_std + self.state_mean

    def calculate_solubility_torch(self, T_celsius: torch.Tensor) -> torch.Tensor:
        """Differentiable Apelblat solubility in PyTorch."""
        T_clamped = torch.clamp(T_celsius, min=1.0)
        exponent = config.A1_APELBLAT + (config.A2_APELBLAT / T_clamped) + (config.A3_APELBLAT * torch.log(T_clamped))
        return 1000.0 * torch.exp(exponent)

    def forward(
        self,
        pred_norm: torch.Tensor,
        input_seq_norm: torch.Tensor,
        input_mean: np.ndarray,
        input_std: np.ndarray,
    ) -> torch.Tensor:
        """
        Computes the physics residual loss across the predicted trajectory.
        
        Parameters
        ----------
        pred_norm : torch.Tensor
            Predicted state sequence (batch_size, seq_len, 6).
        input_seq_norm : torch.Tensor
            Input sequence (batch_size, seq_len, 8).
        input_mean, input_std : np.ndarray
            Scalers for the input parameters [cr, eps].
        """
        # Unnormalize predicted states
        x = self.unnormalize(pred_norm)

        c = x[..., 0]      # Concentration [kg/m^3]
        T = x[..., 1]      # Temperature [°C]
        mu00 = torch.clamp(x[..., 2], min=1.0e8)
        mu10 = torch.clamp(x[..., 3], min=1.0)
        mu01 = torch.clamp(x[..., 4], min=1.0)
        mu11 = torch.clamp(x[..., 5], min=1.0)

        # Extract process inputs (unnormalized)
        in_std = torch.tensor(input_std, device=pred_norm.device, dtype=torch.float32)
        in_mean = torch.tensor(input_mean, device=pred_norm.device, dtype=torch.float32)
        u = input_seq_norm[..., 6:8] * in_std + in_mean
        eps = u[..., 1]    # Stirring energy [W/kg]

        # 1. Thermodynamics: Solubility & supersaturation
        cs = self.calculate_solubility_torch(T)
        sigma = F.relu((c - cs) / torch.clamp(cs, min=1.0))

        # 2. Crystal face sizes (in um) and growth rates
        L1_um = torch.clamp((mu10 / mu00) * 1.0e6, min=1.0)
        L2_um = torch.clamp((mu01 / mu00) * 1.0e6, min=1.0)

        f1 = 1.0 + config.GAMMA1_GROWTH * (L1_um ** config.ALPHA1_EXPONENT)
        f2 = 1.0 + config.GAMMA2_GROWTH * (L2_um ** config.ALPHA2_EXPONENT)

        G1 = self.k1 * (sigma ** config.G1_EXPONENT) * f1
        G2 = self.k2 * (sigma ** config.G2_EXPONENT) * f2

        # 3. Secondary nucleation rate
        B = self.kS * eps * mu11 * (sigma ** config.B1_NUCLEATION)

        # 4. Volumetric growth rate
        # RV = kV * [ G1 * mu01 + G2 * mu10 ]
        RV = self.kV * (G1 * mu01 + G2 * mu10)

        # 5. Discrete time finite-difference residuals along sequence
        # Numerical time derivatives: Delta x / Delta t
        delta_c = (c[:, 1:] - c[:, :-1]) / self.dt
        delta_mu00 = (mu00[:, 1:] - mu00[:, :-1]) / self.dt
        delta_mu10 = (mu10[:, 1:] - mu10[:, :-1]) / self.dt
        delta_mu01 = (mu01[:, 1:] - mu01[:, :-1]) / self.dt
        delta_mu11 = (mu11[:, 1:] - mu11[:, :-1]) / self.dt

        # Midpoint theoretical rates
        rate_c = - self.rho_c * RV[:, :-1]
        rate_mu00 = B[:, :-1]
        rate_mu10 = (G1[:, :-1] * mu00[:, :-1])
        rate_mu01 = (G2[:, :-1] * mu00[:, :-1])
        rate_mu11 = (G1[:, :-1] * mu01[:, :-1] + G2[:, :-1] * mu10[:, :-1])

        # Dimensionless normalized residuals
        res_mass = (delta_c - rate_c) / (self.state_std[0] / self.dt)
        res_mu00 = (delta_mu00 - rate_mu00) / (self.state_std[2] / self.dt)
        res_mu10 = (delta_mu10 - rate_mu10) / (self.state_std[3] / self.dt)
        res_mu01 = (delta_mu01 - rate_mu01) / (self.state_std[4] / self.dt)
        res_mu11 = (delta_mu11 - rate_mu11) / (self.state_std[5] / self.dt)

        total_physics_loss = (
            torch.mean(res_mass ** 2) +
            0.5 * torch.mean(res_mu00 ** 2) +
            torch.mean(res_mu10 ** 2) +
            torch.mean(res_mu01 ** 2) +
            torch.mean(res_mu11 ** 2)
        )
        return total_physics_loss


class PIRNNModel(nn.Module):
    """
    Composite Physics-Informed Recurrent Neural Network Model.
    """

    def __init__(
        self,
        scalers: Dict[str, np.ndarray],
        input_dim: int = 8,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 6,
        physics_weight: float = 0.35,
    ):
        super().__init__()
        self.predictor = RecurrentPredictor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
        )
        self.physics_module = PhysicsLossModule(
            state_mean=scalers["state_mean"],
            state_std=scalers["state_std"],
        )
        self.physics_weight = physics_weight
        self.input_mean = scalers["input_mean"]
        self.input_std = scalers["input_std"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred, _ = self.predictor(x)
        return pred

    def compute_loss(
        self,
        x_input: torch.Tensor,
        y_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes composite loss: L_data + lambda_phys * L_physics.
        """
        pred = self.forward(x_input)
        data_loss = F.mse_loss(pred, y_target)

        if self.physics_weight > 0.0:
            physics_loss = self.physics_module(pred, x_input, self.input_mean, self.input_std)
        else:
            physics_loss = torch.tensor(0.0, device=pred.device)

        total_loss = data_loss + self.physics_weight * physics_loss
        return total_loss, data_loss, physics_loss


if __name__ == "__main__":
    from dataset_generator import DataManager
    dm = DataManager()
    _, _, _, scalers = dm.prepare_data(num_batches=2)
    model = PIRNNModel(scalers=scalers)
    print("PIRNN Model initialized successfully!")
    dummy_x = torch.randn(4, 25, 8)
    dummy_y = torch.randn(4, 25, 6)
    tot, d_loss, p_loss = model.compute_loss(dummy_x, dummy_y)
    print(f"Total Loss: {tot.item():.4f}, Data Loss: {d_loss.item():.4f}, Physics Loss: {p_loss.item():.4f}")
