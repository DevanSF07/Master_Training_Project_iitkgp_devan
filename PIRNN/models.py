"""
Recurrent Neural Network Architectures for Crystallization Trajectory Forecasting.

Implements:
- CrystallizerRecurrentModel: Encoder-Decoder architecture supporting GRU, LSTM, and standard tanh-RNN.
- Autoregressive full-batch rollout methods for open-loop validation and MPC simulation.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional


class CrystallizerRecurrentModel(nn.Module):
    """
    Recurrent Encoder-Decoder Network for multi-step ahead crystallization forecasting.

    Encoder:
        Processes past history window [x_{t-L+1}, ..., x_t, u_{t-L+1}, ..., u_t]
        producing a compact latent context representation h_t.
    Decoder:
        Unrolls across the future horizon H, taking future cooling inputs u_{t+1:t+H}
        and forecasting state increments Delta x_{t+1:t+H}.
    """

    def __init__(
        self,
        input_dim: int = 9,          # 7 states + 2 inputs in history
        future_input_dim: int = 2,   # 2 manipulated inputs [cooling_rate, epsilon]
        target_dim: int = 7,         # 7 predicted state increments
        hidden_dim: int = 64,
        num_layers: int = 2,
        rnn_type: str = "GRU",       # 'GRU', 'LSTM', or 'RNN'
        dropout: float = 0.05,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.future_input_dim = future_input_dim
        self.target_dim = target_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type.upper()

        # Encoder
        if self.rnn_type == "GRU":
            self.encoder = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            # Decoder step takes [future_input, previous_increment_prediction]
            self.decoder_cell = nn.GRUCell(
                input_size=future_input_dim + target_dim,
                hidden_size=hidden_dim,
            )
        elif self.rnn_type == "LSTM":
            self.encoder = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.decoder_cell = nn.LSTMCell(
                input_size=future_input_dim + target_dim,
                hidden_size=hidden_dim,
            )
        else:  # Standard tanh-RNN (matches paper's default architecture)
            self.encoder = nn.RNN(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                nonlinearity="tanh",
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.decoder_cell = nn.RNNCell(
                input_size=future_input_dim + target_dim,
                hidden_size=hidden_dim,
                nonlinearity="tanh",
            )

        # Output projection head: latent hidden state -> state increment Delta x
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, target_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Initializes weights using uniform distribution as described in PIRNN literature."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        history: torch.Tensor,
        future_inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for rolling horizon forecasting.

        Parameters:
        - history: [B, L, input_dim]
        - future_inputs: [B, H, future_input_dim]

        Returns:
        - pred_increments: [B, H, target_dim] (Delta y = y(t+k) - y(t))
        """
        B, H, _ = future_inputs.shape

        # 1. Encode past history
        if self.rnn_type == "LSTM":
            _, (h_enc, c_enc) = self.encoder(history)
            h_dec = h_enc[-1]  # top layer hidden state
            c_dec = c_enc[-1]
        else:
            _, h_enc = self.encoder(history)
            h_dec = h_enc[-1]  # [B, hidden_dim]

        # 2. Decode across future horizon H
        pred_increments = []
        # Initial previous increment is zero at t = t_0
        prev_pred = torch.zeros(B, self.target_dim, device=history.device, dtype=history.dtype)

        for k in range(H):
            u_k = future_inputs[:, k, :]  # [B, future_input_dim]
            dec_in = torch.cat([u_k, prev_pred], dim=-1)  # [B, future_input_dim + target_dim]

            if self.rnn_type == "LSTM":
                h_dec, c_dec = self.decoder_cell(dec_in, (h_dec, c_dec))
            else:
                h_dec = self.decoder_cell(dec_in, h_dec)

            delta_k = self.fc_out(h_dec)  # [B, target_dim]
            pred_increments.append(delta_k)
            prev_pred = delta_k

        return torch.stack(pred_increments, dim=1)  # [B, H, target_dim]

    @torch.no_grad()
    def autoregressive_rollout(
        self,
        initial_history: torch.Tensor,
        future_inputs_full: torch.Tensor,
        norm_stats: Dict[str, Any],
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Executes a complete full-batch open-loop autoregressive rollout:
        Starting with only initial history at t=0, forecasts step-by-step
        across the entire batch duration using its own past predictions!

        Parameters:
        - initial_history: [L, input_dim] (normalized)
        - future_inputs_full: [N_total, future_input_dim] (normalized)
        - norm_stats: Normalization mean and std tensors

        Returns:
        - unnormalized_predictions: [N_total, target_dim]
        """
        self.eval()
        L = initial_history.shape[0]
        N_total = future_inputs_full.shape[0]

        curr_hist = initial_history.clone().unsqueeze(0).to(device)  # [1, L, input_dim]
        all_pred_states = []

        s_mean = norm_stats["state_mean"].to(device)
        s_std = norm_stats["state_std"].to(device)

        # Current normalized state at t = 0
        curr_state_norm = curr_hist[0, -1, : self.target_dim]

        for step in range(N_total):
            u_step = future_inputs_full[step : step + 1].unsqueeze(0).to(device)  # [1, 1, 2]
            # Predict 1 step ahead increment
            pred_inc = self.forward(curr_hist, u_step)  # [1, 1, target_dim]
            delta_norm = pred_inc[0, 0]  # [target_dim]

            # Update state: y_{next} = y_{curr} + Delta y
            next_state_norm = curr_state_norm + delta_norm
            curr_state_norm = next_state_norm

            # Denormalize for physical state storage
            next_state_phys = (next_state_norm * s_std) + s_mean
            all_pred_states.append(next_state_phys.cpu())

            # Update rolling history window: shift left and append [next_state_norm, u_step]
            new_entry = torch.cat([next_state_norm, u_step[0, 0]], dim=-1).unsqueeze(0).unsqueeze(0)
            curr_hist = torch.cat([curr_hist[:, 1:, :], new_entry], dim=1)

        return torch.stack(all_pred_states, dim=0)  # [N_total, target_dim]
