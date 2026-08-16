import torch
import torch.nn as nn
import torch.optim as optim
from config import Config

def cstr_physics_residual(pred_trajectory, u_dev, F_param, k_0_param):
    """
    Computes residual of CSTR ODE equations across predicted trajectory sub-steps.
    pred_trajectory: Tensor of shape (batch_size, sub_steps + 1, 2) in deviation variables [ΔC_A, ΔT]
    u_dev: Tensor of shape (batch_size, 2) in deviation variables [ΔC_A0, ΔQ]
    F_param: scalar or Tensor (volumetric flow rate)
    k_0_param: scalar or Tensor (pre-exponential rate constant)
    """
    batch_size, num_steps, _ = pred_trajectory.shape
    dt = Config.H_C  # sub-step dt = 0.001 h
    
    # Absolute physical values
    C_A_abs = pred_trajectory[:, :, 0] + Config.C_As
    T_abs = pred_trajectory[:, :, 1] + Config.T_s
    
    C_A0_abs = u_dev[:, 0].unsqueeze(1) + Config.C_A0s
    Q_abs = u_dev[:, 1].unsqueeze(1) + Config.Q_s
    
    # Finite difference state derivatives: dX/dt ≈ (X_{t+1} - X_t) / dt
    dC_A_dt_num = (C_A_abs[:, 1:] - C_A_abs[:, :-1]) / dt
    dT_dt_num = (T_abs[:, 1:] - T_abs[:, :-1]) / dt
    
    # Evaluate ODE right-hand sides at intermediate points
    C_A_mid = C_A_abs[:, :-1]
    T_mid = torch.clamp(T_abs[:, :-1], min=100.0)
    
    k_T = k_0_param * torch.exp(-Config.E / (Config.R * T_mid))
    r_A = k_T * (C_A_mid ** 2)
    
    dC_A_dt_physics = (F_param / Config.V) * (C_A0_abs - C_A_mid) - r_A
    
    heat_coef = -Config.DELTA_H / (Config.RHO_L * Config.C_P)
    dT_dt_physics = (F_param / Config.V) * (Config.T_0 - T_mid) + heat_coef * r_A + Q_abs / (Config.RHO_L * Config.C_P * Config.V)
    
    res_C_A = dC_A_dt_num - dC_A_dt_physics
    res_T = dT_dt_num - dT_dt_physics
    
    physics_mse = torch.mean(res_C_A ** 2) + torch.mean((res_T / 100.0) ** 2)
    return physics_mse

class PIRNN(nn.Module):
    """
    Physics-Informed Recurrent Neural Network for predicting CSTR dynamic trajectories.
    Inputs: [initial state x_0 (2), control u (2)] -> total 4 input features.
    Outputs: State trajectory across time steps (batch_size, SUB_STEPS+1, 2).
    """
    def __init__(self, hidden_size=Config.HIDDEN_SIZE, sub_steps=Config.SUB_STEPS):
        super(PIRNN, self).__init__()
        self.sub_steps = sub_steps
        
        self.fc_in = nn.Sequential(
            nn.Linear(4, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        
        self.rnn_cell = nn.GRUCell(hidden_size, hidden_size)
        
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 2)
        )

    def forward(self, x_0, u_dev):
        """
        x_0: (batch_size, 2)
        u_dev: (batch_size, 2)
        Returns: trajectory (batch_size, sub_steps + 1, 2)
        """
        batch_size = x_0.size(0)
        inputs = torch.cat([x_0, u_dev], dim=1)
        h = self.fc_in(inputs)
        
        trajectory = [x_0]
        
        for step in range(self.sub_steps):
            h = self.rnn_cell(h, h)
            delta_state = self.fc_out(h)
            next_state = trajectory[-1] + delta_state
            trajectory.append(next_state)
            
        # Stack along sequence dimension
        out_trajectory = torch.stack(trajectory, dim=1)
        return out_trajectory

def compute_pirnn_loss(model, x_0, u_dev, true_trajectory, F_param, k_0_param, eta=Config.ETA_PHYSICS):
    """
    Hybrid loss function: Loss = MSE_data + eta * MSE_physics (eq 5 & eq 7).
    """
    pred_trajectory = model(x_0, u_dev)
    
    mse_data = nn.MSELoss()(pred_trajectory, true_trajectory)
    mse_physics = cstr_physics_residual(pred_trajectory, u_dev, F_param, k_0_param)
    
    total_loss = mse_data + eta * mse_physics
    return total_loss, mse_data, mse_physics
