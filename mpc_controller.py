import torch
import numpy as np
from scipy.optimize import minimize
from config import Config

class PIRNN_MPC:
    """
    Model Predictive Controller using PIRNN surrogate model for trajectory prediction.
    """
    def __init__(self, model, device=Config.DEVICE):
        self.model = model
        self.device = device
        
        # Physical bounds on input deviations
        self.bounds = [
            (-Config.DELTA_C_A0_MAX, Config.DELTA_C_A0_MAX),
            (-Config.DELTA_Q_MAX, Config.DELTA_Q_MAX)
        ]
        
    def objective_function(self, u_vec, current_state_dev):
        """
        Calculates control objective cost over prediction horizon:
        Cost = w_C * ΔC_A^2 + w_T * (ΔT / 100)^2 + w_u1 * ΔC_A0^2 + w_u2 * (ΔQ / 1e5)^2
        """
        x_0_t = torch.tensor([current_state_dev], dtype=torch.float32).to(self.device)
        u_t = torch.tensor([u_vec], dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            pred_traj = self.model(x_0_t, u_t)  # Shape (1, SUB_STEPS+1, 2)
            
        final_state = pred_traj[0, -1, :]  # State at t + Δ
        
        c_a_dev = final_state[0].item()
        t_dev = final_state[1].item()
        
        # State penalization weights
        w_ca = 10.0
        w_t = 1.0
        w_u1 = 0.1
        w_u2 = 1e-10
        
        cost = w_ca * (c_a_dev ** 2) + w_t * ((t_dev / 100.0) ** 2) + w_u1 * (u_vec[0] ** 2) + w_u2 * (u_vec[1] ** 2)
        return cost

    def compute_control(self, current_state_dev, prev_u=None):
        """
        Solves online optimization problem to return optimal control input u.
        """
        if prev_u is None:
            x0_guess = [0.0, 0.0]
        else:
            x0_guess = prev_u
            
        res = minimize(
            fun=self.objective_function,
            x0=x0_guess,
            args=(current_state_dev,),
            method='SLSQP',
            bounds=self.bounds,
            options={'maxiter': 50, 'ftol': 1e-4}
        )
        
        optimal_u = res.x
        return optimal_u
