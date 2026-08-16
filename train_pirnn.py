import os
import time
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from config import Config
from pirnn_model import PIRNN, compute_pirnn_loss

def cstr_ode_torch(state, u_dev, F_val=Config.F_0, k_0_val=Config.k_0_t0):
    """
    Vectorized PyTorch implementation of CSTR ODE equations (eq 14a & 14b).
    state: Tensor of shape (N, 2) [C_A, T] (absolute physical values)
    u_dev: Tensor of shape (N, 2) [delta_C_A0, delta_Q] (deviation values)
    """
    C_A = state[:, 0]
    T = state[:, 1]
    
    C_A0 = Config.C_A0s + u_dev[:, 0]
    Q = Config.Q_s + u_dev[:, 1]
    
    T_safe = torch.clamp(T, min=100.0)
    k_T = k_0_val * torch.exp(-Config.E / (Config.R * T_safe))
    r_A = k_T * (C_A ** 2)
    
    dC_A_dt = (F_val / Config.V) * (C_A0 - C_A) - r_A
    
    heat_coef = -Config.DELTA_H / (Config.RHO_L * Config.C_P)
    dT_dt = (F_val / Config.V) * (Config.T_0 - T) + heat_coef * r_A + Q / (Config.RHO_L * Config.C_P * Config.V)
    
    return torch.stack([dC_A_dt, dT_dt], dim=1)

def rk4_step_torch(state, u_dev, dt, F_val=Config.F_0, k_0_val=Config.k_0_t0):
    """Vectorized 4th-order Runge-Kutta step."""
    k1 = cstr_ode_torch(state, u_dev, F_val, k_0_val)
    k2 = cstr_ode_torch(state + 0.5 * dt * k1, u_dev, F_val, k_0_val)
    k3 = cstr_ode_torch(state + 0.5 * dt * k2, u_dev, F_val, k_0_val)
    k4 = cstr_ode_torch(state + dt * k3, u_dev, F_val, k_0_val)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def generate_collocation_dataset_vectorized():
    """
    Vectorized generation of 240,000 CSTR Collocation Trajectories in <0.5 seconds!
    """
    print("Generating 240,000 CSTR Collocation Points (Vectorized PyTorch RK4)...")
    start_time = time.time()
    
    torch.manual_seed(Config.SEED)
    
    # 300 initial states x 800 control inputs = 240,000
    n_init = Config.NUM_COLLOCATION_INITIAL
    n_u = Config.NUM_COLLOCATION_INPUTS
    
    C_A_dev = (torch.rand(n_init) * 3.0) - 1.5   # [-1.5, 1.5]
    T_dev = (torch.rand(n_init) * 100.0) - 50.0  # [-50, 50]
    init_states = torch.stack([C_A_dev, T_dev], dim=1)
    
    C_A0_dev = (torch.rand(n_u) * 2 * Config.DELTA_C_A0_MAX) - Config.DELTA_C_A0_MAX
    Q_dev = (torch.rand(n_u) * 2 * Config.DELTA_Q_MAX) - Config.DELTA_Q_MAX
    inputs = torch.stack([C_A0_dev, Q_dev], dim=1)
    
    # Meshgrid repeat for all 240,000 combinations
    x_0_tensor = init_states.repeat_interleave(n_u, dim=0)  # (240000, 2)
    u_dev_tensor = inputs.repeat(n_init, 1)                  # (240000, 2)
    
    # Integrate forward for Config.SUB_STEPS (10 sub-steps)
    dt = Config.H_C
    curr_abs_state = torch.stack([
        x_0_tensor[:, 0] + Config.C_As,
        x_0_tensor[:, 1] + Config.T_s
    ], dim=1)
    
    trajectory = [x_0_tensor]
    
    for step in range(Config.SUB_STEPS):
        curr_abs_state = rk4_step_torch(curr_abs_state, u_dev_tensor, dt)
        curr_dev = torch.stack([
            curr_abs_state[:, 0] - Config.C_As,
            curr_abs_state[:, 1] - Config.T_s
        ], dim=1)
        trajectory.append(curr_dev)
        
    traj_tensor = torch.stack(trajectory, dim=1)  # (240000, 11, 2)
    
    elapsed = time.time() - start_time
    print(f" Dataset generated in {elapsed:.2f}s | Total samples: {len(x_0_tensor)}")
    return x_0_tensor, u_dev_tensor, traj_tensor

def train_nominal_pirnn():
    torch.manual_seed(Config.SEED)
    print("=" * 60)
    print("Training Nominal PIRNN Model (Offline Stage)")
    print(f"Device: {Config.DEVICE}")
    print("=" * 60)
    
    x_0, u_dev, traj = generate_collocation_dataset_vectorized()
    
    dataset = TensorDataset(x_0, u_dev, traj)
    
    # 60% Train, 20% Val, 20% Test split (Section 5 in Paper)
    total_size = len(dataset)
    train_size = int(0.6 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(Config.SEED)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    model = PIRNN(hidden_size=Config.HIDDEN_SIZE, sub_steps=Config.SUB_STEPS).to(Config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, Config.EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_data_loss = 0.0
        train_phys_loss = 0.0
        
        for batch_x0, batch_u, batch_traj in train_loader:
            batch_x0 = batch_x0.to(Config.DEVICE)
            batch_u = batch_u.to(Config.DEVICE)
            batch_traj = batch_traj.to(Config.DEVICE)
            
            optimizer.zero_grad()
            total_loss, data_loss, phys_loss = compute_pirnn_loss(
                model, batch_x0, batch_u, batch_traj, Config.F_0, Config.k_0_t0, Config.ETA_PHYSICS
            )
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item() * len(batch_x0)
            train_data_loss += data_loss.item() * len(batch_x0)
            train_phys_loss += phys_loss.item() * len(batch_x0)
            
        train_loss /= train_size
        train_data_loss /= train_size
        train_phys_loss /= train_size
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x0, batch_u, batch_traj in val_loader:
                batch_x0 = batch_x0.to(Config.DEVICE)
                batch_u = batch_u.to(Config.DEVICE)
                batch_traj = batch_traj.to(Config.DEVICE)
                
                tot_loss, _, _ = compute_pirnn_loss(
                    model, batch_x0, batch_u, batch_traj, Config.F_0, Config.k_0_t0, Config.ETA_PHYSICS
                )
                val_loss += tot_loss.item() * len(batch_x0)
        val_loss /= val_size
        
        print(f"Epoch [{epoch:02d}/{Config.EPOCHS}] | Train Loss: {train_loss:.6f} (Data: {train_data_loss:.6f}, Phys: {train_phys_loss:.6f}) | Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.NOMINAL_MODEL_PATH)
            print(f" Saved new best PIRNN model checkpoint to {Config.NOMINAL_MODEL_PATH}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}.")
                break
                
    print("Nominal PIRNN Offline Training Complete!")

if __name__ == "__main__":
    train_nominal_pirnn()
