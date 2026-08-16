import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from config import Config
from cstr_env import CSTREnvironment
from pirnn_model import PIRNN, compute_pirnn_loss, cstr_physics_residual
from mpc_controller import PIRNN_MPC

def online_estimate_and_update(model, history_states, history_inputs, initial_F, initial_k0, use_physics=True):
    """
    Estimates uncertain process parameters (F, k_0) and updates PIRNN model weights in real-time (<30s).
    """
    print(f"   Triggering Online Model Update (Physics-Informed={use_physics})...")
    start_t = time.time()
    
    # Trainable parameter parameters
    F_param = torch.tensor(initial_F, dtype=torch.float32, requires_grad=True)
    k0_param = torch.tensor(initial_k0, dtype=torch.float32, requires_grad=True)
    
    optimizer = torch.optim.Adam(list(model.parameters()) + [F_param, k0_param], lr=1e-3)
    
    # Convert recent history to PyTorch tensors
    x0_t = torch.tensor(np.array(history_states[:-1]), dtype=torch.float32).to(Config.DEVICE)
    u_t = torch.tensor(np.array(history_inputs), dtype=torch.float32).to(Config.DEVICE)
    next_x_t = torch.tensor(np.array(history_states[1:]), dtype=torch.float32).to(Config.DEVICE)
    
    # Online adaptation loop (50 fast steps)
    for step in range(50):
        optimizer.zero_grad()
        pred_traj = model(x0_t, u_t)
        pred_next = pred_traj[:, -1, :]
        
        mse_data = torch.mean((pred_next - next_x_t) ** 2)
        
        if use_physics:
            phys_loss = cstr_physics_residual(pred_traj, u_t, F_param, k0_param)
            loss = mse_data + Config.ETA_PHYSICS * phys_loss
        else:
            loss = mse_data
            
        loss.backward()
        optimizer.step()
        
    est_F = F_param.item()
    est_k0 = k0_param.item()
    elapsed = time.time() - start_t
    print(f"   Update Complete in {elapsed:.2f}s | Est F: {est_F:.2f} m^3/h (True: {initial_F:.2f}), Est k_0: {est_k0:.2e}")
    return est_F, est_k0

def simulate_scheme(scheme_name, nominal_model_path):
    """
    Simulates a closed-loop CSTR control scheme for t = 0 to 0.3 h (30 sampling periods).
    """
    print(f"\nRunning Simulation Scheme: {scheme_name}...")
    
    env = CSTREnvironment(F=Config.F_0, k_0=Config.k_0_t0)
    
    # Load model instance
    model = PIRNN(hidden_size=Config.HIDDEN_SIZE, sub_steps=Config.SUB_STEPS).to(Config.DEVICE)
    model.load_state_dict(torch.load(nominal_model_path, map_location=Config.DEVICE))
    
    mpc = PIRNN_MPC(model)
    
    num_steps = 30
    dt = Config.DELTA  # 0.01 h
    
    time_pts = [0.0]
    states_hist = [[0.0, 0.0]]
    inputs_hist = []
    e_rnn_hist = [0.0]
    
    current_state = np.array([0.0, 0.0])
    prev_u = np.array([0.0, 0.0])
    
    current_F = Config.F_0
    current_k0 = Config.k_0_t0
    
    consecutive_trigger = 0
    
    for step_idx in range(1, num_steps + 1):
        t_curr = step_idx * dt
        
        # Apply paper disturbances at designated time steps
        if step_idx == 10:  # t = 0.09 h
            env.set_disturbance(F_ratio=1.60, k_0_ratio=0.80)
            current_F = Config.F_0 * 1.60
            current_k0 = Config.k_0_t0 * 0.80
        elif step_idx == 20:  # t = 0.19 h
            env.set_disturbance(F_ratio=2.30, k_0_ratio=0.30)
            current_F = Config.F_0 * 2.30
            current_k0 = Config.k_0_t0 * 0.30
            
        # 1. Compute control action via MPC
        u_opt = mpc.compute_control(current_state, prev_u)
        prev_u = u_opt
        inputs_hist.append(u_opt)
        
        # 2. Evaluate model prediction error E_RNN(t)
        x0_t = torch.tensor([current_state], dtype=torch.float32).to(Config.DEVICE)
        u_t = torch.tensor([u_opt], dtype=torch.float32).to(Config.DEVICE)
        with torch.no_grad():
            pred_traj = model(x0_t, u_t)
            pred_next = pred_traj[0, -1, :].cpu().numpy()
            
        # 3. Apply control to real plant
        next_state = env.step(u_opt, dt=dt)
        states_hist.append(next_state.tolist())
        time_pts.append(t_curr)
        
        # Calculate prediction MSE error E_RNN
        e_rnn = np.mean((pred_next - next_state) ** 2)
        e_rnn_hist.append(e_rnn)
        
        # 4. Error-triggered mechanism
        if scheme_name != "PIRNN_no_update":
            if e_rnn > Config.ERROR_THRESHOLD_E_T:
                consecutive_trigger += 1
            else:
                consecutive_trigger = 0
                
            if consecutive_trigger >= 3:
                use_phys = (scheme_name == "PIRNN_physics_enhanced")
                recent_states = states_hist[-5:]
                recent_inputs = inputs_hist[-4:]
                online_estimate_and_update(model, recent_states, recent_inputs, current_F, current_k0, use_physics=use_phys)
                consecutive_trigger = 0
                
        current_state = next_state
        
    return {
        'time': time_pts,
        'states': np.array(states_hist),
        'inputs': np.array(inputs_hist),
        'e_rnn': np.array(e_rnn_hist)
    }

def main():
    if not os.path.exists(Config.NOMINAL_MODEL_PATH):
        print("Nominal PIRNN checkpoint not found. Running train_pirnn.py first...")
        from train_pirnn import train_nominal_pirnn
        train_nominal_pirnn()
        
    # Run 3 paper comparison schemes
    res_no_update = simulate_scheme("PIRNN_no_update", Config.NOMINAL_MODEL_PATH)
    res_data_online = simulate_scheme("PIRNN_data_online", Config.NOMINAL_MODEL_PATH)
    res_physics_enhanced = simulate_scheme("PIRNN_physics_enhanced", Config.NOMINAL_MODEL_PATH)
    
    # -------------------------------------------------------------
    # Plot Figure 4: Closed-Loop System State Trajectories (ΔC_A, ΔT)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    t_axis = res_physics_enhanced['time']
    
    # ΔC_A
    ax1.plot(t_axis, res_no_update['states'][:, 0], 'r--', linewidth=2, label='PIRNN_no_update')
    ax1.plot(t_axis, res_data_online['states'][:, 0], 'g-.', linewidth=2, label='PIRNN_data_online')
    ax1.plot(t_axis, res_physics_enhanced['states'][:, 0], 'b-', linewidth=2.5, label='PIRNN_physics_enhanced (Proposed)')
    ax1.axhline(0, color='black', linestyle=':', alpha=0.7)
    ax1.set_ylabel('ΔC_A (kmol/m³)', fontsize=12)
    ax1.set_title('Figure 4: Closed-Loop CSTR State Trajectories under Parameter Uncertainty', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=10)
    
    # ΔT
    ax2.plot(t_axis, res_no_update['states'][:, 1], 'r--', linewidth=2, label='PIRNN_no_update')
    ax2.plot(t_axis, res_data_online['states'][:, 1], 'g-.', linewidth=2, label='PIRNN_data_online')
    ax2.plot(t_axis, res_physics_enhanced['states'][:, 1], 'b-', linewidth=2.5, label='PIRNN_physics_enhanced (Proposed)')
    ax2.axhline(0, color='black', linestyle=':', alpha=0.7)
    ax2.set_xlabel('Time t (hr)', fontsize=12)
    ax2.set_ylabel('ΔT (K)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(fontsize=10)
    
    fig.tight_layout()
    fig_4_path = os.path.join(Config.PLOT_DIR, "closed_loop_states.png")
    plt.savefig(fig_4_path, dpi=300)
    plt.close()
    print(f" Figure 4 plot saved to {fig_4_path}")
    
    # -------------------------------------------------------------
    # Plot Figure 5: Control Actions (ΔC_A0, ΔQ)
    # -------------------------------------------------------------
    fig5, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    t_inputs = t_axis[1:]
    
    # ΔC_A0
    ax3.step(t_inputs, res_no_update['inputs'][:, 0], 'r--', linewidth=2, where='post', label='PIRNN_no_update')
    ax3.step(t_inputs, res_data_online['inputs'][:, 0], 'g-.', linewidth=2, where='post', label='PIRNN_data_online')
    ax3.step(t_inputs, res_physics_enhanced['inputs'][:, 0], 'b-', linewidth=2.5, where='post', label='PIRNN_physics_enhanced')
    ax3.set_ylabel('ΔC_A0 (kmol/m³)', fontsize=12)
    ax3.set_title('Figure 5: Control Actions Adopted by the PIRNN-LMPC Schemes', fontsize=14)
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.legend(fontsize=10)
    
    # ΔQ
    ax4.step(t_inputs, res_no_update['inputs'][:, 1] / 1e5, 'r--', linewidth=2, where='post', label='PIRNN_no_update')
    ax4.step(t_inputs, res_data_online['inputs'][:, 1] / 1e5, 'g-.', linewidth=2, where='post', label='PIRNN_data_online')
    ax4.step(t_inputs, res_physics_enhanced['inputs'][:, 1] / 1e5, 'b-', linewidth=2.5, where='post', label='PIRNN_physics_enhanced')
    ax4.set_xlabel('Time t (hr)', fontsize=12)
    ax4.set_ylabel('ΔQ (x10⁵ kJ/h)', fontsize=12)
    ax4.grid(True, linestyle='--', alpha=0.6)
    ax4.legend(fontsize=10)
    
    fig5.tight_layout()
    fig_5_path = os.path.join(Config.PLOT_DIR, "control_actions.png")
    plt.savefig(fig_5_path, dpi=300)
    plt.close()
    print(f" Figure 5 plot saved to {fig_5_path}")

    # -------------------------------------------------------------
    # Plot Figure 6: Moving-Horizon Error E_RNN(t)
    # -------------------------------------------------------------
    fig6, ax5 = plt.subplots(figsize=(10, 5))
    ax5.plot(t_axis, res_no_update['e_rnn'], 'r--o', linewidth=2, label='PIRNN_no_update')
    ax5.plot(t_axis, res_data_online['e_rnn'], 'g-.s', linewidth=2, label='PIRNN_data_online')
    ax5.plot(t_axis, res_physics_enhanced['e_rnn'], 'b-d', linewidth=2.5, label='PIRNN_physics_enhanced')
    ax5.axhline(Config.ERROR_THRESHOLD_E_T, color='black', linestyle=':', linewidth=2, label='Error Threshold E_T = 1e-4')
    ax5.set_yscale('log')
    ax5.set_xlabel('Time t (hr)', fontsize=12)
    ax5.set_ylabel('MSE Error E_RNN(t)', fontsize=12)
    ax5.set_title('Figure 6: Evolution of Moving-Horizon Error E_RNN(t) with Error-Triggered Mechanism', fontsize=14)
    ax5.grid(True, which='both', linestyle='--', alpha=0.6)
    ax5.legend(fontsize=10)
    
    fig6.tight_layout()
    fig_6_path = os.path.join(Config.PLOT_DIR, "moving_horizon_error.png")
    plt.savefig(fig_6_path, dpi=300)
    plt.close()
    print(f" Figure 6 plot saved to {fig_6_path}")

if __name__ == "__main__":
    main()
