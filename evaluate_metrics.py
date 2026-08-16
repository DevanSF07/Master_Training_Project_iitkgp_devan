import os
import numpy as np
import torch
from config import Config
from cstr_env import CSTREnvironment
from pirnn_model import PIRNN
from mpc_controller import PIRNN_MPC
from run_closed_loop_simulation import simulate_scheme

def compute_performance_metrics():
    print("=" * 60)
    print("Computing Quantitative Performance Metrics for CSTR MPC Schemes")
    print("=" * 60)
    
    schemes = ["PIRNN_no_update", "PIRNN_data_online", "PIRNN_physics_enhanced"]
    results = {}
    
    for s in schemes:
        res = simulate_scheme(s, Config.NOMINAL_MODEL_PATH)
        results[s] = res
        
    dt = Config.DELTA  # 0.01 h
    
    print("\n" + "=" * 80)
    print(f"{'Control Scheme':<25} | {'IAE (C_A)':<12} | {'IAE (T)':<12} | {'ISE (C_A)':<12} | {'ISE (T)':<12} | {'Mean MSE E_RNN':<15}")
    print("=" * 80)
    
    metrics_summary = {}
    
    for s in schemes:
        c_a_dev = results[s]['states'][:, 0]
        t_dev = results[s]['states'][:, 1]
        e_rnn = results[s]['e_rnn']
        
        iae_ca = np.sum(np.abs(c_a_dev)) * dt
        iae_t = np.sum(np.abs(t_dev)) * dt
        
        ise_ca = np.sum(c_a_dev ** 2) * dt
        ise_t = np.sum(t_dev ** 2) * dt
        
        mean_e_rnn = np.mean(e_rnn)
        
        metrics_summary[s] = {
            'iae_ca': iae_ca,
            'iae_t': iae_t,
            'ise_ca': ise_ca,
            'ise_t': ise_t,
            'mean_e_rnn': mean_e_rnn
        }
        
        print(f"{s:<25} | {iae_ca:<12.4f} | {iae_t:<12.4f} | {ise_ca:<12.4f} | {ise_t:<12.4f} | {mean_e_rnn:<15.6e}")
        
    print("=" * 80)
    
    # Compute percentage improvements of Proposed vs Static
    static_iae_ca = metrics_summary['PIRNN_no_update']['iae_ca']
    proposed_iae_ca = metrics_summary['PIRNN_physics_enhanced']['iae_ca']
    impr_ca = ((static_iae_ca - proposed_iae_ca) / static_iae_ca) * 100
    
    static_iae_t = metrics_summary['PIRNN_no_update']['iae_t']
    proposed_iae_t = metrics_summary['PIRNN_physics_enhanced']['iae_t']
    impr_t = ((static_iae_t - proposed_iae_t) / static_iae_t) * 100

    print(f"\n Proposed Physics-Enhanced Scheme Improvements over Static Baseline:")
    print(f"   • Concentration Control Error Reduction (IAE C_A): {impr_ca:.2f}%")
    print(f"   • Temperature Control Error Reduction (IAE T):    {impr_t:.2f}%")

if __name__ == "__main__":
    compute_performance_metrics()
