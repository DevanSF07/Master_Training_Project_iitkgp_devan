"""
Direct Comparison Benchmark: Baseline Pure RNN vs Forward PIRNN.

Compares:
1. Multi-step ahead forecasting accuracy (NRMSE, MAE) on unseen test windows.
2. Full 2.8-hour open-loop autoregressive rollouts without ground-truth feedback.
3. Physical conservation law satisfaction:
   - Mass balance residual: |Delta c + rho_c * kV * Delta mu11|
   - Aspect ratio consistency: |AR - L1 / L2|
   - Dynamic crystal face growth rates
4. Generates publication-ready comparative multi-panel figure.

Usage:
    PYTHONPATH=. .venv/bin/python3 PIRNN/compare_models.py
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import config
from PIRNN.dataset import CrystallizerWindowDataset
from PIRNN.models import CrystallizerRecurrentModel


def calculate_nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Normalized Root Mean Square Error (NRMSE = RMSE / range)."""
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    val_range = np.ptp(y_true)
    if val_range < 1e-8:
        val_range = np.std(y_true) + 1e-8
    return float(rmse / val_range)


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def compare_models(
    baseline_ckpt_path: str = "PIRNN/checkpoints/baseline_gru_best.pt",
    pirnn_ckpt_path: str = "PIRNN/checkpoints/forward_pirnn_best.pt",
    test_data_path: str = "PIRNN/data/crystallizer_test.pt",
    output_dir: str = "PIRNN/plots",
):
    os.makedirs(output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"--> [Comparison Engine] Running on device: {device}")

    # 1. Load Checkpoints
    b_ckpt = torch.load(baseline_ckpt_path, map_location=device, weights_only=False)
    p_ckpt = torch.load(pirnn_ckpt_path, map_location=device, weights_only=False)

    norm_stats = b_ckpt["norm_stats"]
    cfg_b = b_ckpt["config"]
    cfg_p = p_ckpt["config"]

    model_baseline = CrystallizerRecurrentModel(
        input_dim=cfg_b["input_dim"],
        future_input_dim=2,
        target_dim=cfg_b["target_dim"],
        hidden_dim=cfg_b["hidden_dim"],
        num_layers=cfg_b["num_layers"],
        rnn_type=cfg_b["rnn_type"],
    ).to(device)
    model_baseline.load_state_dict(b_ckpt["model_state_dict"])
    model_baseline.eval()

    model_pirnn = CrystallizerRecurrentModel(
        input_dim=cfg_p["input_dim"],
        future_input_dim=2,
        target_dim=cfg_p["target_dim"],
        hidden_dim=cfg_p["hidden_dim"],
        num_layers=cfg_p["num_layers"],
        rnn_type=cfg_p["rnn_type"],
    ).to(device)
    model_pirnn.load_state_dict(p_ckpt["model_state_dict"])
    model_pirnn.eval()

    print(f"--> Loaded Pure Baseline {cfg_b['rnn_type']} (Best Epoch {b_ckpt['epoch']}, Val MSE: {b_ckpt['best_val_loss']:.6f})")
    print(f"--> Loaded Forward PIRNN {cfg_p['rnn_type']} (Best Epoch {p_ckpt['epoch']}, Val MSE: {p_ckpt['best_val_loss']:.6f})")

    # 2. Test Set Multi-Step Evaluation
    test_data = torch.load(test_data_path, weights_only=False)
    test_batches = test_data["batches"]
    test_ds = CrystallizerWindowDataset(test_data["windows"])
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False)

    s_mean = norm_stats["state_mean"].to(device)
    s_std = norm_stats["state_std"].to(device)
    state_names = norm_stats["state_names"]

    b_preds_all, p_preds_all, targets_all = [], [], []

    with torch.no_grad():
        for hist, fut_in, targ_inc, targ_abs in test_loader:
            hist = hist.to(device)
            fut_in = fut_in.to(device)

            curr_state_norm = hist[:, -1, :7]
            curr_state = (curr_state_norm * s_std) + s_mean

            # Baseline inference
            pred_inc_b = model_baseline(hist, fut_in)
            pred_abs_b = curr_state.unsqueeze(1) + (pred_inc_b * s_std)
            b_preds_all.append(pred_abs_b.cpu())

            # PIRNN inference
            pred_inc_p = model_pirnn(hist, fut_in)
            pred_abs_p = curr_state.unsqueeze(1) + (pred_inc_p * s_std)
            p_preds_all.append(pred_abs_p.cpu())

            targets_all.append(targ_abs)

    b_preds_all = torch.cat(b_preds_all, dim=0).numpy()
    p_preds_all = torch.cat(p_preds_all, dim=0).numpy()
    targets_all = torch.cat(targets_all, dim=0).numpy()

    # Compare 10-step ahead test metrics
    metrics_summary = []
    for idx, name in enumerate(state_names):
        true_col = targets_all[:, :, idx]
        b_col = b_preds_all[:, :, idx]
        p_col = p_preds_all[:, :, idx]

        mae_b = calculate_mae(true_col, b_col)
        mae_p = calculate_mae(true_col, p_col)
        nrmse_b = calculate_nrmse(true_col, b_col) * 100.0
        nrmse_p = calculate_nrmse(true_col, p_col) * 100.0

        improvement_mae = ((mae_b - mae_p) / max(mae_b, 1e-8)) * 100.0

        metrics_summary.append({
            "Variable": name,
            "Baseline MAE": mae_b,
            "PIRNN MAE": mae_p,
            "PIRNN MAE Gain (%)": improvement_mae,
            "Baseline NRMSE (%)": nrmse_b,
            "PIRNN NRMSE (%)": nrmse_p,
        })

    df_test_metrics = pd.DataFrame(metrics_summary)
    print("\n=================== 1. MULTI-STEP TEST SET BENCHMARK ===================")
    print(df_test_metrics.to_string(index=False))

    # 3. Full 2.8-Hour Open-Loop Autoregressive Rollout
    print("\n============= 2. UNCONSTRAINED OPEN-LOOP ROLLOUT BENCHMARK =============")
    # Use test batch 0 (unseen batch)
    b_test = test_batches[0]
    p_type = b_test["profile_type"]
    N_steps = len(b_test["time"])
    L = 5

    print(f"--> Evaluating on Test Batch #{b_test['batch_id']} ({p_type} cooling, {N_steps} timestamps = {b_test['time'][-1]/3600:.2f} h)")

    # Prepare initial history
    hist_raw_states = np.column_stack([
        b_test["T_meas"][:L],
        b_test["c_meas"][:L],
        b_test["mean_L1_meas"][:L],
        b_test["mean_L2_meas"][:L],
        b_test["aspect_ratio_meas"][:L],
        np.log10(np.maximum(b_test["mu00_meas"][:L], 1e-5)),
        np.log10(np.maximum(b_test["mu11_meas"][:L], 1e-5)),
    ])
    hist_raw_inputs = np.column_stack([
        b_test["cooling_rate"][:L],
        np.full(L, b_test["epsilon"]),
    ])

    u_mean = norm_stats["input_mean"].cpu().numpy()
    u_std = norm_stats["input_std"].cpu().numpy()
    s_mean_np = norm_stats["state_mean"].cpu().numpy()
    s_std_np = norm_stats["state_std"].cpu().numpy()

    hist_norm_states = (hist_raw_states - s_mean_np) / s_std_np
    hist_norm_inputs = (hist_raw_inputs - u_mean) / u_std
    init_hist_tensor = torch.tensor(np.hstack([hist_norm_states, hist_norm_inputs]), dtype=torch.float32)

    # Future inputs for entire horizon
    fut_inputs_full_raw = np.column_stack([
        b_test["cooling_rate"][L:],
        np.full(N_steps - L, b_test["epsilon"]),
    ])
    fut_inputs_full_norm = torch.tensor((fut_inputs_full_raw - u_mean) / u_std, dtype=torch.float32)

    # Autoregressive rollouts
    print("--> Rolling out Pure Baseline RNN open-loop...")
    rollout_baseline = model_baseline.autoregressive_rollout(
        initial_history=init_hist_tensor,
        future_inputs_full=fut_inputs_full_norm,
        norm_stats=norm_stats,
        device=device,
    ).numpy()

    print("--> Rolling out Forward PIRNN open-loop...")
    rollout_pirnn = model_pirnn.autoregressive_rollout(
        initial_history=init_hist_tensor,
        future_inputs_full=fut_inputs_full_norm,
        norm_stats=norm_stats,
        device=device,
    ).numpy()

    time_fut_h = b_test["time"][L:] / 3600.0
    time_all_h = b_test["time"] / 3600.0

    # Ground truth future states
    gt_T = b_test["T"][L:]
    gt_c = b_test["c"][L:]
    gt_L1 = b_test["mean_L1"][L:] * 1e6
    gt_L2 = b_test["mean_L2"][L:] * 1e6
    gt_AR = b_test["aspect_ratio"][L:]
    gt_mu11 = b_test["mu11"][L:]

    # Baseline predicted states
    b_T = rollout_baseline[:, 0]
    b_c = rollout_baseline[:, 1]
    b_L1 = rollout_baseline[:, 2] * 1e6
    b_L2 = rollout_baseline[:, 3] * 1e6
    b_AR = rollout_baseline[:, 4]
    b_mu11 = 10.0 ** rollout_baseline[:, 6]

    # PIRNN predicted states
    p_T = rollout_pirnn[:, 0]
    p_c = rollout_pirnn[:, 1]
    p_L1 = rollout_pirnn[:, 2] * 1e6
    p_L2 = rollout_pirnn[:, 3] * 1e6
    p_AR = rollout_pirnn[:, 4]
    p_mu11 = 10.0 ** rollout_pirnn[:, 6]

    # Mass conservation invariant: Total Solute = c(t) + rho_c * kV * mu11(t)
    rho_c_kV = config.RHO_CRYSTAL * config.K_VOL  # ~41.25 kg/m^3
    c0_true = b_test["c"][0]
    mu11_0_true = b_test["mu11"][0]
    total_solute_ideal = c0_true + rho_c_kV * mu11_0_true

    gt_total_solute = gt_c + rho_c_kV * gt_mu11
    b_total_solute = b_c + rho_c_kV * b_mu11
    p_total_solute = p_c + rho_c_kV * p_mu11

    # Mass balance error metrics
    b_mass_err_mean = np.mean(np.abs(b_total_solute - total_solute_ideal))
    b_mass_err_max = np.max(np.abs(b_total_solute - total_solute_ideal))

    p_mass_err_mean = np.mean(np.abs(p_total_solute - total_solute_ideal))
    p_mass_err_max = np.max(np.abs(p_total_solute - total_solute_ideal))

    # Rollout comparison table
    rollout_comparison = [
        {
            "Variable": "Temperature T (°C)",
            "True Final": f"{gt_T[-1]:.2f}",
            "Baseline Final": f"{b_T[-1]:.2f}",
            "PIRNN Final": f"{p_T[-1]:.2f}",
            "Baseline RMSE": f"{np.sqrt(np.mean((gt_T - b_T)**2)):.3f}",
            "PIRNN RMSE": f"{np.sqrt(np.mean((gt_T - p_T)**2)):.3f}",
        },
        {
            "Variable": "Concentration c (kg/m³)",
            "True Final": f"{gt_c[-1]:.2f}",
            "Baseline Final": f"{b_c[-1]:.2f}",
            "PIRNN Final": f"{p_c[-1]:.2f}",
            "Baseline RMSE": f"{np.sqrt(np.mean((gt_c - b_c)**2)):.2f}",
            "PIRNN RMSE": f"{np.sqrt(np.mean((gt_c - p_c)**2)):.2f}",
        },
        {
            "Variable": "Length <L1> (µm)",
            "True Final": f"{gt_L1[-1]:.1f}",
            "Baseline Final": f"{b_L1[-1]:.1f}",
            "PIRNN Final": f"{p_L1[-1]:.1f}",
            "Baseline RMSE": f"{np.sqrt(np.mean((gt_L1 - b_L1)**2)):.1f}",
            "PIRNN RMSE": f"{np.sqrt(np.mean((gt_L1 - p_L1)**2)):.1f}",
        },
        {
            "Variable": "Width <L2> (µm)",
            "True Final": f"{gt_L2[-1]:.1f}",
            "Baseline Final": f"{b_L2[-1]:.1f}",
            "PIRNN Final": f"{p_L2[-1]:.1f}",
            "Baseline RMSE": f"{np.sqrt(np.mean((gt_L2 - b_L2)**2)):.1f}",
            "PIRNN RMSE": f"{np.sqrt(np.mean((gt_L2 - p_L2)**2)):.1f}",
        },
        {
            "Variable": "Aspect Ratio AR (-)",
            "True Final": f"{gt_AR[-1]:.2f}",
            "Baseline Final": f"{b_AR[-1]:.2f}",
            "PIRNN Final": f"{p_AR[-1]:.2f}",
            "Baseline RMSE": f"{np.sqrt(np.mean((gt_AR - b_AR)**2)):.3f}",
            "PIRNN RMSE": f"{np.sqrt(np.mean((gt_AR - p_AR)**2)):.3f}",
        },
        {
            "Variable": "Mass Invariant Err (kg/m³)",
            "True Final": "0.00 (Ideal)",
            "Baseline Final": f"{abs(b_total_solute[-1] - total_solute_ideal):.2f}",
            "PIRNN Final": f"{abs(p_total_solute[-1] - total_solute_ideal):.2f}",
            "Baseline RMSE": f"{b_mass_err_mean:.2f} (Mean)",
            "PIRNN RMSE": f"{p_mass_err_mean:.2f} (Mean)",
        },
    ]

    df_rollout = pd.DataFrame(rollout_comparison)
    print("\n--- OPEN-LOOP ROLLOUT COMPARISON (2.8 Hours Unassisted) ---")
    print(df_rollout.to_string(index=False))

    # 4. Generate High-Res Comparative Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        f"Model Comparison: Baseline Recurrent Model vs Forward PIRNN\n"
        f"Unseen Test Batch #{b_test['batch_id']} ({p_type} Profile, 2.8-Hour Full Rollout)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # 1. Temperature Profile
    axes[0, 0].plot(time_all_h, b_test["T"], "k-", linewidth=2.5, label="Plant Ground Truth")
    axes[0, 0].plot(time_fut_h, b_T, color="#e66101", linestyle="--", linewidth=2.0, label="Baseline GRU")
    axes[0, 0].plot(time_fut_h, p_T, color="#5e3c99", linestyle="-", linewidth=2.0, label="Forward PIRNN")
    axes[0, 0].set_title("a) Slurry Temperature $T(t)$ [°C]", fontweight="bold", fontsize=12)
    axes[0, 0].set_xlabel("Time [h]")
    axes[0, 0].set_ylabel("Temperature [°C]")
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)
    axes[0, 0].legend(frameon=True, facecolor="white", edgecolor="none")

    # 2. Solute Concentration
    axes[0, 1].plot(time_all_h, b_test["c"], "k-", linewidth=2.5, label="Plant Ground Truth")
    axes[0, 1].scatter(time_all_h[::6], b_test["c_meas"][::6], color="gray", alpha=0.5, s=12, label="Noisy Sensor")
    axes[0, 1].plot(time_fut_h, b_c, color="#e66101", linestyle="--", linewidth=2.0, label="Baseline GRU")
    axes[0, 1].plot(time_fut_h, p_c, color="#5e3c99", linestyle="-", linewidth=2.2, label="Forward PIRNN")
    axes[0, 1].set_title("b) Solute Concentration $c(t)$ [kg/m³]", fontweight="bold", fontsize=12)
    axes[0, 1].set_xlabel("Time [h]")
    axes[0, 1].set_ylabel("Concentration [kg/m³]")
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)
    axes[0, 1].legend(frameon=True, facecolor="white", edgecolor="none")

    # 3. Crystal Length L1
    axes[0, 2].plot(time_all_h, b_test["mean_L1"] * 1e6, "k-", linewidth=2.5, label="Plant Ground Truth")
    axes[0, 2].scatter(time_all_h[::6], b_test["mean_L1_meas"][::6] * 1e6, color="gray", alpha=0.5, s=12, label="Noisy Sensor")
    axes[0, 2].plot(time_fut_h, b_L1, color="#e66101", linestyle="--", linewidth=2.0, label="Baseline GRU")
    axes[0, 2].plot(time_fut_h, p_L1, color="#5e3c99", linestyle="-", linewidth=2.2, label="Forward PIRNN")
    axes[0, 2].set_title("c) Mean Crystal Length $\\langle L_1 \\rangle$ [µm]", fontweight="bold", fontsize=12)
    axes[0, 2].set_xlabel("Time [h]")
    axes[0, 2].set_ylabel("Length $\\langle L_1 \\rangle$ [µm]")
    axes[0, 2].grid(True, linestyle=":", alpha=0.6)
    axes[0, 2].legend(frameon=True, facecolor="white", edgecolor="none")

    # 4. Crystal Width L2
    axes[1, 0].plot(time_all_h, b_test["mean_L2"] * 1e6, "k-", linewidth=2.5, label="Plant Ground Truth")
    axes[1, 0].scatter(time_all_h[::6], b_test["mean_L2_meas"][::6] * 1e6, color="gray", alpha=0.5, s=12, label="Noisy Sensor")
    axes[1, 0].plot(time_fut_h, b_L2, color="#e66101", linestyle="--", linewidth=2.0, label="Baseline GRU")
    axes[1, 0].plot(time_fut_h, p_L2, color="#5e3c99", linestyle="-", linewidth=2.2, label="Forward PIRNN")
    axes[1, 0].set_title("d) Mean Crystal Width $\\langle L_2 \\rangle$ [µm]", fontweight="bold", fontsize=12)
    axes[1, 0].set_xlabel("Time [h]")
    axes[1, 0].set_ylabel("Width $\\langle L_2 \\rangle$ [µm]")
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)
    axes[1, 0].legend(frameon=True, facecolor="white", edgecolor="none")

    # 5. Aspect Ratio AR
    axes[1, 1].plot(time_all_h, b_test["aspect_ratio"], "k-", linewidth=2.5, label="Plant Ground Truth")
    axes[1, 1].plot(time_fut_h, b_AR, color="#e66101", linestyle="--", linewidth=2.0, label="Baseline GRU")
    axes[1, 1].plot(time_fut_h, p_AR, color="#5e3c99", linestyle="-", linewidth=2.2, label="Forward PIRNN")
    axes[1, 1].set_title("e) Crystal Aspect Ratio $\\mathrm{AR} = \\langle L_1 \\rangle / \\langle L_2 \\rangle$ [-]", fontweight="bold", fontsize=12)
    axes[1, 1].set_xlabel("Time [h]")
    axes[1, 1].set_ylabel("Aspect Ratio [-]")
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)
    axes[1, 1].legend(frameon=True, facecolor="white", edgecolor="none")

    # 6. Mass Balance Invariant
    axes[1, 2].axhline(total_solute_ideal, color="black", linestyle="-", linewidth=2.5, label="Physical Conservation Law")
    axes[1, 2].plot(time_fut_h, b_total_solute, color="#e66101", linestyle="--", linewidth=2.0, label="Baseline GRU (Drifting)")
    axes[1, 2].plot(time_fut_h, p_total_solute, color="#5e3c99", linestyle="-", linewidth=2.2, label="Forward PIRNN (Conserved)")
    axes[1, 2].set_title("f) Solute Mass Invariant $c + \\rho_c k_V \\mu_{11}$ [kg/m³]", fontweight="bold", fontsize=12)
    axes[1, 2].set_xlabel("Time [h]")
    axes[1, 2].set_ylabel("Total Solute Mass [kg/m³]")
    axes[1, 2].grid(True, linestyle=":", alpha=0.6)
    axes[1, 2].legend(frameon=True, facecolor="white", edgecolor="none")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_file = os.path.join(output_dir, "baseline_vs_pirnn_comparison.png")
    plt.savefig(plot_file, dpi=200)
    plt.close()
    print(f"\n--> Successfully saved comprehensive comparison figure to: {plot_file}")

    # Return metric tables
    return df_test_metrics, df_rollout


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Baseline RNN vs Forward PIRNN")
    parser.add_argument("--baseline", type=str, default="PIRNN/checkpoints/baseline_gru_best.pt")
    parser.add_argument("--pirnn", type=str, default="PIRNN/checkpoints/forward_pirnn_best.pt")
    parser.add_argument("--test_data", type=str, default="PIRNN/data/crystallizer_test.pt")
    args = parser.parse_args()

    compare_models(
        baseline_ckpt_path=args.baseline,
        pirnn_ckpt_path=args.pirnn,
        test_data_path=args.test_data,
    )
