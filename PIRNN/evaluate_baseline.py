"""
Evaluation and Full-Batch Autoregressive Rollout Benchmark for Baseline RNN.

1. Evaluates multi-step ahead metrics on unseen test batches.
2. Performs full-batch open-loop autoregressive rollouts (running 200+ steps without ground-truth feedback).
3. Produces publication-grade diagnostic plots comparing true plant dynamics vs baseline RNN predictions.

Usage:
    .venv/bin/python3 PIRNN/evaluate_baseline.py
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from PIRNN.dataset import CrystallizerWindowDataset
from PIRNN.models import CrystallizerRecurrentModel


def evaluate_baseline_model(
    checkpoint_path: str = "PIRNN/checkpoints/baseline_gru_best.pt",
    test_data_path: str = "PIRNN/data/crystallizer_test.pt",
    output_plot_path: str = "PIRNN/plots/baseline_rollout_evaluation.png",
):
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"--> Evaluating using compute device: {device}")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    norm_stats = ckpt["norm_stats"]
    model = CrystallizerRecurrentModel(
        input_dim=cfg["input_dim"],
        future_input_dim=2,
        target_dim=cfg["target_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        rnn_type=cfg["rnn_type"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"--> Loaded {cfg['rnn_type']} model checkpoint (Epoch {ckpt['epoch']}, Val Loss: {ckpt['best_val_loss']:.6f})")

    # Load test dataset
    test_data = torch.load(test_data_path, weights_only=False)
    test_batches = test_data["batches"]
    test_windows = test_data["windows"]
    test_ds = CrystallizerWindowDataset(test_windows)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False)

    s_mean = norm_stats["state_mean"].to(device)
    s_std = norm_stats["state_std"].to(device)

    # 1. Multi-Step Ahead Error Analysis on Test Windows
    print("\n--- 1. MULTI-STEP ROLLING FORECAST EVALUATION (TEST SET) ---")
    all_abs_errors = []
    with torch.no_grad():
        for hist, fut_in, targ_inc, targ_abs in test_loader:
            hist = hist.to(device)
            fut_in = fut_in.to(device)
            targ_abs = targ_abs.to(device)

            pred_inc = model(hist, fut_in)  # [B, H, 7]
            # Current unnormalized state at end of history:
            curr_state_norm = hist[:, -1, : cfg["target_dim"]]  # [B, 7]
            curr_state = (curr_state_norm * s_std) + s_mean

            # Predicted absolute states: y_{t+k} = y_t + Delta y_{t+k} * s_std
            pred_abs = curr_state.unsqueeze(1) + (pred_inc * s_std)  # [B, H, 7]
            abs_err = torch.abs(pred_abs - targ_abs)  # [B, H, 7]
            all_abs_errors.append(abs_err.cpu())

    all_abs_errors = torch.cat(all_abs_errors, dim=0)  # [N_test, H, 7]
    mae_per_horizon = torch.mean(all_abs_errors, dim=0).numpy()  # [H, 7]
    state_names = norm_stats["state_names"]

    print("Mean Absolute Error (MAE) by Forecast Horizon Step (1 step = 60 s):")
    df_mae = pd.DataFrame(mae_per_horizon, columns=state_names)
    df_mae.index = [f"+{(k+1)*60}s" for k in range(len(df_mae))]
    print(df_mae[["T", "c", "mean_L1", "mean_L2", "aspect_ratio"]].to_string())

    # 2. Full-Batch Autoregressive Open-Loop Rollout on Unseen Test Batch
    print("\n--- 2. FULL-BATCH AUTOREGRESSIVE OPEN-LOOP ROLLOUT ---")
    # Choose representative test batch
    b_test = test_batches[0]
    p_type = b_test["profile_type"]
    N_steps = len(b_test["time"])
    print(f"--> Selected Test Batch #{b_test['batch_id']} ({p_type} profile, {N_steps} timestamps = {b_test['time'][-1]/3600:.1f} hours)")

    # Prepare initial history at t = 0
    L = 5
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

    # Future inputs for the full batch duration
    fut_inputs_full_raw = np.column_stack([
        b_test["cooling_rate"][L:],
        np.full(N_steps - L, b_test["epsilon"]),
    ])
    fut_inputs_full_norm = torch.tensor((fut_inputs_full_raw - u_mean) / u_std, dtype=torch.float32)

    # Execute full open-loop rollout
    pred_full_states = model.autoregressive_rollout(
        initial_history=init_hist_tensor,
        future_inputs_full=fut_inputs_full_norm,
        norm_stats=norm_stats,
        device=device,
    ).numpy()

    time_hist = b_test["time"][:L] / 3600.0
    time_future = b_test["time"][L:] / 3600.0
    time_all = b_test["time"] / 3600.0

    # 3. Create Diagnostic Plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Baseline Data-Driven {cfg['rnn_type']} Model Evaluation on Unseen Test Batch (#{b_test['batch_id']}: {p_type})", fontsize=15, fontweight="bold")

    # a) Temperature Profile
    axes[0, 0].plot(time_all, b_test["T"], color="black", linestyle="--", label="True Plant T(t)")
    axes[0, 0].plot(time_future, pred_full_states[:, 0], color="#1f77b4", label=f"Predicted T(t)")
    axes[0, 0].set_title("a) Temperature Trajectory $T(t)$ [°C]", fontweight="bold")
    axes[0, 0].set_xlabel("Time [h]")
    axes[0, 0].set_ylabel("Temperature [°C]")
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)
    axes[0, 0].legend()

    # b) Mean Length <L1>
    axes[0, 1].plot(time_all, b_test["mean_L1"] * 1e6, color="black", linestyle="--", linewidth=2, label="True Plant ODE")
    axes[0, 1].scatter(time_all[::5], b_test["mean_L1_meas"][::5] * 1e6, color="gray", alpha=0.5, s=15, label="Noisy Sensor")
    axes[0, 1].plot(time_future, pred_full_states[:, 2] * 1e6, color="#d62728", linewidth=2.2, label=f"Autoregressive {cfg['rnn_type']}")
    axes[0, 1].set_title("b) Mean Length $\\langle L_1 \\rangle$ [µm]", fontweight="bold")
    axes[0, 1].set_xlabel("Time [h]")
    axes[0, 1].set_ylabel("$\\langle L_1 \\rangle$ [µm]")
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)
    axes[0, 1].legend()

    # c) Mean Width <L2>
    axes[0, 2].plot(time_all, b_test["mean_L2"] * 1e6, color="black", linestyle="--", linewidth=2, label="True Plant ODE")
    axes[0, 2].scatter(time_all[::5], b_test["mean_L2_meas"][::5] * 1e6, color="gray", alpha=0.5, s=15, label="Noisy Sensor")
    axes[0, 2].plot(time_future, pred_full_states[:, 3] * 1e6, color="#2ca02c", linewidth=2.2, label=f"Autoregressive {cfg['rnn_type']}")
    axes[0, 2].set_title("c) Mean Width $\\langle L_2 \\rangle$ [µm]", fontweight="bold")
    axes[0, 2].set_xlabel("Time [h]")
    axes[0, 2].set_ylabel("$\\langle L_2 \\rangle$ [µm]")
    axes[0, 2].grid(True, linestyle=":", alpha=0.6)
    axes[0, 2].legend()

    # d) Aspect Ratio <L1>/<L2>
    axes[1, 0].plot(time_all, b_test["aspect_ratio"], color="black", linestyle="--", linewidth=2, label="True Plant ODE")
    axes[1, 0].plot(time_future, pred_full_states[:, 4], color="#9467bd", linewidth=2.2, label=f"Autoregressive {cfg['rnn_type']}")
    axes[1, 0].set_title("d) Aspect Ratio $\\langle L_1 \\rangle / \\langle L_2 \\rangle$ [-]", fontweight="bold")
    axes[1, 0].set_xlabel("Time [h]")
    axes[1, 0].set_ylabel("Aspect Ratio [-]")
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)
    axes[1, 0].legend()

    # e) Concentration c(t)
    axes[1, 1].plot(time_all, b_test["c"], color="black", linestyle="--", linewidth=2, label="True Plant ODE")
    axes[1, 1].plot(time_future, pred_full_states[:, 1], color="#ff7f0e", linewidth=2.2, label=f"Autoregressive {cfg['rnn_type']}")
    axes[1, 1].set_title("e) Solute Concentration $c(t)$ [kg/m³]", fontweight="bold")
    axes[1, 1].set_xlabel("Time [h]")
    axes[1, 1].set_ylabel("Concentration [kg/m³]")
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)
    axes[1, 1].legend()

    # f) Forecast Horizon Error Growth (NRMSE across horizon)
    horizon_minutes = np.arange(1, 11)
    norm_err_L1 = (mae_per_horizon[:, 2] / np.mean(b_test["mean_L1"])) * 100.0
    norm_err_c = (mae_per_horizon[:, 1] / np.mean(b_test["c"])) * 100.0
    axes[1, 2].plot(horizon_minutes, norm_err_L1, marker="o", color="#d62728", label="Length $\\langle L_1 \\rangle$ Error (%)")
    axes[1, 2].plot(horizon_minutes, norm_err_c, marker="s", color="#ff7f0e", label="Concentration $c$ Error (%)")
    axes[1, 2].set_title("f) Error Growth Across Prediction Horizon", fontweight="bold")
    axes[1, 2].set_xlabel("Forecast Horizon [Minutes Ahead]")
    axes[1, 2].set_ylabel("Relative Error [%]")
    axes[1, 2].grid(True, linestyle=":", alpha=0.6)
    axes[1, 2].legend()

    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=180)
    plt.close()
    print(f"--> Diagnostic rollout evaluation plot saved to: {output_plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Baseline RNN Model")
    parser.add_argument("--checkpoint", type=str, default="PIRNN/checkpoints/baseline_gru_best.pt")
    args = parser.parse_args()
    evaluate_baseline_model(checkpoint_path=args.checkpoint)
