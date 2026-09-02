"""
Comparative Performance Analysis: PI-RNN vs Black-Box RNN vs First-Principles Benchmark.

Evaluates:
1. Multi-step recursive forecasting on unseen test crystallization batches.
2. Error metrics: MSE, RMSE, MAE, R^2 score.
3. Physics consistency: Solute Mass Balance Residual Error.
4. Generates side-by-side comparison plots, parity plots, and residual error distributions.

Author: Devan Singh Faujdar
Master Training Project, IIT Kharagpur
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import config
from crystallizer_mom import BatchCrystallizerMOM, calculate_solubility, calculate_growth_rates
from pirnn_model import PIRNNModel

CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "plots/pirnn_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Styling configuration
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 2,
    "grid.alpha": 0.4,
})


def load_models_and_scalers(device: str = "cpu"):
    """Loads scalers and model checkpoints."""
    scaler_path = os.path.join(CHECKPOINT_DIR, "scalers.npz")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scalers not found at {scaler_path}. Run train_pirnn.py first.")

    loaded = np.load(scaler_path)
    scalers = {
        "state_mean": loaded["state_mean"],
        "state_std": loaded["state_std"],
        "input_mean": loaded["input_mean"],
        "input_std": loaded["input_std"],
    }

    # Initialize models
    pirnn = PIRNNModel(scalers=scalers, physics_weight=config.PI_RNN_CONFIG["physics_weight"])
    bbrnn = PIRNNModel(scalers=scalers, physics_weight=0.0)

    pirnn_path = os.path.join(CHECKPOINT_DIR, "pirnn_crystallizer.pth")
    bbrnn_path = os.path.join(CHECKPOINT_DIR, "bbrnn_crystallizer.pth")

    pirnn.load_state_dict(torch.load(pirnn_path, map_location=device, weights_only=True))
    bbrnn.load_state_dict(torch.load(bbrnn_path, map_location=device, weights_only=True))

    pirnn.to(device).eval()
    bbrnn.to(device).eval()

    return pirnn, bbrnn, scalers


def multi_step_rollout(
    model: PIRNNModel,
    initial_sequence: np.ndarray,
    full_inputs: np.ndarray,
    scalers: dict,
    total_steps: int,
    device: str = "cpu",
) -> np.ndarray:
    """
    Performs recursive multi-step forecasting across the full batch time horizon.
    """
    state_mean = scalers["state_mean"]
    state_std = scalers["state_std"]
    input_mean = scalers["input_mean"]
    input_std = scalers["input_std"]

    norm_mean = np.hstack([state_mean, input_mean])
    norm_std = np.hstack([state_std, input_std])

    # Current window buffer (shape: seq_len, 8)
    cur_window = initial_sequence.copy()
    predictions = [cur_window[-1, :6]]  # Starting at initial state

    seq_len = len(initial_sequence)

    with torch.no_grad():
        for t_step in range(seq_len, total_steps):
            # Normalize window
            norm_in = (cur_window - norm_mean) / norm_std
            tensor_in = torch.tensor(norm_in[np.newaxis, ...], dtype=torch.float32, device=device)

            # Predict next step
            norm_pred = model(tensor_in)
            # Take the prediction at the last sequence position
            next_state_norm = norm_pred[0, -1, :].cpu().numpy()
            next_state = next_state_norm * state_std + state_mean

            # Physical non-negativity clipping: [c, T, L1*1e4, L2*1e4, mu11*1e-4, mu00*1e-11]
            next_state[0] = np.maximum(next_state[0], 40.0)    # c >= 40 kg/m^3
            next_state[1] = np.maximum(next_state[1], 20.0)    # T >= 20 °C
            next_state[2] = np.maximum(next_state[2], 0.5)     # L1 >= 0.5 x 10^-4 m
            next_state[3] = np.maximum(next_state[3], 0.4)     # L2 >= 0.4 x 10^-4 m
            next_state[4] = np.maximum(next_state[4], 0.05)    # mu11 >= 0.05 x 10^4 m^2/m^3
            next_state[5] = np.maximum(next_state[5], 1.0)     # mu00 >= 1.0 x 10^11 #/m^3

            predictions.append(next_state)

            # Roll sliding window forward with the true control input at next step
            next_input = full_inputs[t_step]
            new_row = np.hstack([next_state, next_input])
            cur_window = np.vstack([cur_window[1:], new_row])

    return np.array(predictions)


def evaluate_batch(
    ground_truth: dict,
    pirnn: PIRNNModel,
    bbrnn: PIRNNModel,
    scalers: dict,
    device: str = "cpu",
    seq_len: int = 5,
) -> dict:
    """Evaluates and compares rollout trajectories against ground truth."""
    states_gt = np.column_stack([
        ground_truth["concentration"],
        ground_truth["temperature"],
        ground_truth["mean_L1"] * 1.0e4,
        ground_truth["mean_L2"] * 1.0e4,
        ground_truth["mu11"] * 1.0e-4,
        ground_truth["mu00"] * 1.0e-11,
    ])

    inputs_gt = np.column_stack([
        np.full(len(states_gt), ground_truth["cooling_rate"] * 1.0e3),
        np.full(len(states_gt), ground_truth["epsilon"] / 100.0),
    ])

    combined_gt = np.hstack([states_gt, inputs_gt])
    initial_window = combined_gt[:seq_len]

    # Run multi-step rollouts
    pirnn_pred_states = multi_step_rollout(
        pirnn, initial_window, inputs_gt, scalers, total_steps=len(states_gt), device=device
    )
    bbrnn_pred_states = multi_step_rollout(
        bbrnn, initial_window, inputs_gt, scalers, total_steps=len(states_gt), device=device
    )

    t_eval = ground_truth["time"][seq_len - 1:]
    gt_eval = states_gt[seq_len - 1:]

    # Direct physical crystal sizes:
    pirnn_L1 = pirnn_pred_states[:, 2] * 1.0e-4
    pirnn_L2 = pirnn_pred_states[:, 3] * 1.0e-4
    pirnn_AR = pirnn_L1 / np.maximum(pirnn_L2, 1e-9)

    bbrnn_L1 = bbrnn_pred_states[:, 2] * 1.0e-4
    bbrnn_L2 = bbrnn_pred_states[:, 3] * 1.0e-4
    bbrnn_AR = bbrnn_L1 / np.maximum(bbrnn_L2, 1e-9)

    gt_L1 = gt_eval[:, 2] * 1.0e-4
    gt_L2 = gt_eval[:, 3] * 1.0e-4
    gt_AR = gt_L1 / np.maximum(gt_L2, 1e-9)

    # Compute Mass Conservation Residuals: |dc/dt + rho_c * RV|
    dt = t_eval[1] - t_eval[0]
    rhoc = config.RHO_CRYSTAL
    kV = config.KV_SHAPE

    def calculate_mass_residual(pred_c, pred_T, pred_L1, pred_L2, pred_mu00):
        cs = np.array([calculate_solubility(temp) for temp in pred_T])
        sig = np.maximum(0.0, (pred_c - cs) / cs)
        g1, g2 = calculate_growth_rates(sig, pred_L1, pred_L2)
        mu10 = pred_L1 * (pred_mu00 * 1.0e11)
        mu01 = pred_L2 * (pred_mu00 * 1.0e11)
        rv = kV * (g1 * mu01 + g2 * mu10)
        dc_dt = np.gradient(pred_c, dt)
        return np.abs(dc_dt + rhoc * rv)

    pirnn_mass_res = calculate_mass_residual(
        pirnn_pred_states[:, 0], pirnn_pred_states[:, 1],
        pirnn_L1, pirnn_L2, pirnn_pred_states[:, 5]
    )
    bbrnn_mass_res = calculate_mass_residual(
        bbrnn_pred_states[:, 0], bbrnn_pred_states[:, 1],
        bbrnn_L1, bbrnn_L2, bbrnn_pred_states[:, 5]
    )

    return {
        "time": t_eval,
        "gt": {
            "c": gt_eval[:, 0],
            "T": gt_eval[:, 1],
            "mu00": gt_eval[:, 5] * 1.0e11,
            "mu11": gt_eval[:, 4] * 1.0e4,
            "L1": gt_L1,
            "L2": gt_L2,
            "AR": gt_AR,
        },
        "pirnn": {
            "c": pirnn_pred_states[:, 0],
            "T": pirnn_pred_states[:, 1],
            "mu00": pirnn_pred_states[:, 5] * 1.0e11,
            "mu11": pirnn_pred_states[:, 4] * 1.0e4,
            "L1": pirnn_L1,
            "L2": pirnn_L2,
            "AR": pirnn_AR,
            "mass_res": pirnn_mass_res,
        },
        "bbrnn": {
            "c": bbrnn_pred_states[:, 0],
            "T": bbrnn_pred_states[:, 1],
            "mu00": bbrnn_pred_states[:, 5] * 1.0e11,
            "mu11": bbrnn_pred_states[:, 4] * 1.0e4,
            "L1": bbrnn_L1,
            "L2": bbrnn_L2,
            "AR": bbrnn_AR,
            "mass_res": bbrnn_mass_res,
        },
    }


def compute_metrics_table(results: dict):
    """Computes comprehensive statistical metrics table."""
    variables = [
        ("c", "Solute Concentration [kg/m³]"),
        ("L1", "Mean Crystal Length ⟨L₁⟩ [m]"),
        ("L2", "Mean Crystal Width ⟨L₂⟩ [m]"),
        ("AR", "Aspect Ratio ⟨L₁⟩/⟨L₂⟩ [-]"),
        ("mu11", "Cross Moment μ₁₁ [m²/m³]"),
    ]

    print("\n" + "=" * 80)
    print(f"{'VARIABLE':<32} | {'MODEL':<8} | {'MSE':<10} | {'RMSE':<10} | {'MAE':<10} | {'R²':<6}")
    print("=" * 80)

    summary = {}
    for key, name in variables:
        y_true = results["gt"][key]
        y_pirnn = results["pirnn"][key]
        y_bbrnn = results["bbrnn"][key]

        # PI-RNN metrics
        mse_pi = mean_squared_error(y_true, y_pirnn)
        rmse_pi = np.sqrt(mse_pi)
        mae_pi = mean_absolute_error(y_true, y_pirnn)
        r2_pi = r2_score(y_true, y_pirnn)

        # BB-RNN metrics
        mse_bb = mean_squared_error(y_true, y_bbrnn)
        rmse_bb = np.sqrt(mse_bb)
        mae_bb = mean_absolute_error(y_true, y_bbrnn)
        r2_bb = r2_score(y_true, y_bbrnn)

        summary[key] = {
            "PIRNN": {"MSE": mse_pi, "RMSE": rmse_pi, "MAE": mae_pi, "R2": r2_pi},
            "BBRNN": {"MSE": mse_bb, "RMSE": rmse_bb, "MAE": mae_bb, "R2": r2_bb},
        }

        print(f"{name:<32} | {'PI-RNN':<8} | {mse_pi:<10.4e} | {rmse_pi:<10.4e} | {mae_pi:<10.4e} | {r2_pi:<6.4f}")
        print(f"{'':<32} | {'BB-RNN':<8} | {mse_bb:<10.4e} | {rmse_bb:<10.4e} | {mae_bb:<10.4e} | {r2_bb:<6.4f}")
        print("-" * 80)

    # Physics consistency comparison
    mean_mass_pi = np.mean(results["pirnn"]["mass_res"])
    mean_mass_bb = np.mean(results["bbrnn"]["mass_res"])
    print(f"\nMean Solute Mass Balance Residual | PI-RNN: {mean_mass_pi:.4e} kg/(m³·s) | BB-RNN: {mean_mass_bb:.4e} kg/(m³·s)")
    print(f"Physical Consistency Improvement: {(1.0 - mean_mass_pi / mean_mass_bb) * 100:.1f}% reduction in physics violation!")
    print("=" * 80)

    return summary


def plot_comparison_figures(results: dict):
    """Plots multi-panel trajectory comparison, parity plots, and mass residuals."""
    t = results["time"]
    gt = results["gt"]
    pi = results["pirnn"]
    bb = results["bbrnn"]

    # 1. Multi-Panel Batch Trajectory Tracking Comparison
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # (a) Solute Concentration
    axes[0, 0].plot(t, gt["c"], "k-", label="Paper Benchmark (MOM)", linewidth=2.5)
    axes[0, 0].plot(t, pi["c"], "b--", label="PI-RNN", linewidth=2)
    axes[0, 0].plot(t, bb["c"], "r:", label="Black-Box RNN", linewidth=2)
    axes[0, 0].set_ylabel("Concentration [kg/m³]")
    axes[0, 0].set_title("(a) Solute Concentration c(t)")
    axes[0, 0].grid(True, linestyle="--")
    axes[0, 0].legend()

    # (b) Mean Length <L1>
    axes[0, 1].plot(t, gt["L1"] * 1e4, "k-", label="Benchmark (MOM)", linewidth=2.5)
    axes[0, 1].plot(t, pi["L1"] * 1e4, "b--", label="PI-RNN", linewidth=2)
    axes[0, 1].plot(t, bb["L1"] * 1e4, "r:", label="Black-Box RNN", linewidth=2)
    axes[0, 1].set_ylabel("⟨L₁⟩ [m] × 10⁻⁴")
    axes[0, 1].set_title("(b) Mean Crystal Length ⟨L₁⟩(t)")
    axes[0, 1].grid(True, linestyle="--")
    axes[0, 1].legend()

    # (c) Mean Width <L2>
    axes[0, 2].plot(t, gt["L2"] * 1e4, "k-", label="Benchmark (MOM)", linewidth=2.5)
    axes[0, 2].plot(t, pi["L2"] * 1e4, "b--", label="PI-RNN", linewidth=2)
    axes[0, 2].plot(t, bb["L2"] * 1e4, "r:", label="Black-Box RNN", linewidth=2)
    axes[0, 2].set_ylabel("⟨L₂⟩ [m] × 10⁻⁴")
    axes[0, 2].set_title("(c) Mean Crystal Width ⟨L₂⟩(t)")
    axes[0, 2].grid(True, linestyle="--")
    axes[0, 2].legend()

    # (d) Aspect Ratio AR
    axes[1, 0].plot(t, gt["AR"], "k-", label="Benchmark (MOM)", linewidth=2.5)
    axes[1, 0].plot(t, pi["AR"], "b--", label="PI-RNN", linewidth=2)
    axes[1, 0].plot(t, bb["AR"], "r:", label="Black-Box RNN", linewidth=2)
    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 0].set_ylabel("Aspect Ratio ⟨L₁⟩/⟨L₂⟩ [-]")
    axes[1, 0].set_title("(d) Crystal Aspect Ratio (Shape)")
    axes[1, 0].grid(True, linestyle="--")
    axes[1, 0].legend()

    # (e) Cross Moment mu11
    axes[1, 1].plot(t, gt["mu11"] * 1e-4, "k-", label="Benchmark (MOM)", linewidth=2.5)
    axes[1, 1].plot(t, pi["mu11"] * 1e-4, "b--", label="PI-RNN", linewidth=2)
    axes[1, 1].plot(t, bb["mu11"] * 1e-4, "r:", label="Black-Box RNN", linewidth=2)
    axes[1, 1].set_xlabel("Time [s]")
    axes[1, 1].set_ylabel("μ₁₁ [m²/m³] × 10⁴")
    axes[1, 1].set_title("(e) Surface-Proxy Cross Moment μ₁₁(t)")
    axes[1, 1].grid(True, linestyle="--")
    axes[1, 1].legend()

    # (f) Mass Balance Residual Error
    axes[1, 2].plot(t, pi["mass_res"], "b-", label="PI-RNN Residual", linewidth=2)
    axes[1, 2].plot(t, bb["mass_res"], "r--", label="BB-RNN Residual", linewidth=2)
    axes[1, 2].set_xlabel("Time [s]")
    axes[1, 2].set_ylabel("|dc/dt + ρ_c·RV| [kg/(m³·s)]")
    axes[1, 2].set_title("(f) Instantaneous Mass Balance Error")
    axes[1, 2].set_yscale("log")
    axes[1, 2].grid(True, linestyle="--")
    axes[1, 2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "nominal_batch_comparison.png"), dpi=300)
    plt.close()

    # 2. Parity Plots (Predicted vs Benchmark)
    fig_parity, axes_p = plt.subplots(1, 3, figsize=(15, 4.8))

    # Concentration parity
    axes_p[0].plot(gt["c"], pi["c"], "bo", label="PI-RNN", alpha=0.7)
    axes_p[0].plot(gt["c"], bb["c"], "r^", label="BB-RNN", alpha=0.7)
    lim_c = [min(gt["c"]), max(gt["c"])]
    axes_p[0].plot(lim_c, lim_c, "k--", label="Perfect Agreement")
    axes_p[0].set_xlabel("Ground Truth c [kg/m³]")
    axes_p[0].set_ylabel("Predicted c [kg/m³]")
    axes_p[0].set_title("Parity: Solute Concentration")
    axes_p[0].grid(True, linestyle="--")
    axes_p[0].legend()

    # Length parity
    axes_p[1].plot(gt["L1"] * 1e4, pi["L1"] * 1e4, "bo", label="PI-RNN", alpha=0.7)
    axes_p[1].plot(gt["L1"] * 1e4, bb["L1"] * 1e4, "r^", label="BB-RNN", alpha=0.7)
    lim_l1 = [min(gt["L1"] * 1e4), max(gt["L1"] * 1e4)]
    axes_p[1].plot(lim_l1, lim_l1, "k--", label="Perfect Agreement")
    axes_p[1].set_xlabel("Ground Truth ⟨L₁⟩ [m] × 10⁻⁴")
    axes_p[1].set_ylabel("Predicted ⟨L₁⟩ [m] × 10⁻⁴")
    axes_p[1].set_title("Parity: Mean Length ⟨L₁⟩")
    axes_p[1].grid(True, linestyle="--")
    axes_p[1].legend()

    # Aspect ratio parity
    axes_p[2].plot(gt["AR"], pi["AR"], "bo", label="PI-RNN", alpha=0.7)
    axes_p[2].plot(gt["AR"], bb["AR"], "r^", label="BB-RNN", alpha=0.7)
    lim_ar = [min(gt["AR"]), max(gt["AR"])]
    axes_p[2].plot(lim_ar, lim_ar, "k--", label="Perfect Agreement")
    axes_p[2].set_xlabel("Ground Truth Aspect Ratio [-]")
    axes_p[2].set_ylabel("Predicted Aspect Ratio [-]")
    axes_p[2].set_title("Parity: Crystal Aspect Ratio")
    axes_p[2].grid(True, linestyle="--")
    axes_p[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "parity_plots.png"), dpi=300)
    plt.close()

    print("\nComparison figures saved successfully in:", OUTPUT_DIR)


def main():
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading models onto device: {device}...")

    pirnn, bbrnn, scalers = load_models_and_scalers(device=device)

    # Simulate an unseen test batch (Nominal case from Szilagyi & Lakatos 2015)
    print("Simulating unseen test batch (Nominal operating conditions)...")
    simulator = BatchCrystallizerMOM()
    res_test = simulator.simulate(
        T_seed=config.T_SEED,
        T_final=config.T_FINAL,
        cooling_rate=config.COOLING_RATE_NOMINAL,
        epsilon=config.EPSILON_NOMINAL,
        num_points=100,
    )
    res_test["cooling_rate"] = config.COOLING_RATE_NOMINAL
    res_test["epsilon"] = config.EPSILON_NOMINAL

    # Run multi-step forecasting evaluation
    results = evaluate_batch(res_test, pirnn, bbrnn, scalers, device=device)

    # Compute metric tables and generate publication plots
    compute_metrics_table(results)
    plot_comparison_figures(results)


if __name__ == "__main__":
    main()
