"""
Export Full Timestamp Time-Series and Generate Standardized Benchmark Trajectories.

1. Exports individual timestamped CSVs for each batch into PIRNN/data/timeseries/
2. Creates PIRNN/data/starting_conditions.csv (explicit t=0 states)
3. Simulates canonical benchmark batches with IDENTICAL starting conditions across all 6 profile families
4. Generates an apples-to-apples comparison plot of all profiles starting from the exact same initial state.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from PIRNN.plant_simulator import DynamicCrystallizerPlant
from PIRNN.temperature_profiles import (
    LinearProfile,
    CubicProfile,
    NaturalProfile,
    TwoStageProfile,
    CoolHoldCoolProfile,
    RandomMonotoneSplineProfile,
)


def export_existing_batches_to_timeseries(
    data_dir: str = "PIRNN/data",
    output_ts_dir: str = "PIRNN/data/timeseries",
):
    """Exports full timestamp-by-timestamp CSVs for all train/val/test batches."""
    os.makedirs(output_ts_dir, exist_ok=True)

    splits = ["train", "val", "test"]
    all_batches = []
    for s in splits:
        path = os.path.join(data_dir, f"crystallizer_{s}.pt")
        if os.path.exists(path):
            data = torch.load(path, weights_only=False)
            all_batches.extend(data["batches"])

    all_batches = sorted(all_batches, key=lambda b: b["batch_id"])
    print(f"--> Exporting timestamped time-series for {len(all_batches)} batches...")

    starting_conditions_rows = []
    combined_rows = []

    for b in all_batches:
        b_id = b["batch_id"]
        p_type = b["profile_type"]
        N = len(b["time"])

        # Construct DataFrame for this batch
        df_batch = pd.DataFrame({
            "batch_id": b_id,
            "time_s": b["time"],
            "time_min": b["time"] / 60.0,
            "time_hr": b["time"] / 3600.0,
            "T_celsius": b["T"],
            "T_meas_celsius": b["T_meas"],
            "cooling_rate_degC_s": b["cooling_rate"],
            "c_kg_m3": b["c"],
            "c_meas_kg_m3": b["c_meas"],
            "cs_kg_m3": b["cs"],
            "S_supersat_ratio": b["S"],
            "sigma_rel_supersat": b["sigma"],
            "mean_L1_um": b["mean_L1"] * 1e6,
            "mean_L1_meas_um": b["mean_L1_meas"] * 1e6,
            "mean_L2_um": b["mean_L2"] * 1e6,
            "mean_L2_meas_um": b["mean_L2_meas"] * 1e6,
            "aspect_ratio": b["aspect_ratio"],
            "aspect_ratio_meas": b["aspect_ratio_meas"],
            "nucleation_rate_m3_s": b["nucleation_rate"],
            "mu00": b["mu00"],
            "mu00_meas": b["mu00_meas"],
            "mu10": b["mu10"],
            "mu01": b["mu01"],
            "mu20": b["mu20"],
            "mu11": b["mu11"],
            "mu02": b["mu02"],
        })

        # Save individual batch CSV
        filename = f"batch_{b_id:03d}_{p_type}.csv"
        df_batch.to_csv(os.path.join(output_ts_dir, filename), index=False)

        # Collect for combined timeseries
        combined_rows.append(df_batch)

        # Record t = 0 starting conditions
        starting_conditions_rows.append({
            "batch_id": b_id,
            "profile_type": p_type,
            "T0_seed_celsius": b["T"][0],
            "c0_solute_kg_m3": b["c"][0],
            "cs0_solubility_kg_m3": b["cs"][0],
            "S0_supersat_ratio": b["S"][0],
            "sigma0_rel_supersat": b["sigma"][0],
            "mean_L1_0_um": b["mean_L1"][0] * 1e6,
            "mean_L2_0_um": b["mean_L2"][0] * 1e6,
            "aspect_ratio_0": b["aspect_ratio"][0],
            "mu00_0": b["mu00"][0],
            "mu11_0": b["mu11"][0],
            "epsilon_W_kg": b["epsilon"],
            "lambda_k1": b["lambda_k1"],
            "lambda_k2": b["lambda_k2"],
            "lambda_kS": b["lambda_kS"],
            "duration_s": b["time"][-1],
            "duration_hr": b["time"][-1] / 3600.0,
            "total_timestamps": N,
        })

    # Save starting conditions CSV
    df_start = pd.DataFrame(starting_conditions_rows)
    df_start.to_csv(os.path.join(data_dir, "starting_conditions.csv"), index=False)
    print(f"--> Saved explicit starting conditions to: {os.path.join(data_dir, 'starting_conditions.csv')}")

    # Save consolidated timeseries CSV (sampled every 60 s)
    df_combined = pd.concat(combined_rows, ignore_index=True)
    df_combined.to_csv(os.path.join(data_dir, "all_batches_timeseries.csv"), index=False)
    print(f"--> Saved consolidated timeseries CSV ({len(df_combined)} rows) to: {os.path.join(data_dir, 'all_batches_timeseries.csv')}")


def generate_benchmark_identical_starting_conditions(output_dir: str = "PIRNN/data/benchmark_identical_start"):
    """
    Simulates all 6 profile families starting from the EXACT same initial conditions:
    Ts = 35.0 °C, Tf = 25.0 °C, c0 = 240.0 kg/m^3, 2% seed (100 um x 60 um), duration = 12000 s, eps = 350 W/kg.
    """
    os.makedirs(output_dir, exist_ok=True)
    plant = DynamicCrystallizerPlant()

    T_seed = 35.0
    T_final = 25.0
    duration = 12000.0
    c0 = 240.0
    seed_mass_frac = 0.02
    L1_seed = 100.0e-6
    L2_seed = 60.0e-6
    epsilon = 350.0

    profiles = {
        "Linear": LinearProfile(T_seed, T_final, duration),
        "Cubic": CubicProfile(T_seed, T_final, duration),
        "Natural": NaturalProfile(T_seed, T_final, duration, decay_factor=2.5),
        "TwoStage": TwoStageProfile(T_seed, T_final, duration, T_mid=30.0, split_frac=0.4),
        "CoolHoldCool": CoolHoldCoolProfile(T_seed, T_final, duration, T_hold=30.5, hold_start_frac=0.25, hold_end_frac=0.60),
        "RandomSpline": RandomMonotoneSplineProfile(T_seed, T_final, duration, num_segments=5, seed=123),
    }

    colors = {
        "Linear": "#1f77b4",
        "Cubic": "#ff7f0e",
        "Natural": "#2ca02c",
        "TwoStage": "#d62728",
        "CoolHoldCool": "#9467bd",
        "RandomSpline": "#8c564b",
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Controlled Benchmark: 6 Temperature Profiles from IDENTICAL Starting Conditions\n(Ts=35.0°C, Tf=25.0°C, c0=240 kg/m³, 2% Seed Loading, Duration=12,000 s)", fontsize=15, fontweight="bold")

    benchmark_summaries = []

    for name, prof in profiles.items():
        traj = plant.simulate_batch(
            profile=prof,
            duration=duration,
            dt_sample=60.0,
            c0=c0,
            seed_mass_fraction=seed_mass_frac,
            mean_L1_seed=L1_seed,
            mean_L2_seed=L2_seed,
            epsilon=epsilon,
        )

        # Save individual benchmark CSV
        df = pd.DataFrame({
            "time_s": traj["time"],
            "time_min": traj["time"] / 60.0,
            "T_celsius": traj["T"],
            "cooling_rate_degC_s": traj["cooling_rate"],
            "c_kg_m3": traj["c"],
            "sigma": traj["sigma"],
            "mean_L1_um": traj["mean_L1"] * 1e6,
            "mean_L2_um": traj["mean_L2"] * 1e6,
            "aspect_ratio": traj["aspect_ratio"],
            "nucleation_rate": traj["nucleation_rate"],
            "mu00": traj["mu00"],
            "mu11": traj["mu11"],
        })
        df.to_csv(os.path.join(output_dir, f"benchmark_{name.lower()}.csv"), index=False)

        benchmark_summaries.append({
            "profile_name": name,
            "initial_T": T_seed,
            "final_T": T_final,
            "initial_c": c0,
            "final_c": traj["c"][-1],
            "initial_L1_um": traj["mean_L1"][0] * 1e6,
            "final_L1_um": traj["mean_L1"][-1] * 1e6,
            "final_L2_um": traj["mean_L2"][-1] * 1e6,
            "final_aspect_ratio": traj["aspect_ratio"][-1],
            "peak_sigma": np.max(traj["sigma"]),
            "final_mu00": traj["mu00"][-1],
        })

        t_hr = traj["time"] / 3600.0
        c_code = colors[name]

        # 1. Temperature
        axes[0, 0].plot(t_hr, traj["T"], color=c_code, linewidth=2.2, label=name)
        # 2. Cooling rate
        axes[0, 1].plot(t_hr, traj["cooling_rate"] * 1000.0, color=c_code, linewidth=2.2, label=name)
        # 3. Relative supersaturation
        axes[0, 2].plot(t_hr, traj["sigma"], color=c_code, linewidth=2.2, label=name)
        # 4. Length L1
        axes[1, 0].plot(t_hr, traj["mean_L1"] * 1e6, color=c_code, linewidth=2.2, label=name)
        # 5. Aspect Ratio
        axes[1, 1].plot(t_hr, traj["aspect_ratio"], color=c_code, linewidth=2.2, label=name)
        # 6. Concentration
        axes[1, 2].plot(t_hr, traj["c"], color=c_code, linewidth=2.2, label=name)

    axes[0, 0].set_title("a) Temperature Profiles $T(t)$ [°C]", fontweight="bold")
    axes[0, 0].set_xlabel("Time [h]")
    axes[0, 0].set_ylabel("Temperature [°C]")
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)
    axes[0, 0].legend(loc="upper right", framealpha=0.9, fontsize=9)

    axes[0, 1].set_title("b) Cooling Rates $cr(t)$ [×10⁻³ °C/s]", fontweight="bold")
    axes[0, 1].set_xlabel("Time [h]")
    axes[0, 1].set_ylabel("Cooling Rate [m°C/s]")
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)

    axes[0, 2].set_title("c) Supersaturation $\\sigma(t)$ [-]", fontweight="bold")
    axes[0, 2].set_xlabel("Time [h]")
    axes[0, 2].set_ylabel("$\\sigma = (c - c_s)/c_s$")
    axes[0, 2].grid(True, linestyle=":", alpha=0.6)

    axes[1, 0].set_title("d) Mean Crystal Length $\\langle L_1 \\rangle$ [µm]", fontweight="bold")
    axes[1, 0].set_xlabel("Time [h]")
    axes[1, 0].set_ylabel("Length $\\langle L_1 \\rangle$ [µm]")
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)

    axes[1, 1].set_title("e) Mean Aspect Ratio $\\langle L_1 \\rangle / \\langle L_2 \\rangle$ [-]", fontweight="bold")
    axes[1, 1].set_xlabel("Time [h]")
    axes[1, 1].set_ylabel("Aspect Ratio [-]")
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)

    axes[1, 2].set_title("f) Solute Concentration $c(t)$ [kg/m³]", fontweight="bold")
    axes[1, 2].set_xlabel("Time [h]")
    axes[1, 2].set_ylabel("Concentration [kg/m³]")
    axes[1, 2].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    comp_plot_path = os.path.join(output_dir, "benchmark_identical_start_comparison.png")
    plt.savefig(comp_plot_path, dpi=180)
    plt.close()

    df_bench = pd.DataFrame(benchmark_summaries)
    df_bench.to_csv(os.path.join(output_dir, "benchmark_summary.csv"), index=False)
    print(f"--> Saved identical starting condition benchmarks to: {output_dir}/")
    print(f"--> Saved comparison figure to: {comp_plot_path}")


if __name__ == "__main__":
    export_existing_batches_to_timeseries()
    generate_benchmark_identical_starting_conditions()
