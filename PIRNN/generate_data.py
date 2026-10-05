"""
CLI Script to Generate, Validate, and Visualize the Crystallization PIRNN Dataset.

Usage:
    .venv/bin/python3 PIRNN/generate_data.py --num_batches 60
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from PIRNN.data_generator import CrystallizationDatasetGenerator


def plot_dataset_summary(batch_bundle: dict, output_path: str = "PIRNN/data/dataset_verification.png"):
    """Visualizes sample batches across profile families and confirms physical bounds."""
    all_batches = batch_bundle["train_batches"] + batch_bundle["val_batches"] + batch_bundle["test_batches"]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("PIRNN Crystallization Dataset: Multi-Profile Synthesis & Dynamic Trajectories", fontsize=16, fontweight="bold")

    family_colors = {
        "linear": "#1f77b4",
        "cubic": "#ff7f0e",
        "natural": "#2ca02c",
        "two_stage": "#d62728",
        "cool_hold_cool": "#9467bd",
        "random_spline": "#8c564b",
    }

    plotted_labels = set()

    for b in all_batches[:35]:  # plot subset of 35 batches for clear visual clarity
        t_hr = b["time"] / 3600.0
        p_type = b["profile_type"]
        color = family_colors.get(p_type, "gray")
        label = p_type if p_type not in plotted_labels else None
        if label:
            plotted_labels.add(p_type)

        # 1. Temperature trajectories
        axes[0, 0].plot(t_hr, b["T_meas"], color=color, alpha=0.6, linewidth=1.5, label=label)

        # 2. Cooling rates
        axes[0, 1].plot(t_hr, b["cooling_rate"] * 1000.0, color=color, alpha=0.6, linewidth=1.5)

        # 3. Relative supersaturation sigma
        axes[0, 2].plot(t_hr, b["sigma"], color=color, alpha=0.6, linewidth=1.5)

        # 4. Product Lengths <L1>
        axes[1, 0].plot(t_hr, b["mean_L1_meas"] * 1e6, color=color, alpha=0.6, linewidth=1.5)

        # 5. Aspect Ratio
        axes[1, 1].plot(t_hr, b["aspect_ratio_meas"], color=color, alpha=0.6, linewidth=1.5)

        # 6. Solute Concentration
        axes[1, 2].plot(t_hr, b["c_meas"], color=color, alpha=0.6, linewidth=1.5)

    # Subplot formatting
    axes[0, 0].set_title("a) Temperature Profiles $T(t)$ [°C]", fontweight="bold")
    axes[0, 0].set_xlabel("Time [h]")
    axes[0, 0].set_ylabel("Temperature [°C]")
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)
    axes[0, 0].legend(loc="upper right", framealpha=0.9, fontsize=9)

    axes[0, 1].set_title("b) Cooling Rates $cr(t)$ [×10⁻³ °C/s]", fontweight="bold")
    axes[0, 1].set_xlabel("Time [h]")
    axes[0, 1].set_ylabel("Cooling Rate [m°C/s]")
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)

    axes[0, 2].set_title("c) Relative Supersaturation $\\sigma(t)$ [-]", fontweight="bold")
    axes[0, 2].set_xlabel("Time [h]")
    axes[0, 2].set_ylabel("$\\sigma = (c - c_s)/c_s$ [-]")
    axes[0, 2].grid(True, linestyle=":", alpha=0.6)

    axes[1, 0].set_title("d) Mean Crystal Length $\\langle L_1 \\rangle$ [µm]", fontweight="bold")
    axes[1, 0].set_xlabel("Time [h]")
    axes[1, 0].set_ylabel("$\\langle L_1 \\rangle$ [µm]")
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
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"--> Diagnostic plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate PIRNN Crystallization Dataset")
    parser.add_argument("--num_batches", type=int, default=60, help="Total number of batches to synthesize")
    parser.add_argument("--output_dir", type=str, default="PIRNN/data", help="Output directory")
    args = parser.parse_args()

    generator = CrystallizationDatasetGenerator(output_dir=args.output_dir)
    bundle = generator.generate_batch_dataset(num_batches=args.num_batches)

    # Plot verification
    plot_dataset_summary(bundle, output_path=os.path.join(args.output_dir, "dataset_verification.png"))

    # Print summary statistics
    meta_path = os.path.join(args.output_dir, "batch_metadata.csv")
    df = pd.read_csv(meta_path)
    print("\n================ DATASET SYNTHESIS SUMMARY ================")
    print(f"Total Batches Generated: {len(df)}")
    print(f"Profile Distribution:")
    print(df["profile_type"].value_counts().to_string())
    print("\nPhysical Properties (Mean ± Std):")
    print(f"  Duration:         {df['duration_s'].mean():.1f} ± {df['duration_s'].std():.1f} s")
    print(f"  Seeding Temp Ts:  {df['T_seed'].mean():.2f} ± {df['T_seed'].std():.2f} °C")
    print(f"  Final Temp Tf:    {df['T_final'].mean():.2f} ± {df['T_final'].std():.2f} °C")
    print(f"  Final Length L1:  {df['final_mean_L1_um'].mean():.1f} ± {df['final_mean_L1_um'].std():.1f} µm")
    print(f"  Final Width L2:   {df['final_mean_L2_um'].mean():.1f} ± {df['final_mean_L2_um'].std():.1f} µm")
    print(f"  Final Aspect Ratio: {df['final_aspect_ratio'].mean():.2f} ± {df['final_aspect_ratio'].std():.2f}")
    print("===========================================================\n")


if __name__ == "__main__":
    main()
