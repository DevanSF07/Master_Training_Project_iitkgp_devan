"""
Paper Results Reproduction: Batch Cooling Crystallization of Plate-like Crystals.

Reproduces the key simulation findings from:
    Botond Szilagyi and Bela G. Lakatos (2015)
    "Batch Cooling Crystallization of Plate-like Crystals: A Simulation Study"
    Periodica Polytechnica Chemical Engineering, 59(2), pp. 151-158.

Figures Generated:
- Fig. 1: Solute concentration and solubility vs temperature in batch crystallization.
- Fig. 3: Time evolution of mean crystal sizes <L1> and <L2> for various stirring powers.
- Fig. 6: Final mean crystal sizes vs stirring power for different cooling rates.
- Fig. 8 & 9: Final crystal sizes and aspect ratio vs cooling rate for different seeding temperatures.
- Fig. 10: Dynamic evolution of secondary nucleation rate B(t) across seeding temperatures.
- Fig. 11: 3D Phase space trajectory in (mu11, S, RV) subspace.

Author: Devan Singh Faujdar
Master Training Project, IIT Kharagpur
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import config
from crystallizer_mom import BatchCrystallizerMOM, calculate_solubility

# Ensure output directory exists
OUTPUT_DIR = "plots/paper_reproduction"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Styling configuration for publication-quality figures
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


def generate_figure_1(simulator: BatchCrystallizerMOM):
    """Fig. 1: Solute concentration and solubility vs temperature."""
    print("Generating Figure 1: Solute concentration and solubility vs temperature...")
    
    # 1. Full equilibrium solubility line from 25.0 to 35.2 °C
    T_sol = np.linspace(25.0, 35.2, 200)
    cs_sol = np.array([calculate_solubility(t) for t in T_sol])

    # 2. Main cooling trajectory from seeding temp (Ts = 34.0 °C) down to 25.0 °C
    res = simulator.simulate(T_seed=34.0, T_final=25.0, cooling_rate=config.COOLING_RATE_NOMINAL, num_points=120)

    # 3. Construct exact actual concentration path:
    #    a) Pre-cooling from Th = 35.2 °C to Ts = 34.0 °C at constant c = 240.0 kg/m^3
    #    b) Vertical desaturation drop upon seeding at 34.0 °C down to near solubility line
    #    c) Dynamic cooling trajectory from 34.0 °C to 25.0 °C
    cs_at_seeding = float(calculate_solubility(34.0))
    T_actual = np.concatenate([
        [35.2, 34.0],              # Pre-cooling at constant concentration
        [34.0],                    # Rapid desaturation drop at seeding
        res["temperature"]         # Cooling desaturation path down to 25.0 °C
    ])
    c_actual = np.concatenate([
        [240.0, 240.0],            # Pre-cooling
        [cs_at_seeding],           # Drops to solubility at seeding
        res["concentration"]       # Trajectory
    ])

    plt.figure(figsize=(7, 5))
    # Blue solid actual concentration trajectory
    plt.plot(T_actual, c_actual, color="#1e3a8a", linewidth=2.2, label="Actual concentration")
    # Red dashed solubility curve
    plt.plot(T_sol, cs_sol, color="#dc2626", linestyle="--", linewidth=2.0, label="Solubility line")
    # Green circle at concentrated hot solution (35.2 °C, 240 kg/m^3)
    plt.plot([35.2], [240.0], color="#16a34a", marker="o", markersize=6.5, linestyle="None", label="Concentrated solution")
    # Purple star at seeding point (34.0 °C, 240 kg/m^3)
    plt.plot([34.0], [240.0], color="#9333ea", marker="*", markersize=11.0, linestyle="None", label="Seeding")

    plt.xlabel("Temperature [°C]", fontsize=12)
    plt.ylabel("Concentration [kg/m³]", fontsize=12)
    plt.title("Fig. 1: Solute Concentration and Solubility vs Temperature", fontsize=13)
    plt.xlim(25.0, 35.5)
    plt.ylim(60, 255)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="#cbd5e1")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig1_concentration_solubility.png"), dpi=300)
    plt.close()


def generate_figure_3(simulator: BatchCrystallizerMOM):
    """Fig. 3: Time evolution of mean crystal sizes for different stirring powers."""
    print("Generating Figure 3: Time evolution of mean crystal sizes...")
    stirring_powers = [250, 350, 450, 550]
    
    # Exact styling from Figure 3 of the paper:
    styles = [
        {"color": "#1e3a8a", "linestyle": "-",  "marker": "o", "label": "ε = 250 W/kg"},
        {"color": "#b91c1c", "linestyle": ":",  "marker": "o", "label": "ε = 350 W/kg"},
        {"color": "#15803d", "linestyle": "-.", "marker": "o", "label": "ε = 450 W/kg"},
        {"color": "#7e22ce", "linestyle": "--", "marker": "s", "label": "ε = 550 W/kg"},
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 9.5), sharex=True)

    for idx, eps in enumerate(stirring_powers):
        res = simulator.simulate(epsilon=float(eps), num_points=120)
        t = res["time"]
        L1_scaled = res["mean_L1"] * 1.0e4  # in units of 10^-4 m
        L2_scaled = res["mean_L2"] * 1.0e4

        st = styles[idx]
        mark_step = max(1, len(t) // 10)
        ax1.plot(t, L1_scaled, color=st["color"], linestyle=st["linestyle"],
                 marker=st["marker"], markevery=mark_step, markersize=6, label=st["label"], linewidth=2.0)
        ax2.plot(t, L2_scaled, color=st["color"], linestyle=st["linestyle"],
                 marker=st["marker"], markevery=mark_step, markersize=6, label=st["label"], linewidth=2.0)

    # Subplot (a) - Mean Length
    ax1.set_ylabel("⟨L₁⟩ [m]", fontsize=12)
    ax1.text(0.0, 1.02, r"$\times 10^{-4}$", transform=ax1.transAxes, fontsize=11)
    ax1.set_title("a)", fontsize=13, y=-0.22)
    ax1.set_xlim(0, 12000)
    ax1.set_ylim(1.0, 5.0)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="#cbd5e1")

    # Subplot (b) - Mean Width
    ax2.set_xlabel("Time [s]", fontsize=12)
    ax2.set_ylabel("⟨L₂⟩ [m]", fontsize=12)
    ax2.text(0.0, 1.02, r"$\times 10^{-4}$", transform=ax2.transAxes, fontsize=11)
    ax2.set_title("b)", fontsize=13, y=-0.26)
    ax2.set_xlim(0, 12000)
    ax2.set_ylim(0.5, 2.5)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="#cbd5e1")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig3_mean_crystal_sizes.png"), dpi=300)
    plt.close()


def generate_figure_6(simulator: BatchCrystallizerMOM):
    """Fig. 6: Effects of stirring power and cooling rate on final product properties."""
    print("Generating Figure 6: Effects of stirring power and cooling rate...")
    cooling_rates = [0.83e-3, 1.67e-3, 2.78e-3, 8.33e-3]
    cr_labels = ["0.83 × 10⁻³ °C/s", "1.67 × 10⁻³ °C/s", "2.78 × 10⁻³ °C/s", "8.33 × 10⁻³ °C/s"]
    eps_values = np.linspace(200, 500, 7)
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    markers = ["o", "s", "^", "d"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for cr_idx, cr in enumerate(cooling_rates):
        final_L1 = []
        final_L2 = []
        for eps in eps_values:
            res = simulator.simulate(cooling_rate=cr, epsilon=eps, num_points=80)
            final_L1.append(res["mean_L1"][-1] * 1.0e4)
            final_L2.append(res["mean_L2"][-1] * 1.0e4)

        ax1.plot(eps_values, final_L1, color=colors[cr_idx], marker=markers[cr_idx],
                 label=f"cr = {cr_labels[cr_idx]}")
        ax2.plot(eps_values, final_L2, color=colors[cr_idx], marker=markers[cr_idx],
                 label=f"cr = {cr_labels[cr_idx]}")

    ax1.set_xlabel("ε [W/kg]")
    ax1.set_ylabel("⟨L₁⟩ [m] × 10⁻⁴")
    ax1.set_title("Fig. 6(b): Final Mean Length vs Stirring Energy")
    ax1.set_ylim(3.5, 5.3)
    ax1.grid(True, linestyle="--")
    ax1.legend(loc="upper right", frameon=True)

    ax2.set_xlabel("ε [W/kg]")
    ax2.set_ylabel("⟨L₂⟩ [m] × 10⁻⁴")
    ax2.set_title("Fig. 6(a): Final Mean Width vs Stirring Energy")
    ax2.set_ylim(1.3, 2.7)
    ax2.grid(True, linestyle="--")
    ax2.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig6_cooling_stirring_effects.png"), dpi=300)
    plt.close()


def generate_figures_8_and_9(simulator: BatchCrystallizerMOM):
    """Fig. 8 & 9: Effects of cooling rate and seeding temperature."""
    print("Generating Figures 8 & 9: Effects of seeding temperature and cooling rate...")
    seeding_temps = [29.0, 31.0, 33.0, 35.0]
    cr_values = np.linspace(1.0e-3, 8.33e-3, 8)
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    markers = ["o", "s", "^", "d"]

    fig_sizes, (ax_l1, ax_l2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig_ar, ax_ar = plt.subplots(figsize=(8, 5.5))

    for idx, Ts in enumerate(seeding_temps):
        l1_list = []
        l2_list = []
        ar_list = []
        for cr in cr_values:
            res = simulator.simulate(T_seed=Ts, T_final=25.0, cooling_rate=cr, num_points=80)
            l1_final = res["mean_L1"][-1] * 1.0e4
            l2_final = res["mean_L2"][-1] * 1.0e4
            ar_final = l1_final / l2_final

            l1_list.append(l1_final)
            l2_list.append(l2_final)
            ar_list.append(ar_final)

        cr_display = cr_values * 1.0e3
        ax_l1.plot(cr_display, l1_list, color=colors[idx], marker=markers[idx],
                   label=f"Ts = {int(Ts)} °C")
        ax_l2.plot(cr_display, l2_list, color=colors[idx], marker=markers[idx],
                   label=f"Ts = {int(Ts)} °C")
        ax_ar.plot(cr_display, ar_list, color=colors[idx], marker=markers[idx],
                   label=f"Ts = {int(Ts)} °C")

    ax_l1.set_xlabel("cr [× 10⁻³ °C/s]")
    ax_l1.set_ylabel("⟨L₁⟩ [m] × 10⁻⁴")
    ax_l1.set_title("Fig. 8(a): Final Length vs Cooling Rate")
    ax_l1.set_ylim(3.4, 5.2)
    ax_l1.grid(True, linestyle="--")
    ax_l1.legend(loc="lower left", frameon=True)

    ax_l2.set_xlabel("cr [× 10⁻³ °C/s]")
    ax_l2.set_ylabel("⟨L₂⟩ [m] × 10⁻⁴")
    ax_l2.set_title("Fig. 8(b): Final Width vs Cooling Rate")
    ax_l2.set_ylim(0.8, 2.6)
    ax_l2.grid(True, linestyle="--")
    ax_l2.legend(loc="lower left", frameon=True)

    fig_sizes.tight_layout()
    fig_sizes.savefig(os.path.join(OUTPUT_DIR, "fig8_product_sizes_vs_Ts.png"), dpi=300)
    plt.close(fig_sizes)

    ax_ar.set_xlabel("cr [× 10⁻³ °C/s]")
    ax_ar.set_ylabel("⟨L₁⟩ / ⟨L₂⟩ [-]")
    ax_ar.set_title("Fig. 9: Final Aspect Ratio vs Cooling Rate")
    ax_ar.set_ylim(1.8, 4.2)
    ax_ar.grid(True, linestyle="--")
    ax_ar.legend(loc="upper right", frameon=True)
    fig_ar.tight_layout()
    fig_ar.savefig(os.path.join(OUTPUT_DIR, "fig9_aspect_ratio_vs_Ts.png"), dpi=300)
    plt.close(fig_ar)


def generate_figure_10(simulator: BatchCrystallizerMOM):
    """Fig. 10: Dynamic evolution of secondary nucleation rate across seeding temperatures."""
    print("Generating Figure 10: Time evolution of nucleation rate...")
    seeding_temps = [29.0, 31.0, 33.0, 35.0]
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

    plt.figure(figsize=(8, 5.5))

    for idx, Ts in enumerate(seeding_temps):
        res = simulator.simulate(T_seed=Ts, T_final=25.0, cooling_rate=config.COOLING_RATE_NOMINAL, num_points=250)
        t = np.maximum(res["time"], 1.0e-4)
        B = np.maximum(res["nucleation_rate"], 1.0e4)

        plt.loglog(t, B, color=colors[idx], label=f"Ts = {int(Ts)} °C")

    plt.xlabel("Time [s]")
    plt.ylabel("Nucleation rate [# m⁻³ s⁻¹]")
    plt.title("Fig. 10: Time Evolution of Secondary Nucleation Rate")
    plt.xlim(1.0e-4, 2.0e4)
    plt.ylim(1.0e5, 1.0e12)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig10_nucleation_rate_evolution.png"), dpi=300)
    plt.close()


def generate_figure_11(simulator: BatchCrystallizerMOM):
    """Fig. 11: 3D phase space projection in (mu11, S, RV) subspace."""
    print("Generating Figure 11: 3D Phase space trajectory...")
    seeding_temps = [29.0, 31.0, 33.0, 35.0]
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    for idx, Ts in enumerate(seeding_temps):
        res = simulator.simulate(T_seed=Ts, T_final=25.0, cooling_rate=config.COOLING_RATE_NOMINAL, num_points=200)
        S = res["supersaturation_ratio"]
        RV_scaled = res["volumetric_growth_rate"] * config.V_CRYSTALLIZER  # In crystallizer volume scale [m^3/s]
        mu11_scaled = res["mu11"] * 1.0e-4  # In 10^4 m^2/m^3

        ax.plot(S, RV_scaled, mu11_scaled, color=colors[idx], label=f"Ts = {int(Ts)} °C", linewidth=2)

    ax.set_xlabel("Supersaturation Ratio S [-]", labelpad=10)
    ax.set_ylabel("Growth Rate RV [m³/s]", labelpad=10)
    ax.set_zlabel("μ₁₁ [m²/m³] × 10⁴", labelpad=10)
    ax.set_title("Fig. 11: 3D Trajectory in (μ₁₁, S, RV) Subspace", pad=20)
    ax.view_init(elev=25, azim=45)
    ax.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig11_3d_phase_trajectory.png"), dpi=300)
    plt.close()


def main():
    print("=" * 70)
    print("REPRODUCING PAPER RESULTS: SZILAGYI & LAKATOS (2015)")
    print("=" * 70)

    simulator = BatchCrystallizerMOM()

    generate_figure_1(simulator)
    generate_figure_3(simulator)
    generate_figure_6(simulator)
    generate_figures_8_and_9(simulator)
    generate_figure_10(simulator)
    generate_figure_11(simulator)

    print("\nAll reproduction figures successfully generated and saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
