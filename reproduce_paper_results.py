"""
Paper Results Reproduction: Batch Cooling Crystallization of Plate-like Crystals.
Decoupled Simulation & Visualization Pipeline.

Loads pre-computed, exact 10-state QMOM simulation datasets directly from:
  matlab_simulation_data/
and generates all 10 reproduction figures in:
  plots/paper_reproduction/

Figures Generated:
- Fig. 1: Solute concentration and solubility vs temperature (with pre-cooling & seeding).
- Fig. 3: Time evolution of mean crystal sizes <L1> and <L2> for stirring powers [250, 350, 450, 550] W/kg.
- Fig. 4: Final mean crystal sizes <L1> and <L2> vs seed quantity (1-4%) across 4 seed sizes.
- Fig. 5: Final mean aspect ratio <L1>/<L2> vs seed quantity (1-4%) across 4 seed sizes.
- Fig. 6: Final mean crystal sizes vs stirring power (200-500 W/kg) for 4 cooling rates.
- Fig. 7: Final mean aspect ratio vs stirring power (200-500 W/kg) for 4 cooling rates.
- Fig. 8: Final crystal sizes vs cooling rate (1-8 x 10^-3 °C/s) for 4 seeding temperatures.
- Fig. 9: Final aspect ratio vs cooling rate (1-8 x 10^-3 °C/s) for 4 seeding temperatures.
- Fig. 10: Dynamic evolution of secondary nucleation rate B(t) across seeding temperatures (Log-Log).
- Fig. 11: 3D Phase space trajectory in (mu11, S, RV) subspace.

Author: Devan Singh Faujdar
Master Training Project, IIT Kharagpur
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

DATA_DIR = "matlab_simulation_data"
OUTPUT_DIR = "plots/paper_reproduction"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Publication formatting matching Periodica Polytechnica Chemical Engineering style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 2.0,
    "lines.markersize": 6.0,
    "grid.alpha": 0.4,
})

# Paper color palette
COLOR_BLUE   = "#1e3a8a"    # eps=250, Ts=29, seed=(50,30), cr=0.83
COLOR_RED    = "#dc2626"    # eps=350, Ts=31, seed=(100,60), cr=1.67
COLOR_GREEN  = "#16a34a"    # eps=450, Ts=33, seed=(150,90), cr=2.78
COLOR_PURPLE = "#9333ea"    # eps=550, Ts=35, seed=(200,120), cr=8.33


def load_csv_data(filename: str) -> dict:
    """Load simulation data CSV into a dictionary of 1D numpy arrays."""
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Simulation dataset not found at '{filepath}'. "
            "Please run 'simulate_all_cases_matlab.m' in MATLAB or 'generate_simulation_data.py' first."
        )
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        headers = [h.strip() for h in next(reader)]
        data = {h: [] for h in headers}
        for row in reader:
            if not row:
                continue
            for h, v in zip(headers, row):
                data[h].append(float(v))
    return {h: np.array(v) for h, v in data.items()}


def generate_figure_1():
    """Fig. 1: Solute concentration and solubility vs temperature."""
    print("Generating Figure 1: Solute concentration and solubility vs temperature...")
    data = load_csv_data("fig1_data.csv")

    T = data["Temperature_C"]
    c = data["Concentration_kg_m3"]
    cs = data["Solubility_kg_m3"]

    plt.figure(figsize=(7, 5))
    plt.plot(T, c, color=COLOR_BLUE, linewidth=2.4, label="Actual concentration")
    plt.plot(T, cs, color=COLOR_RED, linestyle="--", linewidth=2.0, label="Solubility line")
    plt.plot([35.2], [240.0], color=COLOR_GREEN, marker="o", markersize=7.0, linestyle="None", label="Concentrated solution")
    plt.plot([34.0], [240.0], color=COLOR_PURPLE, marker="*", markersize=11.0, linestyle="None", label="Seeding")

    plt.xlabel("Temperature [°C]", fontsize=12)
    plt.ylabel("Concentration [kg/m³]", fontsize=12)
    plt.title("Fig. 1: Solute Concentration and Solubility vs Temperature", fontsize=13)
    plt.xlim(24.5, 35.5)
    plt.xticks([25, 27, 29, 31, 33, 35])
    plt.ylim(60, 255)
    plt.yticks([70, 100, 150, 200, 240, 250])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="#cbd5e1")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig1_concentration_solubility.png"), dpi=300)
    plt.close()


def generate_figure_3():
    """Fig. 3: Time evolution of mean crystal sizes <L1> and <L2>."""
    print("Generating Figure 3: Time evolution of mean crystal sizes...")
    data = load_csv_data("fig3_data.csv")

    t = data["Time_s"]
    powers = [250, 350, 450, 550]
    styles = [
        {"color": COLOR_BLUE,   "linestyle": "-",  "marker": "o", "label": "ε = 250 W/kg"},
        {"color": COLOR_RED,    "linestyle": "--", "marker": "s", "label": "ε = 350 W/kg"},
        {"color": COLOR_GREEN,  "linestyle": ":",  "marker": "^", "label": "ε = 450 W/kg"},
        {"color": COLOR_PURPLE, "linestyle": "-.", "marker": "d", "label": "ε = 550 W/kg"},
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    mark_step = max(1, len(t) // 10)

    for p_val, st in zip(powers, styles):
        L1_scaled = data[f"L1_eps{p_val}_m"] * 1.0e4  # in 10^-4 m
        L2_scaled = data[f"L2_eps{p_val}_m"] * 1.0e4

        ax1.plot(t, L1_scaled, color=st["color"], linestyle=st["linestyle"],
                 marker=st["marker"], markevery=mark_step, label=st["label"])
        ax2.plot(t, L2_scaled, color=st["color"], linestyle=st["linestyle"],
                 marker=st["marker"], markevery=mark_step, label=st["label"])

    # Subplot (a) - Mean Length
    ax1.set_xlabel("Time [s]", fontsize=12)
    ax1.set_ylabel("⟨L₁⟩ [m]", fontsize=12)
    ax1.text(0.0, 1.02, r"$\times 10^{-4}$", transform=ax1.transAxes, fontsize=11)
    ax1.set_title("a) Mean Length ⟨L₁⟩", fontsize=13)
    ax1.set_xlim(0, 12000)
    ax1.set_xticks([0, 2000, 4000, 6000, 8000, 10000, 12000])
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="#cbd5e1")

    # Subplot (b) - Mean Width
    ax2.set_xlabel("Time [s]", fontsize=12)
    ax2.set_ylabel("⟨L₂⟩ [m]", fontsize=12)
    ax2.text(0.0, 1.02, r"$\times 10^{-4}$", transform=ax2.transAxes, fontsize=11)
    ax2.set_title("b) Mean Width ⟨L₂⟩", fontsize=13)
    ax2.set_xlim(0, 12000)
    ax2.set_xticks([0, 2000, 4000, 6000, 8000, 10000, 12000])
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="#cbd5e1")

    fig.suptitle("Fig. 3: Time Evolution of Characteristic Mean Crystal Sizes (From Stored QMOM Data)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig3_mean_crystal_sizes.png"), dpi=300)
    plt.close()


def generate_figures_4_and_5():
    """Figs. 4 & 5: Seed properties (size and mass percentage) on product properties."""
    print("Generating Figures 4 & 5: Seed quantity and size effects...")
    data = load_csv_data("fig4_fig5_data.csv")

    q = data["SeedQuantity_pct"]
    configs = [
        {"name": "Seed: (50,30)",   "key": "Seed_50_30",   "color": COLOR_BLUE,   "ls": "-",  "m": "o"},
        {"name": "Seed: (100,60)",  "key": "Seed_100_60",  "color": COLOR_RED,    "ls": "--", "m": "s"},
        {"name": "Seed: (150,90)",  "key": "Seed_150_90",  "color": COLOR_GREEN,  "ls": ":",  "m": "^"},
        {"name": "Seed: (200,120)", "key": "Seed_200_120", "color": COLOR_PURPLE, "ls": "-.", "m": "d"},
    ]

    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(13, 5))
    fig5, ax5 = plt.subplots(figsize=(7, 5))

    for cfg in configs:
        k = cfg["key"]
        l1 = data[f"L1_{k}_m"] * 1.0e4
        l2 = data[f"L2_{k}_m"] * 1.0e4
        ar = data[f"AR_{k}"]

        ax4a.plot(q, l1, color=cfg["color"], linestyle=cfg["ls"], marker=cfg["m"], label=cfg["name"])
        ax4b.plot(q, l2, color=cfg["color"], linestyle=cfg["ls"], marker=cfg["m"], label=cfg["name"])
        ax5.plot(q, ar, color=cfg["color"], linestyle=cfg["ls"], marker=cfg["m"], label=cfg["name"])

    # Figure 4(a) - Final Length
    ax4a.set_xlabel("Seed quantity [% of solute]", fontsize=12)
    ax4a.set_ylabel("⟨L₁⟩ [m]", fontsize=12)
    ax4a.text(0.0, 1.02, r"$\times 10^{-4}$", transform=ax4a.transAxes, fontsize=11)
    ax4a.set_title("a) Final Length ⟨L₁⟩", fontsize=13)
    ax4a.set_xlim(0.8, 4.2)
    ax4a.set_xticks([1, 1.5, 2, 2.5, 3, 3.5, 4])
    ax4a.grid(True, linestyle=":", alpha=0.6)
    ax4a.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#cbd5e1")

    # Figure 4(b) - Final Width
    ax4b.set_xlabel("Seed quantity [% of solute]", fontsize=12)
    ax4b.set_ylabel("⟨L₂⟩ [m]", fontsize=12)
    ax4b.text(0.0, 1.02, r"$\times 10^{-4}$", transform=ax4b.transAxes, fontsize=11)
    ax4b.set_title("b) Final Width ⟨L₂⟩", fontsize=13)
    ax4b.set_xlim(0.8, 4.2)
    ax4b.set_xticks([1, 1.5, 2, 2.5, 3, 3.5, 4])
    ax4b.grid(True, linestyle=":", alpha=0.6)
    ax4b.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#cbd5e1")

    fig4.suptitle("Fig. 4: Variation of the Mean Crystal Sizes with Seed Properties (Stored Data)", fontsize=14)
    fig4.tight_layout()
    fig4.savefig(os.path.join(OUTPUT_DIR, "fig4_seed_properties_sizes.png"), dpi=300)
    plt.close(fig4)

    # Figure 5 - Aspect Ratio
    ax5.set_xlabel("Seed quantity [% of solute]", fontsize=12)
    ax5.set_ylabel(r"⟨L₁⟩/⟨L₂⟩", fontsize=12)
    ax5.set_title("Fig. 5: Variation of Mean Aspect Ratio with Seed Properties", fontsize=13)
    ax5.set_xlim(0.8, 4.2)
    ax5.set_xticks([1, 1.5, 2, 2.5, 3, 3.5, 4])
    ax5.grid(True, linestyle=":", alpha=0.6)
    ax5.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#cbd5e1")
    fig5.tight_layout()
    fig5.savefig(os.path.join(OUTPUT_DIR, "fig5_aspect_ratio_seed_properties.png"), dpi=300)
    plt.close(fig5)


def generate_figures_6_and_7():
    """Figs. 6 & 7: Stirring power and cooling rate sensitivity."""
    print("Generating Figures 6 & 7: Cooling rate and stirring power effects...")
    data = load_csv_data("fig6_fig7_data.csv")

    eps = data["Epsilon_W_kg"]
    configs = [
        {"lbl": "083", "label": "cr = 0.83 · 10⁻³ °C/s", "color": COLOR_BLUE,   "ls": "-",  "m": "o"},
        {"lbl": "167", "label": "cr = 1.67 · 10⁻³ °C/s", "color": COLOR_RED,    "ls": "--", "m": "s"},
        {"lbl": "278", "label": "cr = 2.78 · 10⁻³ °C/s", "color": COLOR_GREEN,  "ls": ":",  "m": "^"},
        {"lbl": "833", "label": "cr = 8.33 · 10⁻³ °C/s", "color": COLOR_PURPLE, "ls": "-.", "m": "d"},
    ]

    fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(13, 5))
    fig7, ax7 = plt.subplots(figsize=(7, 5))

    for cfg in configs:
        lbl = cfg["lbl"]
        l1 = data[f"L1_cr{lbl}_m"] * 1.0e4
        l2 = data[f"L2_cr{lbl}_m"] * 1.0e4
        ar = data[f"AR_cr{lbl}"]

        ax6a.plot(eps, l1, color=cfg["color"], linestyle=cfg["ls"], marker=cfg["m"], label=cfg["label"])
        ax6b.plot(eps, l2, color=cfg["color"], linestyle=cfg["ls"], marker=cfg["m"], label=cfg["label"])
        ax7.plot(eps, ar, color=cfg["color"], linestyle=cfg["ls"], marker=cfg["m"], label=cfg["label"])

    # Figure 6(a) - Final Length
    ax6a.set_xlabel("ε [W/kg]", fontsize=12)
    ax6a.set_ylabel("⟨L₁⟩ [m]", fontsize=12)
    ax6a.text(0.0, 1.02, r"$\times 10^{-4}$", transform=ax6a.transAxes, fontsize=11)
    ax6a.set_title("a) Final Length ⟨L₁⟩", fontsize=13)
    ax6a.set_xlim(180, 520)
    ax6a.set_xticks([200, 250, 300, 350, 400, 450, 500])
    ax6a.grid(True, linestyle=":", alpha=0.6)
    ax6a.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#cbd5e1")

    # Figure 6(b) - Final Width
    ax6b.set_xlabel("ε [W/kg]", fontsize=12)
    ax6b.set_ylabel("⟨L₂⟩ [m]", fontsize=12)
    ax6b.text(0.0, 1.02, r"$\times 10^{-4}$", transform=ax6b.transAxes, fontsize=11)
    ax6b.set_title("b) Final Width ⟨L₂⟩", fontsize=13)
    ax6b.set_xlim(180, 520)
    ax6b.set_xticks([200, 250, 300, 350, 400, 450, 500])
    ax6b.grid(True, linestyle=":", alpha=0.6)
    ax6b.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#cbd5e1")

    fig6.suptitle("Fig. 6: Effects of Stirring Power and Cooling Rate on Product Sizes (Stored Data)", fontsize=14)
    fig6.tight_layout()
    fig6.savefig(os.path.join(OUTPUT_DIR, "fig6_cooling_stirring_effects.png"), dpi=300)
    plt.close(fig6)

    # Figure 7 - Aspect Ratio
    ax7.set_xlabel("ε [W/kg]", fontsize=12)
    ax7.set_ylabel(r"⟨L₁⟩/⟨L₂⟩", fontsize=12)
    ax7.set_title("Fig. 7: Effects of Stirring Power and Cooling Rate on Aspect Ratio", fontsize=13)
    ax7.set_xlim(180, 520)
    ax7.set_xticks([200, 250, 300, 350, 400, 450, 500])
    ax7.grid(True, linestyle=":", alpha=0.6)
    ax7.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#cbd5e1")
    fig7.tight_layout()
    fig7.savefig(os.path.join(OUTPUT_DIR, "fig7_aspect_ratio_cooling_stirring.png"), dpi=300)
    plt.close(fig7)


def generate_figures_8_and_9():
    """Figs. 8 & 9: Effects of cooling rate and seeding temperature."""
    print("Generating Figures 8 & 9: Seeding temperature and cooling rate effects...")
    data = load_csv_data("fig8_fig9_data.csv")

    cr = data["CoolingRate_1e3_C_s"]
    configs = [
        {"ts": 29, "label": "Ts = 29 °C", "color": COLOR_BLUE,   "ls": "-",  "m": "o"},
        {"ts": 31, "label": "Ts = 31 °C", "color": COLOR_RED,    "ls": "--", "m": "s"},
        {"ts": 33, "label": "Ts = 33 °C", "color": COLOR_GREEN,  "ls": ":",  "m": "^"},
        {"ts": 35, "label": "Ts = 35 °C", "color": COLOR_PURPLE, "ls": "-.", "m": "d"},
    ]

    fig8, (ax8a, ax8b) = plt.subplots(1, 2, figsize=(13, 5))
    fig9, ax9 = plt.subplots(figsize=(7, 5))

    for cfg in configs:
        ts = cfg["ts"]
        l1 = data[f"L1_Ts{ts}_m"] * 1.0e4
        l2 = data[f"L2_Ts{ts}_m"] * 1.0e4
        ar = data[f"AR_Ts{ts}"]

        ax8a.plot(cr, l1, color=cfg["color"], linestyle=cfg["ls"], marker=cfg["m"], label=cfg["label"])
        ax8b.plot(cr, l2, color=cfg["color"], linestyle=cfg["ls"], marker=cfg["m"], label=cfg["label"])
        ax9.plot(cr, ar, color=cfg["color"], linestyle=cfg["ls"], marker=cfg["m"], label=cfg["label"])

    # Figure 8(a) - Final Length
    ax8a.set_xlabel("cr [°C/s]", fontsize=12)
    ax8a.text(1.0, -0.15, r"$\times 10^{-3}$", transform=ax8a.transAxes, fontsize=11, ha="right")
    ax8a.set_ylabel("⟨L₁⟩ [m]", fontsize=12)
    ax8a.text(0.0, 1.02, r"$\times 10^{-4}$", transform=ax8a.transAxes, fontsize=11)
    ax8a.set_title("a) Final Length ⟨L₁⟩", fontsize=13)
    ax8a.set_xlim(0.5, 8.8)
    ax8a.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
    ax8a.grid(True, linestyle=":", alpha=0.6)
    ax8a.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#cbd5e1")

    # Figure 8(b) - Final Width
    ax8b.set_xlabel("cr [°C/s]", fontsize=12)
    ax8b.text(1.0, -0.15, r"$\times 10^{-3}$", transform=ax8b.transAxes, fontsize=11, ha="right")
    ax8b.set_ylabel("⟨L₂⟩ [m]", fontsize=12)
    ax8b.text(0.0, 1.02, r"$\times 10^{-4}$", transform=ax8b.transAxes, fontsize=11)
    ax8b.set_title("b) Final Width ⟨L₂⟩", fontsize=13)
    ax8b.set_xlim(0.5, 8.8)
    ax8b.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
    ax8b.grid(True, linestyle=":", alpha=0.6)
    ax8b.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#cbd5e1")

    fig8.suptitle("Fig. 8: Effects of Cooling Rate and Seeding Temperature on Product Properties (Stored Data)", fontsize=14)
    fig8.tight_layout()
    fig8.savefig(os.path.join(OUTPUT_DIR, "fig8_product_sizes_vs_Ts.png"), dpi=300)
    plt.close(fig8)

    # Figure 9 - Aspect Ratio
    ax9.set_xlabel("cr [°C/s]", fontsize=12)
    ax9.text(1.0, -0.15, r"$\times 10^{-3}$", transform=ax9.transAxes, fontsize=11, ha="right")
    ax9.set_ylabel(r"⟨L₁⟩/⟨L₂⟩", fontsize=12)
    ax9.set_title("Fig. 9: Effects of Cooling Rate and Seeding Temperature on Aspect Ratio", fontsize=13)
    ax9.set_xlim(0.5, 8.8)
    ax9.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
    ax9.grid(True, linestyle=":", alpha=0.6)
    ax9.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#cbd5e1")
    fig9.tight_layout()
    fig9.savefig(os.path.join(OUTPUT_DIR, "fig9_aspect_ratio_vs_Ts.png"), dpi=300)
    plt.close(fig9)


def generate_figure_10():
    """Fig. 10: Dynamic evolution of secondary nucleation rate B(t) across seeding temperatures."""
    print("Generating Figure 10: Time evolution of secondary nucleation rate (Log-Log)...")
    data = load_csv_data("fig10_data.csv")

    t = data["Time_s"]
    configs = [
        {"ts": 29, "label": "Ts = 29 °C", "color": COLOR_BLUE,   "ls": "-"},
        {"ts": 31, "label": "Ts = 31 °C", "color": COLOR_RED,    "ls": "--"},
        {"ts": 33, "label": "Ts = 33 °C", "color": COLOR_GREEN,  "ls": ":"},
        {"ts": 35, "label": "Ts = 35 °C", "color": COLOR_PURPLE, "ls": "-."},
    ]

    plt.figure(figsize=(7.5, 6))

    for cfg in configs:
        ts = cfg["ts"]
        B = np.maximum(data[f"B_Ts{ts}_m3_s"], 1.0e6)
        plt.loglog(t, B, color=cfg["color"], linestyle=cfg["ls"], linewidth=2.0, label=cfg["label"])

    plt.xlabel("Time [s]", fontsize=12)
    plt.ylabel("Nucleation rate [# m⁻³s⁻¹]", fontsize=12)
    plt.title("Fig. 10: Time Evolution of Nucleation Rate with Different Seeding Temperatures", fontsize=13)
    plt.xlim(1.0e-4, 1.2e4)
    plt.ylim(1.0e6, 1.0e12)
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#cbd5e1", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig10_nucleation_rate_evolution.png"), dpi=300)
    plt.close()


def generate_figure_11():
    """Fig. 11: 3D Phase space trajectory in (mu11, S, RV) subspace."""
    print("Generating Figure 11: 3D Phase space trajectory...")
    data = load_csv_data("fig11_data.csv")

    fig = plt.figure(figsize=(7.8, 6.8))
    ax = fig.add_subplot(111, projection="3d")

    configs = [
        {"ts": 29, "label": "Ts = 29 °C", "color": COLOR_BLUE,   "ls": "-"},
        {"ts": 31, "label": "Ts = 31 °C", "color": COLOR_RED,    "ls": "--"},
        {"ts": 33, "label": "Ts = 33 °C", "color": COLOR_GREEN,  "ls": ":"},
        {"ts": 35, "label": "Ts = 35 °C", "color": COLOR_PURPLE, "ls": "-."},
    ]

    for cfg in configs:
        ts = cfg["ts"]
        S = data[f"S_Ts{ts}"]
        # RV display scaling in 10^-5 m^3/s
        RV_scaled = data[f"RV_Ts{ts}"] / 1.0e-5
        mu11_scaled = data[f"mu11_Ts{ts}"] / 1.0e4

        ax.plot(S, RV_scaled, mu11_scaled, color=cfg["color"], linestyle=cfg["ls"], linewidth=2.2, label=cfg["label"])

    ax.set_xlabel("Supersaturation\nRatio - S [-]", labelpad=14, fontsize=11)
    ax.set_xlim(1.0, 2.5)
    ax.set_xticks([1.0, 1.5, 2.0, 2.5])

    ax.set_ylabel("Growth\nrate [m³/s]", labelpad=14, fontsize=11)
    ax.set_ylim(0.0, 1.5)
    ax.set_yticks([0.0, 0.5, 1.0, 1.5])
    ax.text2D(0.85, 0.16, r"$\times 10^{-5}$", transform=ax.transAxes, fontsize=11)

    ax.set_zlabel("μ₁₁ [m²/m³]", labelpad=10, fontsize=11)
    ax.set_zlim(0.0, 2.5)
    ax.set_zticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    ax.text2D(0.08, 0.88, r"$\times 10^{4}$", transform=ax.transAxes, fontsize=11)

    ax.view_init(elev=22, azim=-42)
    ax.set_title("Fig. 11: 3D Trajectory in (μ₁₁, S, Growth rate) Subspace", pad=15, fontsize=13)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#cbd5e1", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig11_3d_phase_trajectory.png"), dpi=300)
    plt.close()


def main():
    print("=" * 80)
    print("REPRODUCING PAPER RESULTS FROM STORED SIMULATION DATA")
    print(f"Data Source Directory : {DATA_DIR}/")
    print(f"Figure Output Directory: {OUTPUT_DIR}/")
    print("=" * 80)

    # Ensure simulation datasets exist; if missing, alert user
    required_files = [
        "fig1_data.csv", "fig3_data.csv", "fig4_fig5_data.csv",
        "fig6_fig7_data.csv", "fig8_fig9_data.csv", "fig10_data.csv", "fig11_data.csv"
    ]
    missing = [f for f in required_files if not os.path.exists(os.path.join(DATA_DIR, f))]
    if missing:
        print(f"Warning: {len(missing)} dataset(s) missing: {missing}")
        print("Generating simulation datasets first...")
        from generate_simulation_data import generate_all_simulation_datasets
        generate_all_simulation_datasets()

    generate_figure_1()
    generate_figure_3()
    generate_figures_4_and_5()
    generate_figures_6_and_7()
    generate_figures_8_and_9()
    generate_figure_10()
    generate_figure_11()

    print("\n" + "=" * 80)
    print("ALL REPRODUCTION FIGURES SUCCESSFULLY GENERATED FROM STORED DATA IN:", OUTPUT_DIR)
    print("=" * 80)


if __name__ == "__main__":
    main()
