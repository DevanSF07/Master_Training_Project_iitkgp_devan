"""
Crystallization Dataset Generator: Multi-Profile Synthetic Batch Synthesis.

Generates diverse datasets across multiple temperature profile families, randomized
operating conditions, and realistic sensor noise. Formats data for rolling multi-step
RNN/PIRNN training (history window L -> forecast horizon H) and full-batch evaluation.
"""

import os
import math
import numpy as np
import pandas as pd
import torch
from typing import Dict, Any, List, Tuple, Optional

from .temperature_profiles import sample_random_profile
from .plant_simulator import DynamicCrystallizerPlant, apelblat_solubility


class CrystallizationDatasetGenerator:
    """
    Automated synthesis engine for diverse 2D crystallization batches.
    Ensures strict thermodynamic validity (T >= 25.0 °C), metastable seeding
    supersaturation (sigma_0 in [0.015, 0.040]), and physical plate geometry (L1 > L2).
    """

    def __init__(
        self,
        output_dir: str = "PIRNN/data",
        sampling_interval: float = 60.0,
        history_len: int = 5,
        forecast_horizon: int = 10,
    ):
        self.output_dir = output_dir
        self.dt_sample = float(sampling_interval)
        self.history_len = int(history_len)
        self.forecast_horizon = int(forecast_horizon)
        self.plant = DynamicCrystallizerPlant()
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_batch_dataset(
        self,
        num_batches: int = 60,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Synthesizes num_batches batch runs spanning all profile families.
        """
        rng = np.random.default_rng(seed)
        all_batches: List[Dict[str, Any]] = []
        metadata_rows: List[Dict[str, Any]] = []

        profile_families = ["linear", "cubic", "natural", "two_stage", "cool_hold_cool", "random_spline"]

        print(f"--> Starting synthesis of {num_batches} crystallization batches...")
        for i in range(num_batches):
            p_type = profile_families[i % len(profile_families)]

            # 1. Thermodynamically valid temperature range (strictly monotonic Apelblat regime)
            T_seed = float(rng.uniform(33.0, 35.2))
            T_final = float(rng.uniform(25.0, 27.0))

            # 2. Metastable zone seeding (1.5% to 4.0% initial supersaturation)
            sigma_0 = float(rng.uniform(0.015, 0.040))
            cs_seed = apelblat_solubility(T_seed)
            c0 = float(cs_seed * (1.0 + sigma_0))

            # Batch duration in seconds (between 7200 s and 14400 s, rounded to dt_sample)
            raw_dur = rng.uniform(7200.0, 14400.0)
            duration = math.ceil(raw_dur / self.dt_sample) * self.dt_sample

            # 3. Seed crystal plate geometry: L2 in [35, 55] um, AR in [1.8, 2.2], L1 = L2 * AR
            seed_mass_frac = float(rng.uniform(0.010, 0.030))
            mean_L2_seed = float(rng.uniform(35.0e-6, 55.0e-6))
            ar_seed = float(rng.uniform(1.8, 2.2))
            mean_L1_seed = float(mean_L2_seed * ar_seed)

            # Stirring power and kinetic drift perturbations
            epsilon = float(rng.uniform(200.0, 500.0))
            lam_k1 = float(rng.uniform(0.88, 1.12))
            lam_k2 = float(rng.uniform(0.88, 1.12))
            lam_kS = float(rng.uniform(0.88, 1.12))

            # Build temperature profile
            profile = sample_random_profile(
                T_seed=T_seed,
                T_final=T_final,
                duration=duration,
                profile_type=p_type,
                rng=rng,
            )

            # Simulate clean trajectory with closed bivariate moment differential equations
            traj_clean = self.plant.simulate_batch(
                profile=profile,
                duration=duration,
                dt_sample=self.dt_sample,
                c0=c0,
                seed_mass_fraction=seed_mass_frac,
                mean_L1_seed=mean_L1_seed,
                mean_L2_seed=mean_L2_seed,
                epsilon=epsilon,
                lambda_k1=lam_k1,
                lambda_k2=lam_k2,
                lambda_kS=lam_kS,
            )

            # Add sensor noise (preserving both clean ground-truth and noisy measurements)
            traj = self.plant.inject_sensor_noise(traj_clean, seed=int(rng.integers(1e7)))
            traj["batch_id"] = i
            traj["profile_type"] = p_type
            traj["c0"] = c0
            traj["sigma_0"] = sigma_0
            all_batches.append(traj)

            # Metadata summary
            metadata_rows.append({
                "batch_id": i,
                "profile_type": p_type,
                "T_seed": T_seed,
                "T_final": T_final,
                "c0_kg_m3": c0,
                "sigma_0": sigma_0,
                "duration_s": duration,
                "num_steps": len(traj["time"]),
                "seed_mass_fraction": seed_mass_frac,
                "mean_L1_seed_um": mean_L1_seed * 1e6,
                "mean_L2_seed_um": mean_L2_seed * 1e6,
                "seed_aspect_ratio": ar_seed,
                "epsilon_W_kg": epsilon,
                "lambda_k1": lam_k1,
                "lambda_k2": lam_k2,
                "lambda_kS": lam_kS,
                "final_mean_L1_um": traj["mean_L1"][-1] * 1e6,
                "final_mean_L2_um": traj["mean_L2"][-1] * 1e6,
                "final_aspect_ratio": traj["aspect_ratio"][-1],
                "final_c_kg_m3": traj["c"][-1],
            })

            if (i + 1) % 10 == 0 or (i + 1) == num_batches:
                print(f"    [Batch {i+1:3d}/{num_batches:3d}] Family: {p_type:18s} | Ts={T_seed:.1f}°C -> Tf={T_final:.1f}°C | c0={c0:.1f} kg/m³ (σ0={sigma_0:.3f}) | Final L1={traj['mean_L1'][-1]*1e6:.1f} µm, AR={traj['aspect_ratio'][-1]:.2f}")

        # Partition by batch indices (train / val / test)
        n_train = int(num_batches * train_ratio)
        n_val = int(num_batches * val_ratio)
        train_batches = all_batches[:n_train]
        val_batches = all_batches[n_train : n_train + n_val]
        test_batches = all_batches[n_train + n_val :]

        # Extract rolling windows (computed from train set normalization only)
        norm_stats = self._compute_normalization(train_batches)
        train_windows = self._extract_rolling_windows(train_batches, norm_stats)
        val_windows = self._extract_rolling_windows(val_batches, norm_stats)
        test_windows = self._extract_rolling_windows(test_batches, norm_stats)

        # Save metadata CSV
        df_meta = pd.DataFrame(metadata_rows)
        meta_path = os.path.join(self.output_dir, "batch_metadata.csv")
        df_meta.to_csv(meta_path, index=False)

        # Save PyTorch bundle
        bundle = {
            "train_batches": train_batches,
            "val_batches": val_batches,
            "test_batches": test_batches,
            "train_windows": train_windows,
            "val_windows": val_windows,
            "test_windows": test_windows,
            "norm_stats": norm_stats,
        }

        torch.save({
            "batches": train_batches,
            "windows": train_windows,
            "norm_stats": norm_stats,
        }, os.path.join(self.output_dir, "crystallizer_train.pt"))

        torch.save({
            "batches": val_batches,
            "windows": val_windows,
            "norm_stats": norm_stats,
        }, os.path.join(self.output_dir, "crystallizer_val.pt"))

        torch.save({
            "batches": test_batches,
            "windows": test_windows,
            "norm_stats": norm_stats,
        }, os.path.join(self.output_dir, "crystallizer_test.pt"))

        print(f"--> Dataset generation successfully completed!")
        print(f"    Train: {len(train_batches)} batches ({train_windows['history'].shape[0]} windows)")
        print(f"    Val:   {len(val_batches)} batches ({val_windows['history'].shape[0]} windows)")
        print(f"    Test:  {len(test_batches)} batches ({test_windows['history'].shape[0]} windows)")
        print(f"    Saved artifacts to: {self.output_dir}/")
        return bundle

    def _compute_normalization(self, batches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculates global mean and std for states and inputs based on training batches."""
        all_states = []
        all_inputs = []

        for b in batches:
            N = len(b["time"])
            mu00_log = np.log10(np.maximum(b["mu00_meas"], 1e-5))
            mu11_log = np.log10(np.maximum(b["mu11_meas"], 1e-5))
            
            states_matrix = np.column_stack([
                b["T_meas"],
                b["c_meas"],
                b["mean_L1_meas"],
                b["mean_L2_meas"],
                b["aspect_ratio_meas"],
                mu00_log,
                mu11_log,
            ])
            inputs_matrix = np.column_stack([
                b["cooling_rate"],
                np.full(N, b["epsilon"]),
            ])
            all_states.append(states_matrix)
            all_inputs.append(inputs_matrix)

        all_states = np.vstack(all_states)
        all_inputs = np.vstack(all_inputs)

        state_mean = np.mean(all_states, axis=0)
        state_std = np.std(all_states, axis=0) + 1e-8

        input_mean = np.mean(all_inputs, axis=0)
        input_std = np.std(all_inputs, axis=0) + 1e-8

        state_names = ["T", "c", "mean_L1", "mean_L2", "aspect_ratio", "log_mu00", "log_mu11"]
        input_names = ["cooling_rate", "epsilon"]

        return {
            "state_mean": torch.tensor(state_mean, dtype=torch.float32),
            "state_std": torch.tensor(state_std, dtype=torch.float32),
            "input_mean": torch.tensor(input_mean, dtype=torch.float32),
            "input_std": torch.tensor(input_std, dtype=torch.float32),
            "state_names": state_names,
            "input_names": input_names,
        }

    def _extract_rolling_windows(
        self,
        batches: List[Dict[str, Any]],
        norm_stats: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        """
        Extracts rolling window tensors:
        - history: [B, L, num_states + num_inputs] (past noisy measurements)
        - future_inputs: [B, H, num_inputs] (cooling schedule and stirring)
        - target_increments: [B, H, num_states] (Delta y from y(t-1) in normalized space)
        - target_absolute: [B, H, num_states] (clean ground truth for rigorous evaluation)
        - target_noisy: [B, H, num_states] (noisy sensor observations)
        """
        hist_list = []
        fut_inp_list = []
        target_inc_list = []
        target_abs_list = []
        target_noisy_list = []

        s_mean = norm_stats["state_mean"].numpy()
        s_std = norm_stats["state_std"].numpy()
        u_mean = norm_stats["input_mean"].numpy()
        u_std = norm_stats["input_std"].numpy()

        L = self.history_len
        H = self.forecast_horizon

        for b in batches:
            N = len(b["time"])
            if N < (L + H):
                continue

            # Measured states (with sensor noise)
            mu00_log_meas = np.log10(np.maximum(b["mu00_meas"], 1e-5))
            mu11_log_meas = np.log10(np.maximum(b["mu11_meas"], 1e-5))
            states_meas = np.column_stack([
                b["T_meas"],
                b["c_meas"],
                b["mean_L1_meas"],
                b["mean_L2_meas"],
                b["aspect_ratio_meas"],
                mu00_log_meas,
                mu11_log_meas,
            ])
            states_norm = (states_meas - s_mean) / s_std

            # Clean ground-truth states
            mu00_log_clean = np.log10(np.maximum(b["mu00"], 1e-5))
            mu11_log_clean = np.log10(np.maximum(b["mu11"], 1e-5))
            states_clean = np.column_stack([
                b["T"],
                b["c"],
                b["mean_L1"],
                b["mean_L2"],
                b["aspect_ratio"],
                mu00_log_clean,
                mu11_log_clean,
            ])

            # Process inputs
            inputs_raw = np.column_stack([
                b["cooling_rate"],
                np.full(N, b["epsilon"]),
            ])
            inputs_norm = (inputs_raw - u_mean) / u_std

            # Sliding windows
            for t in range(L, N - H):
                # History: past L steps of [normalized noisy states, normalized inputs]
                h_states = states_norm[t - L : t]
                h_inputs = inputs_norm[t - L : t]
                hist_window = np.hstack([h_states, h_inputs])

                # Future inputs for the next H steps
                f_inputs = inputs_norm[t : t + H]

                # Target increments from current state y(t-1)
                curr_state = states_norm[t - 1]
                future_states = states_norm[t : t + H]
                target_increments = future_states - curr_state

                hist_list.append(hist_window)
                fut_inp_list.append(f_inputs)
                target_inc_list.append(target_increments)
                target_abs_list.append(states_clean[t : t + H])
                target_noisy_list.append(states_meas[t : t + H])

        return {
            "history": torch.tensor(np.array(hist_list), dtype=torch.float32),
            "future_inputs": torch.tensor(np.array(fut_inp_list), dtype=torch.float32),
            "target_increments": torch.tensor(np.array(target_inc_list), dtype=torch.float32),
            "target_absolute": torch.tensor(np.array(target_abs_list), dtype=torch.float32),
            "target_noisy": torch.tensor(np.array(target_noisy_list), dtype=torch.float32),
        }
