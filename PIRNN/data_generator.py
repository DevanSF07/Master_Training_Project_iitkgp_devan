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
from .plant_simulator import DynamicCrystallizerPlant


class CrystallizationDatasetGenerator:
    """
    Automated synthesis engine for diverse 2D crystallization batches.
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

            # Randomized conditions
            T_seed = float(rng.uniform(29.0, 35.0))
            T_final = float(rng.uniform(20.0, min(T_seed - 4.0, 26.0)))
            # Batch duration in seconds (between 7200 s and 16800 s, rounded to dt_sample)
            raw_dur = rng.uniform(7200.0, 16800.0)
            duration = math.ceil(raw_dur / self.dt_sample) * self.dt_sample

            # Seed properties
            seed_mass_frac = float(rng.uniform(0.008, 0.035))
            mean_L1_seed = float(rng.uniform(60.0e-6, 160.0e-6))
            mean_L2_seed = float(rng.uniform(35.0e-6, 95.0e-6))

            # Stirring power and kinetic perturbations (for drift studies)
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

            # Simulate clean trajectory
            traj_clean = self.plant.simulate_batch(
                profile=profile,
                duration=duration,
                dt_sample=self.dt_sample,
                seed_mass_fraction=seed_mass_frac,
                mean_L1_seed=mean_L1_seed,
                mean_L2_seed=mean_L2_seed,
                epsilon=epsilon,
                lambda_k1=lam_k1,
                lambda_k2=lam_k2,
                lambda_kS=lam_kS,
            )

            # Add sensor noise
            traj = self.plant.inject_sensor_noise(traj_clean, seed=int(rng.integers(1e7)))
            traj["batch_id"] = i
            traj["profile_type"] = p_type
            all_batches.append(traj)

            # Metadata summary
            metadata_rows.append({
                "batch_id": i,
                "profile_type": p_type,
                "T_seed": T_seed,
                "T_final": T_final,
                "duration_s": duration,
                "num_steps": len(traj["time"]),
                "seed_mass_fraction": seed_mass_frac,
                "mean_L1_seed_um": mean_L1_seed * 1e6,
                "mean_L2_seed_um": mean_L2_seed * 1e6,
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
                print(f"    [Batch {i+1:3d}/{num_batches:3d}] Family: {p_type:18s} | Ts={T_seed:.1f}°C -> Tf={T_final:.1f}°C | Final L1={traj['mean_L1'][-1]*1e6:.1f} µm, AR={traj['aspect_ratio'][-1]:.2f}")

        # Partition by batch indices
        n_train = int(num_batches * train_ratio)
        n_val = int(num_batches * val_ratio)
        train_batches = all_batches[:n_train]
        val_batches = all_batches[n_train : n_train + n_val]
        test_batches = all_batches[n_train + n_val :]

        # Extract rolling windows (for multi-step ahead RNN training)
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
            "config": {
                "sampling_interval": self.dt_sample,
                "history_len": self.history_len,
                "forecast_horizon": self.forecast_horizon,
                "num_batches": num_batches,
                "n_train": len(train_batches),
                "n_val": len(val_batches),
                "n_test": len(test_batches),
            },
        }

        # Save individual split files
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
        """Calculates global mean and std for states and inputs."""
        all_states = []
        all_inputs = []

        for b in batches:
            N = len(b["time"])
            # State vector: [T, c, mean_L1, mean_L2, aspect_ratio, log10(mu00), log10(mu11)]
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
        - history: [B, L, num_states + num_inputs]
        - future_inputs: [B, H, num_inputs]
        - target_increments: [B, H, num_states] (Delta y = y(t+k) - y(t))
        - ground_truth_states: [B, H, num_states] (absolute states for evaluation)
        """
        hist_list = []
        fut_inp_list = []
        target_inc_list = []
        target_abs_list = []

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

            mu00_log = np.log10(np.maximum(b["mu00_meas"], 1e-5))
            mu11_log = np.log10(np.maximum(b["mu11_meas"], 1e-5))

            # Raw and normalized states
            states_raw = np.column_stack([
                b["T_meas"],
                b["c_meas"],
                b["mean_L1_meas"],
                b["mean_L2_meas"],
                b["aspect_ratio_meas"],
                mu00_log,
                mu11_log,
            ])
            states_norm = (states_raw - s_mean) / s_std

            inputs_raw = np.column_stack([
                b["cooling_rate"],
                np.full(N, b["epsilon"]),
            ])
            inputs_norm = (inputs_raw - u_mean) / u_std

            # Sliding windows
            for t in range(L, N - H):
                # History: past L steps of [normalized states, normalized inputs]
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
                target_abs_list.append(states_raw[t : t + H])

        return {
            "history": torch.tensor(np.array(hist_list), dtype=torch.float32),
            "future_inputs": torch.tensor(np.array(fut_inp_list), dtype=torch.float32),
            "target_increments": torch.tensor(np.array(target_inc_list), dtype=torch.float32),
            "target_absolute": torch.tensor(np.array(target_abs_list), dtype=torch.float32),
        }
