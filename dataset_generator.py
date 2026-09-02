"""
Dataset Generator for Physics-Informed Machine Learning on 2D Crystallization.

Generates batch cooling crystallization trajectories spanning the experimental conditions
investigated in Szilagyi & Lakatos (2015):
- Seeding temperatures: Ts in [29.0, 35.2] °C
- Cooling rates: cr in [0.83e-3, 8.33e-3] °C/s
- Stirring power: epsilon in [200, 550] W/kg
- Small realistic perturbations on seed loading (+/- 10%)

Data format:
- State vector x(t) = [c, T, mu00, mu10, mu01, mu11]  (6 dimensions)
- Input vector u(t) = [cr, epsilon]                    (2 dimensions)

Author: Devan Singh Faujdar
Master Training Project, IIT Kharagpur
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict

import config
from crystallizer_mom import BatchCrystallizerMOM


class CrystallizationDataset(Dataset):
    """PyTorch Dataset for sequence-to-sequence batch crystallization trajectories."""

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        """
        Parameters
        ----------
        sequences : np.ndarray
            Input array of shape (N_samples, seq_len, input_dim).
        targets : np.ndarray
            Target array of shape (N_samples, seq_len, output_dim).
        """
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx]


class DataManager:
    """Handles data generation, normalization, and DataLoader construction."""

    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.simulator = BatchCrystallizerMOM()

        # Normalization scalers (mean, std)
        self.state_mean = None
        self.state_std = None
        self.input_mean = None
        self.input_std = None

    def generate_batch_trajectories(self, num_batches: int = 40, num_steps: int = 100) -> list[Dict[str, np.ndarray]]:
        """Simulates diverse crystallization batches across parameter ranges."""
        trajectories = []

        # Grid and Latin-hypercube-style sampling over operational envelope
        for i in range(num_batches):
            if i == 0:
                # Nominal baseline case
                Ts = config.T_SEED
                cr = config.COOLING_RATE_NOMINAL
                eps = config.EPSILON_NOMINAL
                w_scale = 1.0
            else:
                # Varied process parameters
                Ts = np.random.uniform(29.0, 35.2)
                cr = np.random.uniform(0.83e-3, 8.33e-3)
                eps = np.random.uniform(200.0, 550.0)
                w_scale = np.random.uniform(0.85, 1.15)

            weights = config.INITIAL_WEIGHTS * w_scale
            res = self.simulator.simulate(
                T_seed=Ts,
                T_final=config.T_FINAL,
                cooling_rate=cr,
                epsilon=eps,
                weights_init=weights,
                num_points=num_steps,
            )

            # Package state and inputs
            states = np.column_stack([
                res["concentration"],
                res["temperature"],
                res["mu00"],
                res["mu10"],
                res["mu01"],
                res["mu11"],
            ])
            inputs = np.column_stack([
                np.full(num_steps, cr),
                np.full(num_steps, eps),
            ])

            trajectories.append({
                "time": res["time"],
                "states": states,
                "inputs": inputs,
                "mean_L1": res["mean_L1"],
                "mean_L2": res["mean_L2"],
                "aspect_ratio": res["aspect_ratio"],
                "cooling_rate": cr,
                "epsilon": eps,
                "T_seed": Ts,
            })

        return trajectories

    def create_sliding_windows(
        self,
        trajectories: list[Dict[str, np.ndarray]],
        window_size: int = 25,
        stride: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Converts raw batch trajectories into fixed sliding sequence windows."""
        seq_list = []
        target_list = []

        for traj in trajectories:
            states = traj["states"]
            inputs = traj["inputs"]
            N = len(states)

            # Combined input feature: [states, inputs]
            combined = np.hstack([states, inputs])

            for start in range(0, N - window_size, stride):
                end = start + window_size
                # Input: sequence from t to t + window_size - 1
                seq_in = combined[start:end]
                # Target: sequence of next states from t+1 to t + window_size
                target_out = states[start + 1:end + 1]

                seq_list.append(seq_in)
                target_list.append(target_out)

        return np.array(seq_list, dtype=np.float32), np.array(target_list, dtype=np.float32)

    def prepare_data(
        self,
        num_batches: int = 40,
        window_size: int = 25,
        batch_size: int = 16,
    ) -> Tuple[DataLoader, DataLoader, list[Dict[str, np.ndarray]], Dict[str, np.ndarray]]:
        """
        Generates dataset, fits normalizer on train split, and returns DataLoaders.
        """
        print(f"Simulating {num_batches} crystallization batches...")
        trajectories = self.generate_batch_trajectories(num_batches=num_batches)

        # Train / Validation / Test split: 70% train, 15% val, 15% test
        n_train = int(0.70 * num_batches)
        n_val = int(0.15 * num_batches)

        train_trajs = trajectories[:n_train]
        val_trajs = trajectories[n_train:n_train + n_val]
        test_trajs = trajectories[n_train + n_val:]

        # Compute scalers from training set only
        all_train_states = np.vstack([t["states"] for t in train_trajs])
        all_train_inputs = np.vstack([t["inputs"] for t in train_trajs])

        self.state_mean = np.mean(all_train_states, axis=0)
        self.state_std = np.std(all_train_states, axis=0) + 1.0e-8

        self.input_mean = np.mean(all_train_inputs, axis=0)
        self.input_std = np.std(all_train_inputs, axis=0) + 1.0e-8

        # Create windowed datasets
        X_train, Y_train = self.create_sliding_windows(train_trajs, window_size=window_size)
        X_val, Y_val = self.create_sliding_windows(val_trajs, window_size=window_size)

        # Normalize features
        norm_mean = np.hstack([self.state_mean, self.input_mean])
        norm_std = np.hstack([self.state_std, self.input_std])

        X_train_norm = (X_train - norm_mean) / norm_std
        Y_train_norm = (Y_train - self.state_mean) / self.state_std

        X_val_norm = (X_val - norm_mean) / norm_std
        Y_val_norm = (Y_val - self.state_mean) / self.state_std

        train_dataset = CrystallizationDataset(X_train_norm, Y_train_norm)
        val_dataset = CrystallizationDataset(X_val_norm, Y_val_norm)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        scalers = {
            "state_mean": self.state_mean,
            "state_std": self.state_std,
            "input_mean": self.input_mean,
            "input_std": self.input_std,
            "norm_mean": norm_mean,
            "norm_std": norm_std,
        }

        print(f"Data preparation complete: {len(X_train)} training sequences, {len(X_val)} validation sequences.")
        return train_loader, val_loader, test_trajs, scalers


if __name__ == "__main__":
    dm = DataManager()
    train_loader, val_loader, test_trajs, scalers = dm.prepare_data(num_batches=10)
    for x_batch, y_batch in train_loader:
        print("Batch X shape:", x_batch.shape)
        print("Batch Y shape:", y_batch.shape)
        break
