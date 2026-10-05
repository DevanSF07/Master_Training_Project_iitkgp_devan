"""
PyTorch Dataset and DataLoader Utilities for Crystallization Time-Series.

Provides:
- CrystallizerWindowDataset: PyTorch Dataset for rolling-window training
- Extraction of both 7-state physical representations and 11-moment representations
- Tensor normalization and inverse scaling utilities
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, Tuple, Optional


class CrystallizerWindowDataset(Dataset):
    """
    PyTorch Dataset yielding rolling multi-step prediction samples:
    - history: [L, num_features]
    - future_inputs: [H, num_inputs]
    - target_increments: [H, num_targets] (Delta y = y(t+k) - y(t))
    - target_absolute: [H, num_targets] (raw ground truth for evaluation)
    """

    def __init__(self, windows_dict: Dict[str, torch.Tensor]):
        self.history = windows_dict["history"]
        self.future_inputs = windows_dict["future_inputs"]
        self.target_increments = windows_dict["target_increments"]
        self.target_absolute = windows_dict["target_absolute"]

    def __len__(self) -> int:
        return self.history.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.history[idx],
            self.future_inputs[idx],
            self.target_increments[idx],
            self.target_absolute[idx],
        )


def load_dataloaders(
    data_dir: str = "PIRNN/data",
    batch_size: int = 64,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """Loads Train, Validation, and Test DataLoaders alongside normalization statistics."""
    train_data = torch.load(os.path.join(data_dir, "crystallizer_train.pt"), weights_only=False)
    val_data = torch.load(os.path.join(data_dir, "crystallizer_val.pt"), weights_only=False)
    test_data = torch.load(os.path.join(data_dir, "crystallizer_test.pt"), weights_only=False)

    train_ds = CrystallizerWindowDataset(train_data["windows"])
    val_ds = CrystallizerWindowDataset(val_data["windows"])
    test_ds = CrystallizerWindowDataset(test_data["windows"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_data["norm_stats"]
