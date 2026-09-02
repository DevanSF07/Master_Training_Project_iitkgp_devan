"""
Training Pipeline for Physics-Informed RNN (PI-RNN) vs Black-Box RNN.

Trains both:
1. PI-RNN (Physics-Informed RNN): trained with data loss + physical conservation laws.
2. BB-RNN (Black-Box RNN): trained purely on data loss without physical regularization.

Saves model weights, scalers, and training convergence curves.

Author: Devan Singh Faujdar
Master Training Project, IIT Kharagpur
"""

import os
import time
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

import config
from dataset_generator import DataManager
from pirnn_model import PIRNNModel

CHECKPOINT_DIR = "checkpoints"
PLOT_DIR = "plots/pirnn_comparison"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


def train_single_model(
    model: PIRNNModel,
    train_loader,
    val_loader,
    model_name: str = "PIRNN",
    num_epochs: int = 80,
    lr: float = 1.0e-3,
    device: str = "cpu",
) -> dict:
    """Trains a given recurrent model and returns training logs."""
    print(f"\n---> Training {model_name} on {device.upper()} for {num_epochs} epochs...")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1.0e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1.0e-5)

    history = {
        "train_total": [],
        "train_data": [],
        "train_phys": [],
        "val_loss": [],
    }

    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_tot, epoch_data, epoch_phys = 0.0, 0.0, 0.0
        n_batches = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Denoising regularization: small noise injection on state features to prevent rollout drift
            noise = torch.randn_like(x_batch) * 0.012
            noise[:, :, 6:] = 0.0  # do not add noise to process inputs
            x_input = x_batch + noise

            optimizer.zero_grad()
            tot_loss, d_loss, p_loss = model.compute_loss(x_input, y_batch)
            tot_loss.backward()

            # Gradient clipping for stable training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            epoch_tot += tot_loss.item()
            epoch_data += d_loss.item()
            epoch_phys += p_loss.item()
            n_batches += 1

        scheduler.step()

        # Validation evaluation
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                pred = model(x_val)
                val_loss += torch.nn.functional.mse_loss(pred, y_val).item()
                n_val += 1

        val_loss /= max(n_val, 1)
        train_tot_avg = epoch_tot / max(n_batches, 1)
        train_data_avg = epoch_data / max(n_batches, 1)
        train_phys_avg = epoch_phys / max(n_batches, 1)

        history["train_total"].append(train_tot_avg)
        history["train_data"].append(train_data_avg)
        history["train_phys"].append(train_phys_avg)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"{model_name.lower()}_crystallizer.pth")
            torch.save(model.state_dict(), ckpt_path)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch:3d}/{num_epochs:3d}] | "
                  f"Train Tot: {train_tot_avg:.5f} (Data: {train_data_avg:.5f}, Phys: {train_phys_avg:.5f}) | "
                  f"Val MSE: {val_loss:.5f}")

    elapsed = time.time() - start_time
    print(f"---> {model_name} training finished in {elapsed:.1f}s. Best Val Loss: {best_val_loss:.5f}")
    return history


def plot_training_comparison(pirnn_hist: dict, bbrnn_hist: dict):
    """Plots comparative training curves."""
    epochs = range(1, len(pirnn_hist["train_total"]) + 1)

    plt.figure(figsize=(12, 5))

    # Data / Prediction loss comparison
    plt.subplot(1, 2, 1)
    plt.plot(epochs, bbrnn_hist["train_data"], "r--", label="BB-RNN Train MSE", alpha=0.8)
    plt.plot(epochs, bbrnn_hist["val_loss"], "m-", label="BB-RNN Val MSE", alpha=0.8)
    plt.plot(epochs, pirnn_hist["train_data"], "b-", label="PI-RNN Train MSE", linewidth=2)
    plt.plot(epochs, pirnn_hist["val_loss"], "c-", label="PI-RNN Val MSE", linewidth=2)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Normalized MSE Loss")
    plt.title("Prediction Error (Data Fitting)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)

    # Physical residual evolution
    plt.subplot(1, 2, 2)
    plt.plot(epochs, pirnn_hist["train_phys"], "g-", label="PI-RNN Physics Residual", linewidth=2)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Physical Residual Loss")
    plt.title("Physical Law Residual Convergence")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "training_curves.png"), dpi=300)
    plt.close()
    print("Training comparison plot saved to:", os.path.join(PLOT_DIR, "training_curves.png"))


def main():
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate crystallization dataset
    dm = DataManager(seed=42)
    train_loader, val_loader, test_trajs, scalers = dm.prepare_data(num_batches=35, batch_size=16)

    # Save scalers for inference
    np.savez(
        os.path.join(CHECKPOINT_DIR, "scalers.npz"),
        state_mean=scalers["state_mean"],
        state_std=scalers["state_std"],
        input_mean=scalers["input_mean"],
        input_std=scalers["input_std"],
    )

    # 1. Instantiate and train Physics-Informed RNN
    pirnn = PIRNNModel(scalers=scalers, physics_weight=config.PI_RNN_CONFIG["physics_weight"])
    pirnn_history = train_single_model(
        pirnn,
        train_loader,
        val_loader,
        model_name="PIRNN",
        num_epochs=config.PI_RNN_CONFIG["num_epochs"],
        lr=config.PI_RNN_CONFIG["learning_rate"],
        device=device,
    )

    # 2. Instantiate and train Black-Box RNN
    bbrnn = PIRNNModel(scalers=scalers, physics_weight=0.0)
    bbrnn_history = train_single_model(
        bbrnn,
        train_loader,
        val_loader,
        model_name="BBRNN",
        num_epochs=config.PI_RNN_CONFIG["num_epochs"],
        lr=config.PI_RNN_CONFIG["learning_rate"],
        device=device,
    )

    # Plot convergence comparison
    plot_training_comparison(pirnn_history, bbrnn_history)


if __name__ == "__main__":
    main()
