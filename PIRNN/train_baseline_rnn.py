"""
Training Script for Pure Data-Driven Recurrent Baseline Model.

Usage:
    .venv/bin/python3 PIRNN/train_baseline_rnn.py --epochs 60 --rnn_type GRU
"""

import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from PIRNN.dataset import load_dataloaders
from PIRNN.models import CrystallizerRecurrentModel


def train_baseline_model(
    epochs: int = 60,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    num_layers: int = 2,
    rnn_type: str = "GRU",
    patience: int = 15,
    checkpoint_dir: str = "PIRNN/checkpoints",
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"--> Using compute device: {device}")

    # Load data
    train_loader, val_loader, test_loader, norm_stats = load_dataloaders(batch_size=batch_size)
    print(f"--> Loaded datasets: {len(train_loader.dataset)} train windows, {len(val_loader.dataset)} val windows")

    # Instantiate model
    model = CrystallizerRecurrentModel(
        input_dim=9,
        future_input_dim=2,
        target_dim=7,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        rnn_type=rnn_type,
    ).to(device)

    print(f"--> Initialized {rnn_type} baseline model ({sum(p.numel() for p in model.parameters())} parameters)")

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_checkpoint_path = os.path.join(checkpoint_dir, f"baseline_{rnn_type.lower()}_best.pt")

    start_time = time.time()
    print("\n----------------- BEGINNING TRAINING -----------------")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for hist, fut_in, targ_inc, _ in train_loader:
            hist = hist.to(device)
            fut_in = fut_in.to(device)
            targ_inc = targ_inc.to(device)

            optimizer.zero_grad()
            preds = model(hist, fut_in)
            loss = criterion(preds, targ_inc)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * hist.size(0)

        scheduler.step()
        train_loss = epoch_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_epoch_loss = 0.0
        with torch.no_grad():
            for hist, fut_in, targ_inc, _ in val_loader:
                hist = hist.to(device)
                fut_in = fut_in.to(device)
                targ_inc = targ_inc.to(device)

                preds = model(hist, fut_in)
                loss = criterion(preds, targ_inc)
                val_epoch_loss += loss.item() * hist.size(0)

        val_loss = val_epoch_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        # Early stopping and checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "norm_stats": norm_stats,
                "config": {
                    "rnn_type": rnn_type,
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                    "input_dim": 9,
                    "target_dim": 7,
                },
            }, best_checkpoint_path)
            improved_flag = " [*SAVED BEST*]"
        else:
            patience_counter += 1
            improved_flag = ""

        if epoch % 5 == 0 or epoch == 1 or improved_flag:
            print(f"Epoch [{epoch:3d}/{epochs:3d}] | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e}{improved_flag}")

        if patience_counter >= patience:
            print(f"\n--> Early stopping triggered at epoch {epoch} (no validation improvement for {patience} epochs).")
            break

    elapsed = time.time() - start_time
    print(f"------------------------------------------------------")
    print(f"--> Training completed in {elapsed:.1f} s!")
    print(f"--> Best Validation Loss: {best_val_loss:.6f} at epoch {best_epoch}")
    print(f"--> Best model checkpoint saved to: {best_checkpoint_path}")

    # Plot training loss curve
    loss_plot_path = os.path.join(checkpoint_dir, f"baseline_{rnn_type.lower()}_loss_curve.png")
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss (MSE)", color="#1f77b4", linewidth=2)
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Val Loss (MSE)", color="#d62728", linewidth=2, linestyle="--")
    plt.axvline(best_epoch, color="gray", linestyle=":", label=f"Best Epoch ({best_epoch})")
    plt.title(f"Baseline Data-Driven {rnn_type} Training & Validation Loss", fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (Normalized Increments)")
    plt.yscale("log")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_plot_path, dpi=180)
    plt.close()
    print(f"--> Loss curves saved to: {loss_plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Baseline Recurrent Crystallizer Model")
    parser.add_argument("--epochs", type=int, default=60, help="Maximum epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of recurrent layers")
    parser.add_argument("--rnn_type", type=str, default="GRU", choices=["GRU", "LSTM", "RNN"], help="Recurrent cell type")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    args = parser.parse_args()

    train_baseline_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        rnn_type=args.rnn_type,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
