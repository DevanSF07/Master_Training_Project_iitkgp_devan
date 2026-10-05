"""
Training Script for Forward Physics-Informed Recurrent Neural Network (Forward PIRNN).

Trains the model using composite Data MSE + QMOM Differential Equation Residuals:
    Loss = Loss_data + gamma_phys * Loss_physics

Usage:
    .venv/bin/python3 PIRNN/train_forward_pirnn.py --epochs 60 --gamma_phys 0.10
"""

import os
import argparse
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from PIRNN.dataset import load_dataloaders
from PIRNN.models import CrystallizerRecurrentModel
from PIRNN.qmom_physics import DifferentiableQMOMPhysics


def train_forward_pirnn(
    epochs: int = 60,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    num_layers: int = 2,
    rnn_type: str = "GRU",
    gamma_phys: float = 0.10,
    patience: int = 15,
    checkpoint_dir: str = "PIRNN/checkpoints",
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"--> [Forward PIRNN] Using compute device: {device}")

    # Load dataloaders
    train_loader, val_loader, _, norm_stats = load_dataloaders(batch_size=batch_size)
    print(f"--> Loaded {len(train_loader.dataset)} training windows, {len(val_loader.dataset)} validation windows")

    # Instantiate model and physics operator
    model = CrystallizerRecurrentModel(
        input_dim=9,
        future_input_dim=2,
        target_dim=7,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        rnn_type=rnn_type,
    ).to(device)

    physics = DifferentiableQMOMPhysics().to(device)

    criterion_data = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    s_mean = norm_stats["state_mean"].to(device)
    s_std = norm_stats["state_std"].to(device)
    u_mean = norm_stats["input_mean"].to(device)
    u_std = norm_stats["input_std"].to(device)

    train_losses = []
    val_losses = []
    phys_losses = []

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_checkpoint_path = os.path.join(checkpoint_dir, "forward_pirnn_best.pt")

    start_time = time.time()
    print(f"\n------------- BEGINNING FORWARD PIRNN TRAINING (gamma_phys = {gamma_phys}) -------------")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_data_loss = 0.0
        epoch_phys_loss = 0.0

        for hist, fut_in, targ_inc, _ in train_loader:
            hist = hist.to(device)
            fut_in = fut_in.to(device)
            targ_inc = targ_inc.to(device)

            optimizer.zero_grad()

            # Predict state increments
            pred_inc = model(hist, fut_in)  # [B, H, 7]
            l_data = criterion_data(pred_inc, targ_inc)

            # Reconstruct unnormalized absolute states for physics evaluation:
            curr_state_norm = hist[:, -1, :7]
            curr_state = (curr_state_norm * s_std) + s_mean  # [B, 7]
            states_pred_unnorm = curr_state.unsqueeze(1) + (pred_inc * s_std)  # [B, H, 7]

            # Unnormalize future inputs for physical ODEs:
            inputs_unnorm = (fut_in * u_std) + u_mean  # [B, H, 2]

            # Compute QMOM continuous ODE physics loss
            phys_dict = physics.compute_physics_loss(
                states_pred_unnorm,
                inputs_unnorm,
                curr_state=curr_state,
                dt_sample=60.0,
            )
            l_phys = phys_dict["total_physics_loss"]

            # Combined loss
            total_loss = l_data + gamma_phys * l_phys

            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_data_loss += l_data.item() * hist.size(0)
            epoch_phys_loss += l_phys.item() * hist.size(0)

        scheduler.step()
        train_data_loss = epoch_data_loss / len(train_loader.dataset)
        train_phys_loss = epoch_phys_loss / len(train_loader.dataset)
        train_losses.append(train_data_loss)
        phys_losses.append(train_phys_loss)

        # Validation phase
        model.eval()
        val_epoch_loss = 0.0
        with torch.no_grad():
            for hist, fut_in, targ_inc, _ in val_loader:
                hist = hist.to(device)
                fut_in = fut_in.to(device)
                targ_inc = targ_inc.to(device)

                preds = model(hist, fut_in)
                loss = criterion_data(preds, targ_inc)
                val_epoch_loss += loss.item() * hist.size(0)

        val_loss = val_epoch_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "gamma_phys": gamma_phys,
                "norm_stats": norm_stats,
                "config": {
                    "rnn_type": rnn_type,
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                    "input_dim": 9,
                    "target_dim": 7,
                    "model_type": "Forward_PIRNN",
                },
            }, best_checkpoint_path)
            improved_flag = " [*SAVED BEST*]"
        else:
            patience_counter += 1
            improved_flag = ""

        if epoch % 5 == 0 or epoch == 1 or improved_flag:
            print(f"Epoch [{epoch:3d}/{epochs:3d}] | Data MSE: {train_data_loss:.6f} | Phys Res: {train_phys_loss:.4f} | Val MSE: {val_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e}{improved_flag}")

        if patience_counter >= patience:
            print(f"\n--> Early stopping triggered at epoch {epoch}.")
            break

    elapsed = time.time() - start_time
    print(f"----------------------------------------------------------------------")
    print(f"--> Forward PIRNN training finished in {elapsed:.1f} s!")
    print(f"--> Best Validation Loss: {best_val_loss:.6f} at epoch {best_epoch}")
    print(f"--> Saved best model checkpoint to: {best_checkpoint_path}")

    # Plot loss curves
    loss_plot_path = os.path.join(checkpoint_dir, "forward_pirnn_loss_curve.png")
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(range(1, len(train_losses) + 1), train_losses, label="Data MSE (Train)", color="#1f77b4", linewidth=2)
    ax1.plot(range(1, len(val_losses) + 1), val_losses, label="Data MSE (Val)", color="#d62728", linewidth=2, linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Data MSE Loss", color="#1f77b4")
    ax1.set_yscale("log")
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2 = ax1.twinx()
    ax2.plot(range(1, len(phys_losses) + 1), phys_losses, label="QMOM Physics Residual", color="#2ca02c", linewidth=2, linestyle="-.")
    ax2.set_ylabel("Physics Residual Loss", color="#2ca02c")
    ax2.set_yscale("log")

    plt.title(f"Forward PIRNN Training Loss & Physics Residual Convergence (γ = {gamma_phys})", fontweight="bold")
    plt.tight_layout()
    plt.savefig(loss_plot_path, dpi=180)
    plt.close()
    print(f"--> Saved convergence plot to: {loss_plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Forward PIRNN Model")
    parser.add_argument("--epochs", type=int, default=60, help="Maximum epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of recurrent layers")
    parser.add_argument("--rnn_type", type=str, default="GRU", choices=["GRU", "LSTM", "RNN"], help="Recurrent cell")
    parser.add_argument("--gamma_phys", type=float, default=0.10, help="Physics loss penalty weight")
    parser.add_argument("--patience", type=int, default=15, help="Patience")
    args = parser.parse_args()

    train_forward_pirnn(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        rnn_type=args.rnn_type,
        gamma_phys=args.gamma_phys,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
