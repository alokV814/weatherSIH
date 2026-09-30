import os
import sys
import json
import argparse
import logging
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.stage2_diffusion.ddpm import ConditionalUNetDownscaler, CosineDDPMScheduler
from backend.stage2_diffusion.physics_loss import physics_informed_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def train_ddpm(epochs=5, lr=1e-3, seed=42, output_dir="runs/ddpm"):
    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConditionalUNetDownscaler(in_channels=1, out_channels=1, time_emb_dim=32).to(device)
    scheduler = CosineDDPMScheduler(timesteps=100)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    logger.info(f"Starting Conditional DDPM Downscaler Training | Device: {device}")

    history = []
    model.train()
    for epoch in range(1, epochs + 1):
        batch_size = 4
        target_5km = torch.ones(batch_size, 1, 64, 64, device=device) * 25.0
        coarse_12km = nn.functional.interpolate(target_5km, size=(32, 32), mode="bilinear")

        u = torch.zeros(batch_size, 1, 64, 64, device=device)
        v = torch.zeros(batch_size, 1, 64, 64, device=device)
        q = torch.ones(batch_size, 1, 64, 64, device=device) * 0.015
        T = torch.ones(batch_size, 1, 64, 64, device=device) * 298.15

        t = torch.randint(0, scheduler.timesteps, (batch_size,), device=device).long()
        noise = torch.randn_like(target_5km)
        x_noisy = scheduler.add_noise(target_5km, noise, t)

        optimizer.zero_grad()
        predicted_noise = model(x_noisy, t)

        mse_loss = nn.MSELoss()(predicted_noise, noise)
        p_loss = physics_informed_loss(x_noisy, coarse_12km, u, v, q, T)
        total_loss = mse_loss + 0.1 * p_loss

        total_loss.backward()
        optimizer.step()

        logger.info(f"Epoch [{epoch}/{epochs}] - Loss: {total_loss.item():.4f} (MSE: {mse_loss.item():.4f}, Phys: {p_loss.item():.4f})")
        history.append({"epoch": epoch, "loss": total_loss.item()})

    ckpt_path = os.path.join(output_dir, "checkpoint.pt")
    metrics_path = os.path.join(output_dir, "metrics.json")
    torch.save(model.state_dict(), ckpt_path)

    with open(metrics_path, "w") as f:
        json.dump({"seed": seed, "epochs": epochs, "final_loss": history[-1]["loss"], "history": history}, f, indent=2)

    logger.info(f"DDPM Downscaler Training complete. Checkpoint saved to: {ckpt_path}")

def main():
    parser = argparse.ArgumentParser(description="StormTrace AI DDPM Downscaler Trainer CLI")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="runs/ddpm")
    args = parser.parse_args()

    train_ddpm(epochs=args.epochs, lr=args.lr, seed=args.seed, output_dir=args.output_dir)

if __name__ == "__main__":
    main()
