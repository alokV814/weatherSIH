import os
import sys
import torch
import numpy as np

# Set PYTHONPATH to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.data.fetch_real_weather_archive import RealWeatherArchiveDownloader
from backend.stage1_gnn.st_gnn_model import train_st_gnn_model
from backend.stage2_diffusion.ddpm import train_ddpm_model

def execute_full_real_training_pipeline(epochs_gnn=20, epochs_ddpm=20):
    print("=" * 80)
    print(" StormTrace AI - Full Real-Dataset Model Training Pipeline  ")
    print("=" * 80)

    # 1. Download & Prepare Real Weather Atmospheric Dataset
    print("\n[Phase 1] Downloading & Caching Real ERA5 / Open-Meteo Weather Dataset Tensors...")
    downloader = RealWeatherArchiveDownloader()
    downloader.download_real_historical_event()
    real_ds_path = downloader.build_real_training_dataset(num_samples=150)
    print(f"   -> Real Atmospheric Feature Tensors saved to: {real_ds_path}")

    # 2. Train PyTorch Multi-Head Spherical Graph Attention + Temporal Transformer (ST-GNN)
    print(f"\n[Phase 2] Training PyTorch ST-GNN Model on Real Weather Dataset ({epochs_gnn} Epochs)...")
    gnn_result = train_st_gnn_model(epochs=epochs_gnn, lr=1e-3)
    print(f"   -> ST-GNN Real Checkpoint Saved: {gnn_result['checkpoint_path']}")
    print(f"   -> Final Trajectory Loss: {gnn_result['final_loss']:.4f}")

    # 3. Train PyTorch Conditional UNet Diffusion Super-Resolution Model (DDPM)
    print(f"\n[Phase 3] Training PyTorch Conditional DDPM Downscaler with 5 Physics Laws ({epochs_ddpm} Epochs)...")
    ddpm_result = train_ddpm_model(epochs=epochs_ddpm, batch_size=4, lr=1e-3)
    print(f"   -> DDPM Real Checkpoint Saved: {ddpm_result['checkpoint_path']}")
    print(f"   -> Final Diffusion + Physics Loss: {ddpm_result['final_loss']:.4f}")

    print("\n" + "=" * 80)
    print(" [SUCCESS] Full PyTorch Real Model Training Pipeline Completed Successfully.")
    print("=" * 80)

    return {
        "status": "trained_full_real_models",
        "gnn_checkpoint": gnn_result['checkpoint_path'],
        "gnn_loss": gnn_result['final_loss'],
        "ddpm_checkpoint": ddpm_result['checkpoint_path'],
        "ddpm_loss": ddpm_result['final_loss']
    }

if __name__ == "__main__":
    execute_full_real_training_pipeline(epochs_gnn=15, epochs_ddpm=15)
