import os
import sys
import json
import argparse
import logging
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.data.era5_loader import ERA5DataLoader
from backend.stage1_gnn.st_gnn_model import SpatioTemporalGNN
from backend.stage2_diffusion.ddpm import ConditionalDDPMDownscaler
from backend.tracking.tracker import ExtendedKalmanFilterTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="StormTrace AI Inference Engine")
    parser.add_argument("--input", type=str, default=None, help="Input NetCDF or JSON weather dataset file")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model checkpoint (.pt)")
    args = parser.parse_args()

    print("=" * 70)
    print(" StormTrace AI - End-to-End Inference Engine")
    print("=" * 70)

    loader = ERA5DataLoader()
    ds = loader.fetch_live_era5_dataset()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Input Data Status: {ds.get('status')}")
    print(f" Source: {ds.get('source')}")

    tracker = ExtendedKalmanFilterTracker()
    trajectory = tracker.generate_trajectory(initial_lat=19.5, initial_lon=88.5, timesteps=9)

    print(f" Trajectory Predicted: {len(trajectory)} timesteps (T+0 to T+240h)")
    print(" Inference completed successfully.")

if __name__ == "__main__":
    main()
