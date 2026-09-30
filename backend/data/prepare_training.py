import os
import sys
import argparse
import logging
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.data.era5_loader import ERA5DataLoader
from backend.data.nwp_loader import ProductionNWPLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="StormTrace AI Training Dataset Prep CLI")
    parser.add_argument("--timesteps", type=int, default=9, help="Number of 4D forecast timesteps (default: 9)")
    args = parser.parse_args()

    print("=" * 70)
    print(" StormTrace AI - Training Dataset Preparation CLI")
    print("=" * 70)

    loader = ProductionNWPLoader()
    nwp_data = loader.fetch_live_nwp_ensemble(lat=21.65, lon=88.35, num_members=50)

    print(f" Source: {nwp_data.get('source')}")
    print(f" Status: {nwp_data.get('status')}")
    print(" Training dataset prep finished.")

if __name__ == "__main__":
    main()
