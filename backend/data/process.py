import os
import sys
import argparse
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.data.era5_loader import ERA5DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="StormTrace AI Data Processing CLI")
    parser.add_argument("--dataset", type=str, default="era5", help="Dataset to process")
    args = parser.parse_args()

    print("=" * 70)
    print(" StormTrace AI - Data Preprocessing & Grid Processing CLI")
    print("=" * 70)

    loader = ERA5DataLoader()
    ds = loader.fetch_live_era5_dataset()

    print(f" Source: {ds.get('source')}")
    print(f" Status: {ds.get('status')}")
    print(f" Shape: {ds.get('shape')}")
    print(f" Variables: {list(ds.get('variables', {}).keys())}")
    print(" Processing complete.")

if __name__ == "__main__":
    main()
