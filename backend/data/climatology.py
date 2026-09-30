import os
import sys
import argparse
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.data.climatology import RealERA5ClimatologyEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="StormTrace AI ERA5 Climatology Baseline CLI")
    parser.add_argument("--sample-years", type=int, default=30, help="Number of baseline years (default: 30)")
    args = parser.parse_args()

    print("=" * 70)
    print(" StormTrace AI - 30-Year Real ERA5 Climatology Baseline Builder")
    print("=" * 70)

    engine = RealERA5ClimatologyEngine(sample_years=args.sample_years)
    baseline = engine.fetch_real_era5_climatology()

    print("\nClimatology Baseline Summary:")
    print(f" Period: {baseline['period']}")
    print(f" Source: {baseline['dataSource']}")
    print(f" Samples: {baseline['sampleCount']:,}")
    print(" Quantiles (mm):")
    for q_name, val in baseline['quantiles'].items():
        print(f"   - {q_name}: {val} mm")

if __name__ == "__main__":
    main()
