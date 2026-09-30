import os
import sys
import argparse
import logging
from datetime import datetime

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.data.download_copernicus_era5 import CopernicusERA5Downloader, INDIA_DOMAIN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="StormTrace AI ERA5 Data Ingestion CLI")
    parser.add_argument("--dataset", type=str, default="era5", choices=["era5", "gfs", "neps_g"], help="Dataset to download")
    parser.add_argument("--years", type=int, default=1, help="Number of historical years to download")
    parser.add_argument("--lat-min", type=float, default=0.0, help="Min latitude (default: 0.0)")
    parser.add_argument("--lat-max", type=float, default=40.0, help="Max latitude (default: 40.0)")
    parser.add_argument("--lon-min", type=float, default=50.0, help="Min longitude (default: 50.0)")
    parser.add_argument("--lon-max", type=float, default=110.0, help="Max longitude (default: 110.0)")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    print("=" * 70)
    print(" StormTrace AI - Real Data Download CLI")
    print("=" * 70)
    print(f" Dataset requested: {args.dataset}")
    print(f" Domain: Lat [{args.lat_min}°, {args.lat_max}°N], Lon [{args.lon_min}°, {args.lon_max}°E]")
    print(f" Target Years: {args.years}")
    print("=" * 70)

    if not args.yes:
        confirm = input("Confirm download for regional subset? [y/N]: ")
        if confirm.lower() not in ["y", "yes"]:
            print("Download cancelled by user.")
            return

    if args.dataset == "era5":
        downloader = CopernicusERA5Downloader()
        domain = {
            "lat_min": args.lat_min,
            "lat_max": args.lat_max,
            "lon_min": args.lon_min,
            "lon_max": args.lon_max,
        }
        end_year = datetime.now().year - 1
        start_year = max(1994, end_year - args.years + 1)
        res = downloader.download_era5_single_levels(years=list(range(start_year, end_year + 1)), domain=domain)
        print("\nDownload operation completed successfully:")
        print(f" -> Output file / status: {res}")
    else:
        print(f"Dataset {args.dataset} downloader ready via adapter interface.")

if __name__ == "__main__":
    main()
