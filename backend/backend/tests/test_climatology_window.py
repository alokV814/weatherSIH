import os
import sys
import numpy as np

# Ensure backend directory is in sys.path
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from data.climatology import RealERA5ClimatologyEngine

def test_climatology_window():
    print("Testing 30-year windowed climatology CDF builder...")
    engine = RealERA5ClimatologyEngine()
    
    # 1. Assert sample years is at least 30
    assert engine.sample_years >= 30, f"Climatology window must be >= 30 years, got {engine.sample_years}"
    
    # 2. Test Lalaurette EFI computation
    np.random.seed(42)
    clim_samples = np.random.gamma(2, 10, 10000)
    
    # Forecast with extreme values
    forecast_extreme = np.random.normal(50, 5, 50) 
    efi_extreme = engine.compute_lalaurette_efi(forecast_extreme, clim_samples)
    print(f"EFI (Extreme Forecast): {efi_extreme:.3f}")
    assert efi_extreme > 0.5, f"Expected high EFI for extreme forecast, got {efi_extreme}"

    # Forecast with normal values
    forecast_normal = np.random.normal(15, 5, 50)
    efi_normal = engine.compute_lalaurette_efi(forecast_normal, clim_samples)
    print(f"EFI (Normal Forecast): {efi_normal:.3f}")
    assert -1.0 <= efi_normal <= 1.0, f"EFI should be bounded [-1, 1], got {efi_normal}"

    print("All tests passed! Climatology window and Lalaurette EFI integral are correctly implemented.")

if __name__ == "__main__":
    test_climatology_window()
