"""
StormTrace AI - Synthetic NWP Weather Tensor Generator 
Produces canonical 5D weather tensors matching production schema:
[ensemble, time, variable, latitude, longitude]
"""
import numpy as np
from backend.data.schema import CANONICAL_VARIABLES, DEFAULT_DOMAIN

import os

STORMTRACE_MODE = os.getenv("STORMTRACE_MODE", "REAL")

def generate_synthetic_nwp_tensor(
    members: int = 50,
    timesteps: int = 9,
    lat_points: int = 30,
    lon_points: int = 30,
    seed: int = 42
) -> dict:
    """
    Generates synthetic 5D NWP forecast ensemble tensor for DEMO mode testing only.
    Strictly blocked in REAL mode.
    """
    if STORMTRACE_MODE == "REAL":
        raise RuntimeError(
            "? REAL DATA ENFORCEMENT FAILURE: STORMTRACE_MODE is set to 'REAL'. "
            "Synthetic weather tensor generation is prohibited in REAL mode. "
            "Please use real dataset adapters in backend/data/adapters/."
        )
    np.random.seed(seed)
    num_vars = len(CANONICAL_VARIABLES)
    
    # Base spatial grid with cyclone structure
    lats = np.linspace(DEFAULT_DOMAIN["lat_min"], DEFAULT_DOMAIN["lat_max"], lat_points)
    lons = np.linspace(DEFAULT_DOMAIN["lon_min"], DEFAULT_DOMAIN["lon_max"], lon_points)
    grid_lat, grid_lon = np.meshgrid(lats, lons, indexing='ij')

    tensor = np.zeros((members, timesteps, num_vars, lat_points, lon_points), dtype=np.float32)

    for t in range(timesteps):
        # Moving cyclone centroid
        cyc_lat = 18.0 + (t * 0.8)
        cyc_lon = 88.0 + (t * 0.4)
        dist = np.sqrt((grid_lat - cyc_lat)**2 + (grid_lon - cyc_lon)**2)
        precip_core = 180.0 * np.exp(-(dist / 2.5)**2)

        for m in range(members):
            pert = np.random.normal(1.0, 0.12, (lat_points, lon_points))
            # Variable 0: Precipitation (mm/day)
            tensor[m, t, 0] = np.clip((precip_core + np.random.gamma(2.0, 10.0, (lat_points, lon_points))) * pert, 0, None)
            # Variable 1: Temperature (K)
            tensor[m, t, 1] = 298.15 + np.random.normal(0, 3.0, (lat_points, lon_points))
            # Variable 2: U10 Wind (m/s)
            tensor[m, t, 2] = np.random.normal(5.0, 12.0, (lat_points, lon_points))
            # Variable 3: V10 Wind (m/s)
            tensor[m, t, 3] = np.random.normal(8.0, 14.0, (lat_points, lon_points))
            # Variable 4: Surface Pressure (hPa)
            tensor[m, t, 4] = 1013.25 - (precip_core * 0.15) + np.random.normal(0, 2.0, (lat_points, lon_points))
            # Variable 5: Relative Humidity (%)
            tensor[m, t, 5] = np.clip(70.0 + (precip_core * 0.2) + np.random.normal(0, 5.0, (lat_points, lon_points)), 0, 100)

    return {
        "tensor": tensor,
        "shape": list(tensor.shape),
        "variables": CANONICAL_VARIABLES,
        "members": members,
        "timesteps": timesteps,
        "lats": lats.tolist(),
        "lons": lons.tolist()
    }

if __name__ == "__main__":
    res = generate_synthetic_nwp_tensor(members=50, timesteps=9)
    print("Generated 5D Weather Tensor Shape:", res["shape"])
