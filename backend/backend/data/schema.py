"""
StormTrace AI - Canonical Meteorological Weather Tensor Schema 
Tensor Shape: [Ensemble, Time, Variable, Latitude, Longitude] -> [E, T, V, Y, X]
"""
import numpy as np

CANONICAL_VARIABLES = [
    "precipitation",
    "temperature",
    "u10_wind",
    "v10_wind",
    "surface_pressure",
    "relative_humidity"
]

DEFAULT_DOMAIN = {
    "lat_min": 6.0,
    "lat_max": 38.0,
    "lon_min": 68.0,
    "lon_max": 98.0,
    "spatial_resolution_km": 12.0,
    "grid_shape": (30, 30)
}

def validate_weather_tensor(tensor: np.ndarray) -> bool:
    """
    Validates canonical 5D weather tensor [E, T, V, Y, X]
    """
    if not isinstance(tensor, np.ndarray):
        return False
    if tensor.ndim != 5:
        return False
    if tensor.shape[2] != len(CANONICAL_VARIABLES):
        return False
    return True
