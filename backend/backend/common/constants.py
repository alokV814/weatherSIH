"""
Domain and Physical Constants for  Extreme Weather Tracking Pipeline.
"""

# Meteorological & Physical Domain Defaults
INDIA_DOMAIN_LAT = (6.0, 38.0)
INDIA_DOMAIN_LON = (68.0, 98.0)
DEFAULT_SPATIAL_RES_DEG = 0.12  # ~12 km spatial resolution
KM_PER_DEG_LAT = 111.0  # Approximated km per degree latitude

# Downscaling Parameters
COARSE_RES_KM = 12.0
FINE_RES_KM = 5.0
DOWNSCALE_SCALE_FACTOR = COARSE_RES_KM / FINE_RES_KM  # 2.4

# Extreme Event Thresholds
DEFAULT_RAIN_THRESHOLD_MM_24H = 50.0  # Heavy rainfall threshold
DEFAULT_EFI_ANOMALY_THRESHOLD = 0.65  # Upper quantile extreme threshold
