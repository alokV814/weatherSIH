import os
import json
import urllib.request
import logging
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class GEFSOpenMeteoAdapter:
    """
    NOAA GEFS Seamless Ensemble Adapter via Open-Meteo API.
    Provides 31-member ensemble probabilistic forecasts for tracking extremes.
    """
    def __init__(self, raw_dir: str = None):
        self.raw_dir = raw_dir or os.path.join("data", "raw", "gefs")
        os.makedirs(self.raw_dir, exist_ok=True)
        self.num_members = 31

    def fetch_ensemble_data(self, lat: float, lon: float):
        """
        Fetches multi-member time series for the core centroid.
        """
        cache_path = os.path.join(self.raw_dir, f"gefs_31member_live_{lat}_{lon}.json")
        
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                data = json.load(f)
            return self._parse_ensemble(data, lat, lon)
            
        url = (
            f"https://ensemble-api.open-meteo.com/v1/ensemble?"
            f"latitude={lat}&longitude={lon}&"
            f"hourly=precipitation,wind_speed_10m,wind_direction_10m,temperature_2m&"
            f"models=gfs_seamless"
        )
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'StormTrace-GEFS/1.0'})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
                with open(cache_path, "w") as f:
                    json.dump(data, f)
                return self._parse_ensemble(data, lat, lon)
        except Exception as e:
            logger.error(f"Error fetching GEFS Open-Meteo ensemble: {e}")
            return None

    def _parse_ensemble(self, data, origin_lat, origin_lon):
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        
        # Identify members
        members_precip = [k for k in hourly.keys() if k.startswith("precipitation_member")]
        num_avail = len(members_precip)
        
        members_data = []
        for i in range(1, num_avail + 1):
            pm = hourly.get(f"precipitation_member{i:02d}", [0]*len(times))
            wm = hourly.get(f"wind_speed_10m_member{i:02d}", [0]*len(times))
            dm = hourly.get(f"wind_direction_10m_member{i:02d}", [0]*len(times))
            tm = hourly.get(f"temperature_2m_member{i:02d}", [0]*len(times))
            
            members_data.append({
                "member_id": i,
                "precipitation": [x if x is not None else 0 for x in pm],
                "wind_speed_10m": [x if x is not None else 0 for x in wm],
                "wind_direction_10m": [x if x is not None else 0 for x in dm],
                "temperature_2m": [x if x is not None else 0 for x in tm]
            })
            
        return {
            "source": "NOAA GEFS Seamless (via Open-Meteo)",
            "lat": origin_lat,
            "lon": origin_lon,
            "times": times,
            "num_members": num_avail,
            "members": members_data,
            "status": "VALID_REAL_GEFS"
        }
