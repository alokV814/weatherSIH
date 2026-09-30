import math
import numpy as np
from backend.data.adapters.gefs_openmeteo import GEFSOpenMeteoAdapter

class NWPEnsembleTracker:
    def __init__(self, adapter=None):
        self.adapter = adapter or GEFSOpenMeteoAdapter()

    def generate_tracks(self, origin_lat, origin_lon, threshold_mm=50.0):
        # Fetch ensemble data for origin
        data = self.adapter.fetch_ensemble_data(origin_lat, origin_lon)
        if not data or data["num_members"] == 0:
            return None
            
        times = data["times"]
        members = data["members"]
        
        member_tracks = []
        for mem in members:
            # We will generate a track at 6-hour intervals up to 120 hours
            track = []
            curr_lat = origin_lat
            curr_lon = origin_lon
            
            for h in range(0, min(121, len(times)), 6):
                if h > 0:
                    ws = mem["wind_speed_10m"][h-6]
                    wd = mem["wind_direction_10m"][h-6]
                    rad = math.radians(wd)
                    # Convert wind speed km/h to degrees per 6 hours approximately
                    # 1 degree lat = ~111 km
                    u = -ws * math.sin(rad) * 6.0 / 111.0
                    v = -ws * math.cos(rad) * 6.0 / 111.0
                    curr_lon += u
                    curr_lat += v
                    
                precip = sum(mem["precipitation"][max(0, h-6):h]) if h > 0 else 0
                track.append({
                    "hour": h,
                    "lat": round(curr_lat, 4),
                    "lon": round(curr_lon, 4),
                    "precip_6h": precip
                })
            member_tracks.append({"member_id": mem["member_id"], "path": track})
            
        # Ensemble Mean Track & Spread
        ensemble_mean = []
        spread_cone = []
        exceedance_prob = []
        
        num_steps = len(member_tracks[0]["path"])
        for step_idx in range(num_steps):
            lats = [m["path"][step_idx]["lat"] for m in member_tracks]
            lons = [m["path"][step_idx]["lon"] for m in member_tracks]
            precips = [m["path"][step_idx]["precip_6h"] for m in member_tracks]
            
            mean_lat = np.mean(lats)
            mean_lon = np.mean(lons)
            std_lat = np.std(lats)
            std_lon = np.std(lons)
            radius_deg = max(0.1, np.sqrt(std_lat**2 + std_lon**2))
            
            exc_prob = np.mean([1 if p >= threshold_mm else 0 for p in precips]) * 100
            
            hr = member_tracks[0]["path"][step_idx]["hour"]
            ensemble_mean.append({
                "hour": hr,
                "lat": round(mean_lat, 4),
                "lon": round(mean_lon, 4),
                "intensity_mm": round(float(np.mean(precips)), 1)
            })
            spread_cone.append({
                "hour": hr,
                "radius_km": round(radius_deg * 111.0, 1)
            })
            exceedance_prob.append({
                "hour": hr,
                "prob_pct": round(exc_prob, 1)
            })
            
        return {
            "source": data["source"],
            "num_members": data["num_members"],
            "member_tracks": member_tracks,
            "ensemble_mean": ensemble_mean,
            "spread_cone": spread_cone,
            "exceedance_prob": exceedance_prob
        }
