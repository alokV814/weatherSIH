import os
import sys
import numpy as np

# Ensure backend directory is in sys.path
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_trigger_alert_api():
    print("Testing /api/v1/alert endpoint...")
    
    # Create a 5x5 dummy DDPM anomaly slice with a clear maximum at (2, 2)
    slice_data = np.zeros((5, 5))
    slice_data[2, 2] = 85.0 # Severe extreme precipitation
    
    payload = {
        "slice": slice_data.tolist(),
        "lat_min": 10.0,
        "lat_max": 20.0,
        "lon_min": 70.0,
        "lon_max": 80.0
    }
    
    response = client.post("/api/v1/alert", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "success"
    assert data["peak_intensity"] == 85.0
    assert data["risk_level"] == "severe"
    
    # Check core coordinate mapping
    # 5 rows -> indices 0..4. Row 2 is exactly in the middle. (10 + 20) / 2 = 15.0
    # 5 cols -> indices 0..4. Col 2 is exactly in the middle. (70 + 80) / 2 = 75.0
    assert abs(data["core_coordinate"][0] - 15.0) < 0.01
    assert abs(data["core_coordinate"][1] - 75.0) < 0.01
    
    # Check GeoJSON
    geojson = data["geojson"]
    assert geojson["type"] == "FeatureCollection"
    feature = geojson["features"][0]
    assert feature["geometry"]["coordinates"] == [75.0, 15.0] # GeoJSON uses [lon, lat]
    assert feature["properties"]["radius_km"] == 5.0
    
    print("Alerting logic tests passed successfully!")

if __name__ == "__main__":
    test_trigger_alert_api()
