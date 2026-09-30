import urllib.request
import json

url = "https://ensemble-api.open-meteo.com/v1/ensemble?latitude=19.5&longitude=88.5&hourly=precipitation,wind_speed_10m,wind_direction_10m,temperature_2m&models=gfs_seamless"
req = urllib.request.Request(url, headers={'User-Agent': 'StormTrace-Testing/1.0'})
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode())
        keys = list(data.get('hourly', {}).keys())
        print(f"Success! Keys found: {len(keys)}")
        print(f"Sample keys: {keys[:10]}")
except Exception as e:
    print(f"Error: {e}")
