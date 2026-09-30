import requests
import json
try:
    res = requests.post("http://localhost:8000/api/v1/tracking/predict", json={"lat": 19.5, "lon": 88.5}, timeout=10)
    print("Status Code:", res.status_code)
    print(json.dumps(res.json(), indent=2))
except Exception as e:
    print("Error:", e)
