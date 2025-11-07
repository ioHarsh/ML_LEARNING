import requests
import json

# URL of your local FastAPI endpoint
url = "http://127.0.0.1:8000/predict"

# Example input data (same format as app.py expects)
# Here we’re using one Iris flower sample (4 features)
data = {
    "data": [[5.1, 3.5, 1.4, 0.2]]
}

# Send POST request
response = requests.post(url, json=data)

# Check the response
if response.status_code == 200:
    print("✅ Prediction successful!")
    print("Response:", json.dumps(response.json(), indent=2))
else:
    print("❌ Error:", response.status_code, response.text)
