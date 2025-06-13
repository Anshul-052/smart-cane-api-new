import requests

# FastAPI server URL
url = "http://127.0.0.1:8000/set_mode"

# Available modes: "indoor", "outdoor", "road"
mode = "outdoor"  # Change this to the desired mode

# Sending request to change mode
response = requests.post(url, json={"mode": mode})

# Print server response
print(response.json())
