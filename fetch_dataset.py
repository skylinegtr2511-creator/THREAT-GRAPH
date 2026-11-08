import requests
import json
import os

KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
DATA_DIR = "data"

os.makedirs(DATA_DIR, exist_ok=True)

print("Fetching CISA KEV dataset...")
response = requests.get(KEV_URL)

if response.status_code == 200:
    file_path = os.path.join(DATA_DIR, "cisa_kev.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(response.json(), f, indent=4)
    
    vuln_count = response.json().get("count", 0)
    print(f"Success! {vuln_count} vulnerabilities saved to {file_path}")
else:
    print(f"Failed to fetch dataset. HTTP Status: {response.status_code}") 