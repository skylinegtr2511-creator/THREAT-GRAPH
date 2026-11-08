import requests
import os

# URLs for the pre-partitioned UNSW-NB15 dataset
TRAIN_URL = "https://www.unsw.adfa.edu.au/unsw-clite/cyber/dataset/UNSW-NB15/UNSW_NB15_training-set.csv"
TEST_URL = "https://www.unsw.adfa.edu.au/unsw-clite/cyber/dataset/UNSW-NB15/UNSW_NB15_testing-set.csv"
DATA_DIR = "data"

os.makedirs(DATA_DIR, exist_ok=True)

def download_file(url, filename):
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(os.path.join(DATA_DIR, filename), 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully saved {filename}")
    else:
        print(f"Failed to download {filename}. HTTP Status: {response.status_code}")

# Note: The official academic servers can sometimes be slow or have changed routing.
# If these fail, we can pull them from a Kaggle API or mirror.
download_file(TRAIN_URL, "unsw_train.csv")
download_file(TEST_URL, "unsw_test.csv")