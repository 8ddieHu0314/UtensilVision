import os
import requests
import urllib.request
import time

# --- CONFIGURATION ---
ACCESS_KEY = 'jpfbiouCpM_kPbDy4_WV4YXSbZfAj5Cs9nxe0RoUD7Y'  # Replace this with your actual Unsplash API key
QUERY = 'spoon'
DOWNLOAD_DIR = 'images/spoons'
NUM_IMAGES = 30  # Max: 30 per page (Unsplash API limit without pagination)
currentPage = 1
numberOfPagesToScrap = 15

# --- SETUP OUTPUT FOLDER ---
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

for i in range(currentPage, numberOfPagesToScrap + 1):
    url = f'https://api.unsplash.com/search/photos?query={QUERY}&per_page={NUM_IMAGES}&page={i}&client_id={ACCESS_KEY}'
    response = requests.get(url)
    data = response.json()


    if 'results' in data:
        for j, result in enumerate(data['results']):
            img_url = result['urls']['regular']
            file_path = os.path.join(DOWNLOAD_DIR, f'spoon_{j + 1 + 30 * (i - 1)}.jpg')

            print(f'Downloading image {j + 1 + 30 * (i - 1)}: {img_url}')
            urllib.request.urlretrieve(img_url, file_path)
        print(f"\n✅ Downloaded {len(data['results'])} images to '{DOWNLOAD_DIR}'")
    else:
        print("❌ Failed to fetch images. Check your API key or internet connection.")
    
    time.sleep(3)